from __future__ import annotations

import re

from vimedrag.prompts import render_chatml

_STOP = re.compile(r"<\|im_end\|>|<\|im_start\|>")
_FIRST_DOC = re.compile(r"^\[1\][^\n]*\n(.*)$", re.MULTILINE)
_SENT_END = re.compile(r"(?<=[.!?])\s")


def clean_output(text: str) -> str:
    return _STOP.split(text, maxsplit=1)[0].strip()


def _needs_bos(prompt: str, bos_token: str | None) -> bool:
    """True when the tokenizer should add BOS itself (prompt does not already start with it)."""
    return not (bos_token and prompt.startswith(bos_token))


class FakeGenerator:
    """Deterministic CPU generator: answers with the first sentence of document [1]."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, messages: list[dict]) -> str:
        user = messages[-1]["content"]
        if not user.startswith("Tài liệu:"):
            return ""
        m = _FIRST_DOC.search(user)
        if not m:
            return ""
        return _SENT_END.split(m.group(1), maxsplit=1)[0].strip()


class VinaLLaMAGenerator:
    def __init__(self, model_id: str, max_new_tokens: int = 256, repetition_penalty: float = 1.1,
                 load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.tok = AutoTokenizer.from_pretrained(model_id)
        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto")
        self.model.eval()
        eos = [self.tok.eos_token_id]
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != self.tok.unk_token_id:
            eos.append(im_end)
        self.eos_ids = [e for e in eos if e is not None]
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def _prompt(self, messages: list[dict]) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return render_chatml(messages)

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages: list[dict]) -> str:
        import torch

        prompt = self._prompt(messages)
        enc = self.tok(prompt, return_tensors="pt",
                       add_special_tokens=_needs_bos(prompt, self.tok.bos_token)).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.eos_ids,
                pad_token_id=self.pad_id,
            )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return clean_output(self.tok.decode(new_tokens, skip_special_tokens=True))
