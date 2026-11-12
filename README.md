# vimedrag — Baseline RAG cho ViMedAQA

Pipeline RAG thông thường: **BGE-M3 (dense)** → top-k đoạn → **VinaLLaMA-7B-chat (4-bit)** → câu trả lời.
Đánh giá trên 200 câu validation của `tmnam20/ViMedAQA` (50 câu/topic, seed 42).

## Thí nghiệm
| ID | Lệnh | Ý nghĩa |
|---|---|---|
| E0 | `--mode closed` | Không retrieval (cận dưới) |
| E1 | `--mode rag --top-k 3` | Baseline RAG |
| E2 | `--mode oracle` | Đưa context gốc (cận trên của generator) |
| E3 | `--mode rerank --top-k 3 --candidates-from runs/rag_k3_vinallama7b/predictions.jsonl` | Dùng lại top-10 BGE-M3 của E1, rerank bằng `BAAI/bge-reranker-v2-m3`, đưa top-3 mới vào cùng prompt |

## Chạy local (CPU, kiểm tra luồng)
```bash
python -m pytest -q
python scripts/build_index.py --dry-run --fake-embedder --index-dir artifacts/index_dry_fake
python scripts/run_eval.py --mode rag --dry-run --fake-embedder --index-dir artifacts/index_dry_fake --runs-dir runs_dry_fake
python scripts/summarize.py runs_dry_fake
```

## Chạy trên Kaggle
1. Push repo này lên GitHub (public, hoặc private + Kaggle secret).
2. Kaggle → New Notebook → File → Import Notebook → chọn `notebooks/kaggle_run.ipynb` từ GitHub.
3. Settings: GPU T4, Internet On. Sửa `REPO_URL` ở cell 3. Run All.
4. Lần đầu build index ~10–20 phút; lưu `/kaggle/working/index` thành Dataset `vimedrag-index` để dùng lại.
5. Notebook có cell smoke test (20 câu, `--exp-name smoke_rag`, ~5 phút) trước E0: chạy để kiểm tra retrieval, prompt và stop token trước khi chạy full. Khi nạp VinaLLaMA, generator in ra đường dẫn prompt (`chat_template` hay `render_chatml`), `eos_ids` và id của `<|im_end|>`; câu đầu tiên in nguyên prompt đã render.
6. (Tuỳ chọn) *Add Input* BGE-M3 (Models/Datasets) để khỏi tải lại: cell 6 tự tìm thư mục chứa `config.json` + `tokenizer_config.json` có tên `bge-m3` và truyền `--embed-model` cho build/smoke/E1; không có thì tải `BAAI/bge-m3`. Index build bằng tên HF dùng được với bản Input và ngược lại (cùng model).
7. Nếu 3 câu đầu đều lỗi, pipeline dừng sớm (RuntimeError "dừng sớm") — đọc lỗi, sửa, chạy lại cùng lệnh (resume).

## Output
`runs/<run_name>/` gồm `config.json`, `predictions.jsonl` (resume được), `metrics.json`.
`python scripts/summarize.py runs` in bảng so sánh.

## Metric
- Retrieval (E1): Recall@1/3/5, MRR@10 so với context gốc. Là **cận dưới** vì có context gần trùng.
- Answer: ROUGE-L và token-F1 trên âm tiết (tự cài, không dùng tokenizer của `rouge-score` vì nó xoá dấu tiếng Việt), BERTScore (`lang="vi"`), tỉ lệ từ chối, tỉ lệ lỗi.

## Ghi chú thiết kế
- BGE-M3 dense = CLS + chuẩn hoá L2 qua `transformers`; tìm kiếm exact bằng numpy (tương đương FAISS IndexFlatIP).
- Chunking tự động nếu p95 độ dài > 512 token BGE-M3. Số đo dry-run: chưa đo — xem meta.json sau khi build trên Kaggle.
- Dry-run (hashing embedder, 500 doc): corpus đầy đủ có 17955 context khác nhau từ 44313 dòng.
- Spec: `../docs/superpowers/specs/2026-11-04-vimedaqa-baseline-rag-design.md`.
