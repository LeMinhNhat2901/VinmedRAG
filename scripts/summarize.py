from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from vimedrag.report import summarize_runs  # noqa: E402

if __name__ == "__main__":
    print(summarize_runs(sys.argv[1] if len(sys.argv) > 1 else "runs"))
