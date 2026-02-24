from __future__ import annotations
from pathlib import Path

def repo_root() -> Path:
    # .../src/common/paths.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[2]