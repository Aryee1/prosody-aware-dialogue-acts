from __future__ import annotations

from pathlib import Path
import yaml


def repo_root() -> Path:
    # .../src/common/config.py -> repo root is two levels up
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = (repo_root() / p).resolve()

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = repo_root()

    # Resolve AMI paths relative to repo root
    ami = cfg.get("paths", {}).get("ami", {})
    for k in ["annotations_zip_path", "annotations_dir", "words_dir", "dialogue_acts_dir"]:
        if k in ami:
            ami[k] = str((root / ami[k]).resolve())

    # Resolve outputs relative to repo root
    outs = cfg.get("outputs", {})
    for k in ["utterances_parquet", "utterances_csv", "splits_json"]:
        if k in outs:
            outs[k] = str((root / outs[k]).resolve())

    return cfg