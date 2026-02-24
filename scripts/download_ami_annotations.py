from __future__ import annotations

from pathlib import Path
import sys
import argparse
import zipfile
from urllib.request import urlretrieve

# Ensure repo root is on sys.path so `import src...` works when running scripts directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ami = cfg["paths"]["ami"]

    zip_url = ami["annotations_zip_url"]
    zip_path = Path(ami["annotations_zip_path"])
    out_dir = Path(ami["annotations_dir"])

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"OK: annotations already present at: {out_dir}")
        return 0

    if not zip_path.exists():
        print(f"Downloading: {zip_url}")
        print(f"To: {zip_path}")
        urlretrieve(zip_url, zip_path)

    print(f"Extracting: {zip_path} -> {out_dir.parent}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)

    # Some zips extract to a top-level folder named ami_public_manual_1.6.2
    # If config expects a different name, normalize it here.
    if not out_dir.exists():
        candidate = out_dir.parent / "ami_public_manual_1.6.2"
        if candidate.exists():
            candidate.rename(out_dir)

    if not out_dir.exists():
        raise FileNotFoundError(
            f"Extraction finished but expected folder not found: {out_dir}\n"
            f"Check what was extracted under: {out_dir.parent}"
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())