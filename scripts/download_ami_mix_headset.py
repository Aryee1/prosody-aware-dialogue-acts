from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402


AMI_MIX_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/HeadsetAudio/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max_meetings", type=int, default=None, help="Optional cap for a quick run")
    args = ap.parse_args()

    cfg = load_config(args.config)

    splits_path = Path(cfg["outputs"]["splits_json"])
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    meetings = []
    for s in ["train", "val", "test"]:
        meetings.extend(splits.get(s, []))
    meetings = sorted(set(meetings))

    if args.max_meetings is not None:
        meetings = meetings[: args.max_meetings]

    out_dir = REPO_ROOT / "data" / "raw" / "ami_audio_mix"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    miss = 0
    for m in meetings:
        fname = f"{m}.Mix-Headset.wav"
        url = AMI_MIX_BASE + fname
        out_path = out_dir / fname

        if out_path.exists() and out_path.stat().st_size > 0:
            ok += 1
            continue

        try:
            print(f"Downloading {fname}")
            urlretrieve(url, out_path)
            ok += 1
        except Exception as e:
            miss += 1
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            print(f"Failed: {fname} ({e})")

    print(f"\nDone. Downloaded: {ok}, failed: {miss}")
    print(f"Audio folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
