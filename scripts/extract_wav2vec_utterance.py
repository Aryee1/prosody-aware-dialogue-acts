from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402


def mean_pool(last_hidden: torch.Tensor, feat_lengths: torch.Tensor) -> torch.Tensor:
    b, t, h = last_hidden.shape
    ar = torch.arange(t, device=last_hidden.device).unsqueeze(0).expand(b, t)
    mask = ar < feat_lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = feat_lengths.clamp_min(1).unsqueeze(-1).float()
    return summed / denom


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--audio_dir", default="data/raw/ami_audio_mix")
    ap.add_argument("--segments", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--out", default="features/cache/wav2vec/ami_mix_w2v.parquet")
    ap.add_argument("--model_name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--max_sec", type=float, default=12.0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    audio_dir = (REPO_ROOT / args.audio_dir).resolve()
    seg_path = (REPO_ROOT / args.segments).resolve()
    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seg = pd.read_parquet(seg_path)
    if seg.empty:
        raise RuntimeError(f"Segments file is empty: {seg_path}")

    seg = seg[["meeting", "dact_id", "start", "end"]].copy()
    seg["start"] = seg["start"].astype(float)
    seg["end"] = seg["end"].astype(float)

    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device)
    model.eval()

    rows = []
    meetings = sorted(seg["meeting"].unique().tolist())

    print("Device:", device)
    print("Meetings:", len(meetings))
    print("Segments:", len(seg))
    print("Audio dir:", audio_dir)

    for mi, meeting in enumerate(meetings, start=1):
        wav_path = audio_dir / f"{meeting}.Mix-Headset.wav"
        if not wav_path.exists():
            continue

        wav, sr = torchaudio.load(str(wav_path))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        target_sr = int(args.target_sr)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr

        t = wav.numel()
        dur_total = t / sr

        mseg = seg[seg["meeting"] == meeting].copy()

        batch_wavs = []
        batch_meta = []

        for r in mseg.itertuples(index=False):
            start = max(0.0, float(r.start))
            end = min(dur_total, float(r.end))
            if end <= start:
                continue

            if (end - start) > float(args.max_sec):
                end = start + float(args.max_sec)

            s0 = int(start * sr)
            s1 = int(end * sr)
            s0 = max(0, min(t - 1, s0))
            s1 = max(s0 + 1, min(t, s1))

            seg_wav = wav[s0:s1].cpu().numpy()

            batch_wavs.append(seg_wav)
            batch_meta.append((meeting, str(r.dact_id)))

            if len(batch_wavs) >= int(args.batch_size):
                inputs = processor(batch_wavs, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs["input_values"].to(device)
                attn = inputs.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)

                out = model(input_values, attention_mask=attn)
                last_hidden = out.last_hidden_state

                if attn is None:
                    feat_lengths = torch.full((last_hidden.size(0),), last_hidden.size(1), device=device, dtype=torch.long)
                else:
                    input_lengths = attn.sum(dim=1)
                    feat_lengths = model._get_feat_extract_output_lengths(input_lengths)

                pooled = mean_pool(last_hidden, feat_lengths).detach().cpu().numpy().astype(np.float32)

                for (m, did), emb in zip(batch_meta, pooled):
                    rows.append({"meeting": m, "dact_id": did, "w2v_mean": emb.tolist()})

                batch_wavs, batch_meta = [], []

        if batch_wavs:
            inputs = processor(batch_wavs, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            attn = inputs.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)

            out = model(input_values, attention_mask=attn)
            last_hidden = out.last_hidden_state

            if attn is None:
                feat_lengths = torch.full((last_hidden.size(0),), last_hidden.size(1), device=device, dtype=torch.long)
            else:
                input_lengths = attn.sum(dim=1)
                feat_lengths = model._get_feat_extract_output_lengths(input_lengths)

            pooled = mean_pool(last_hidden, feat_lengths).detach().cpu().numpy().astype(np.float32)

            for (m, did), emb in zip(batch_meta, pooled):
                rows.append({"meeting": m, "dact_id": did, "w2v_mean": emb.tolist()})

        if mi % 5 == 0:
            print(f"Processed {mi}/{len(meetings)} meetings")

    w = pd.DataFrame(rows)
    w.to_parquet(out_path, index=False)

    manifest = {
        "model_name": args.model_name,
        "audio_dir": str(audio_dir),
        "segments": str(seg_path),
        "out": str(out_path),
        "n_rows": int(len(w)),
        "embedding_dim": int(len(w["w2v_mean"].iloc[0])) if len(w) else 0,
    }
    mdir = out_path.parent.parent / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "wav2vec_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nWrote:", out_path)
    print("Rows:", len(w))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
