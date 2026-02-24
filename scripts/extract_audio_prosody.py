from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
from xml.dom import minidom

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402


def load_words_times(words_path: Path):
    doc = minidom.parse(str(words_path))
    nodes = doc.getElementsByTagName("w")

    id_to_pos = {}
    w_st = []
    w_et = []

    for n in nodes:
        wid = n.getAttribute("nite:id")
        st = n.getAttribute("starttime") or ""
        et = n.getAttribute("endtime") or ""
        st_f = float(st) if st.strip() else None
        et_f = float(et) if et.strip() else None

        pos = len(w_st)
        if wid:
            id_to_pos[wid] = pos
        w_st.append(st_f)
        w_et.append(et_f)

    return id_to_pos, w_st, w_et


def span_times(id_to_pos, w_st, w_et, start_id: str | None, end_id: str | None):
    if not start_id or start_id not in id_to_pos:
        return None, None

    a = id_to_pos[start_id]
    b = a
    if end_id and end_id in id_to_pos:
        b = id_to_pos[end_id]
    if b < a:
        a, b = b, a

    sts = [t for t in w_st[a:b+1] if t is not None]
    ets = [t for t in w_et[a:b+1] if t is not None]
    if not sts or not ets:
        return None, None
    return float(min(sts)), float(max(ets))


def frame_rms(wave: torch.Tensor, frame_len: int, hop: int) -> torch.Tensor:
    if wave.numel() < frame_len:
        wave = torch.nn.functional.pad(wave, (0, frame_len - wave.numel()))
    n_frames = 1 + (wave.numel() - frame_len) // hop
    if n_frames <= 0:
        n_frames = 1
    frames = wave.unfold(0, frame_len, hop)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1).clamp_min(1e-12))
    return rms


def count_peaks(x: np.ndarray, thr: float) -> int:
    if x.size < 3:
        return 0
    c = 0
    for i in range(1, x.size - 1):
        if x[i] > thr and x[i] > x[i - 1] and x[i] > x[i + 1]:
            c += 1
    return c


def longest_silence_sec(silent: np.ndarray, hop_sec: float) -> float:
    best = 0
    cur = 0
    for v in silent:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best) * hop_sec


def pitch_stats(seg: torch.Tensor, sr: int):
    """
    Try two parameter conventions for detect_pitch_frequency and pick the one
    that yields more voiced frames.
    """
    candidates = []
    for frame_time, win_length in [(10.0, 30.0), (0.01, 0.03)]:
        try:
            f0 = torchaudio.functional.detect_pitch_frequency(
                seg.unsqueeze(0),
                sample_rate=sr,
                frame_time=frame_time,
                win_length=win_length,
            ).squeeze(0).cpu().numpy()

            f0 = np.asarray(f0, dtype=np.float64)
            voiced = f0 > 0
            vr = float(np.mean(voiced)) if f0.size else 0.0
            candidates.append((vr, f0))
        except Exception:
            continue

    if not candidates:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    voiced_ratio, f0 = candidates[0]
    voiced = f0 > 0
    f0v = f0[voiced]
    if f0v.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        float(voiced_ratio),
        float(np.mean(f0v)),
        float(np.std(f0v)),
        float(np.min(f0v)),
        float(np.max(f0v)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--audio_dir", default="data/raw/ami_audio_mix")
    ap.add_argument("--out", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--max_meetings", type=int, default=None)
    ap.add_argument("--target_sr", type=int, default=16000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    utter_path = Path(cfg["outputs"]["utterances_parquet"])
    df = pd.read_parquet(utter_path)

    audio_dir = (REPO_ROOT / args.audio_dir).resolve()
    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    words_dir = Path(cfg["paths"]["ami"]["words_dir"])

    meetings_all = sorted(df["meeting"].unique().tolist())
    meetings_with_audio = []
    for m in meetings_all:
        wav = audio_dir / f"{m}.Mix-Headset.wav"
        if wav.exists() and wav.stat().st_size > 0:
            meetings_with_audio.append(m)

    if args.max_meetings is not None:
        meetings_with_audio = meetings_with_audio[: args.max_meetings]

    if not meetings_with_audio:
        raise FileNotFoundError(f"No Mix-Headset wav files found in: {audio_dir}")

    frame_len_sec = 0.025
    hop_sec = 0.010

    rows = []
    skipped = 0
    derived = 0

    words_cache = {}  # words_file -> (id_to_pos, w_st, w_et)

    print(f"Audio dir: {audio_dir}")
    print(f"Meetings with audio: {len(meetings_with_audio)}")

    for mi, meeting in enumerate(meetings_with_audio, start=1):
        wav_path = audio_dir / f"{meeting}.Mix-Headset.wav"
        wav, sr = torchaudio.load(str(wav_path))
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze(0)

        target_sr = int(args.target_sr)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr

        T = wav.numel()
        dur_total = T / sr

        frame_len = int(round(frame_len_sec * sr))
        hop = int(round(hop_sec * sr))

        mdf = df[df["meeting"] == meeting].copy()

        for r in mdf.itertuples(index=False):
            start = r.start
            end = r.end

            missing = (
                start is None or end is None or
                (isinstance(start, float) and np.isnan(start)) or
                (isinstance(end, float) and np.isnan(end))
            )

            if missing:
                words_file = getattr(r, "words_file", None)
                start_w = getattr(r, "start_word_id", None)
                end_w = getattr(r, "end_word_id", None)

                if not words_file or not start_w:
                    skipped += 1
                    continue

                if words_file not in words_cache:
                    wp = words_dir / words_file
                    if not wp.exists():
                        hits = list(words_dir.rglob(words_file))
                        if hits:
                            wp = hits[0]
                    if not wp.exists():
                        skipped += 1
                        continue
                    words_cache[words_file] = load_words_times(wp)

                id_to_pos, w_st, w_et = words_cache[words_file]
                ds, de = span_times(id_to_pos, w_st, w_et, str(start_w), str(end_w) if end_w else None)
                if ds is None or de is None:
                    skipped += 1
                    continue
                start = ds
                end = de
                derived += 1

            start = float(start)
            end = float(end)
            if end <= start:
                skipped += 1
                continue

            start = max(0.0, start)
            end = min(dur_total, end)
            seg_dur = end - start
            if seg_dur < 0.05:
                skipped += 1
                continue

            s0 = int(start * sr)
            s1 = int(end * sr)
            s0 = max(0, min(T - 1, s0))
            s1 = max(s0 + 1, min(T, s1))

            seg = wav[s0:s1]

            rms = frame_rms(seg, frame_len=frame_len, hop=hop).cpu().numpy()
            rms_max = float(np.max(rms)) if rms.size else 0.0
            rms_mean = float(np.mean(rms)) if rms.size else 0.0
            rms_std = float(np.std(rms)) if rms.size else 0.0

            thr = max(1e-4, 0.1 * rms_max)
            silent = (rms < thr) if rms.size else np.array([], dtype=bool)
            silence_ratio = float(np.mean(silent)) if silent.size else 0.0
            longest_silence = longest_silence_sec(silent, hop_sec=hop_sec)

            peaks = count_peaks(rms, thr=thr)
            peaks_per_sec = float(peaks / seg_dur)

            voiced_ratio, f0_mean, f0_std, f0_min, f0_max = pitch_stats(seg, sr)

            text = str(getattr(r, "text", "")) if getattr(r, "text", None) is not None else ""
            word_count = int(len(text.split())) if text else 0
            words_per_sec = float(word_count / seg_dur) if seg_dur > 0 else 0.0

            rows.append(
                {
                    "meeting": meeting,
                    "dact_id": str(getattr(r, "dact_id")),
                    "start": float(start),
                    "end": float(end),
                    "duration": float(seg_dur),
                    "word_count": word_count,
                    "words_per_sec": words_per_sec,
                    "rms_mean": rms_mean,
                    "rms_std": rms_std,
                    "silence_ratio": silence_ratio,
                    "longest_silence_sec": longest_silence,
                    "peaks_per_sec": peaks_per_sec,
                    "voiced_ratio": voiced_ratio,
                    "f0_mean": f0_mean,
                    "f0_std": f0_std,
                    "f0_min": f0_min,
                    "f0_max": f0_max,
                }
            )

        if mi % 5 == 0:
            print(f"Processed {mi}/{len(meetings_with_audio)} meetings...")

    feat = pd.DataFrame(rows)
    feat.to_parquet(out_path, index=False)

    manifest = {
        "audio_dir": str(audio_dir),
        "out_path": str(out_path),
        "n_rows": int(len(feat)),
        "n_meetings": int(feat["meeting"].nunique()) if len(feat) else 0,
        "skipped": int(skipped),
        "derived_times_from_words": int(derived),
    }
    mpath = out_path.parent.parent / "manifests"
    mpath.mkdir(parents=True, exist_ok=True)
    (mpath / "prosody_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote: {out_path}")
    print(f"Wrote: {mpath / 'prosody_manifest.json'}")
    print(f"Rows: {len(feat):,}")
    print(f"Derived start/end from words for: {derived:,} utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
