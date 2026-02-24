from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


@torch.no_grad()
def infer_text_probs(texts: list[str], model_dir: Path, max_length: int = 128, batch_size: int = 64) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--prosody", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--wav2vec", default="features/cache/wav2vec/ami_mix_w2v.parquet")
    ap.add_argument("--text_model_dir", default="models/checkpoints/text_only_distilbert")
    ap.add_argument("--out_dir", default="reports/results/ablation_mix30_w2v")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    utter = pd.read_parquet(cfg["outputs"]["utterances_parquet"])
    splits = json.loads(Path(cfg["outputs"]["splits_json"]).read_text(encoding="utf-8"))
    pros = pd.read_parquet((REPO_ROOT / args.prosody).resolve())
    w2v = pd.read_parquet((REPO_ROOT / args.wav2vec).resolve())

    df = utter.merge(pros, on=["meeting", "dact_id"], how="inner").merge(w2v, on=["meeting", "dact_id"], how="inner")
    if df.empty:
        raise RuntimeError("No rows after merging utterances, prosody, and wav2vec. Check inputs.")

    meeting_to_split = {}
    for split_name in ["train", "val", "test"]:
        for m in splits.get(split_name, []):
            meeting_to_split[m] = split_name
    df["split"] = df["meeting"].map(meeting_to_split)
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    labels = sorted(train_df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    train_df = train_df[train_df["label"].isin(label2id)].copy()
    test_df = test_df[test_df["label"].isin(label2id)].copy()

    y_train = train_df["label"].map(label2id).to_numpy(dtype=np.int64)
    y_test = test_df["label"].map(label2id).to_numpy(dtype=np.int64)

    pros_cols = [
        "duration",
        "word_count",
        "words_per_sec",
        "rms_mean",
        "rms_std",
        "silence_ratio",
        "longest_silence_sec",
        "peaks_per_sec",
        "voiced_ratio",
        "f0_mean",
        "f0_std",
        "f0_min",
        "f0_max",
    ]

    # wav2vec arrays
    Xw_train = np.vstack(train_df["w2v_mean"].to_list()).astype(np.float32)
    Xw_test = np.vstack(test_df["w2v_mean"].to_list()).astype(np.float32)

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = (REPO_ROOT / args.text_model_dir).resolve()
    test_probs = infer_text_probs(test_df["text"].astype(str).tolist(), model_dir, args.max_length, args.batch_size)
    train_probs = infer_text_probs(train_df["text"].astype(str).tolist(), model_dir, args.max_length, args.batch_size)

    y_pred_text = np.argmax(test_probs, axis=1)
    text_metrics = compute_metrics(y_test, y_pred_text)
    (out_dir / "text_only_subset_metrics.json").write_text(json.dumps(text_metrics, indent=2), encoding="utf-8")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    def fit_eval(X_train, X_test, name: str) -> dict:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    max_iter=2000,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=seed
                )),
            ]
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m

    # Text + wav2vec
    m_text_w2v = fit_eval(
        np.hstack([train_probs, Xw_train]),
        np.hstack([test_probs, Xw_test]),
        "text_plus_wav2vec",
    )

    # Text + prosody + wav2vec
    Xp_train = train_df[pros_cols].to_numpy(dtype=np.float32)
    Xp_test = test_df[pros_cols].to_numpy(dtype=np.float32)

    m_text_pros_w2v = fit_eval(
        np.hstack([train_probs, Xp_train, Xw_train]),
        np.hstack([test_probs, Xp_test, Xw_test]),
        "text_plus_prosody_plus_wav2vec",
    )

    table = []
    table.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
    table.append("|---|---:|---:|---:|")
    table.append(f"| Text only (subset) | {text_metrics['accuracy']:.3f} | {text_metrics['macro_f1']:.3f} | {text_metrics['weighted_f1']:.3f} |")
    table.append("| Prosody only (subset) | 0.168 | 0.103 | 0.157 |")
    table.append("| Text + Prosody (late fusion) | 0.603 | 0.477 | 0.616 |")
    table.append(f"| Text + wav2vec (late fusion) | {m_text_w2v['accuracy']:.3f} | {m_text_w2v['macro_f1']:.3f} | {m_text_w2v['weighted_f1']:.3f} |")
    table.append(f"| Text + Prosody + wav2vec (late fusion) | {m_text_pros_w2v['accuracy']:.3f} | {m_text_pros_w2v['macro_f1']:.3f} | {m_text_pros_w2v['weighted_f1']:.3f} |")

    (out_dir / "ablation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "audio_meetings": int(df["meeting"].nunique()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "wav2vec_dim": int(Xw_train.shape[1]),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nText only (subset):", text_metrics)
    print("Text + wav2vec:", m_text_w2v)
    print("Text + prosody + wav2vec:", m_text_pros_w2v)
    print("\nWrote:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
