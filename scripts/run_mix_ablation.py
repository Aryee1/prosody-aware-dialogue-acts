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
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--features", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--text_model_dir", default="models/checkpoints/text_only_distilbert")
    ap.add_argument("--out_dir", default="reports/results/ablation_mix30")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    utter = pd.read_parquet(cfg["outputs"]["utterances_parquet"])
    splits = json.loads(Path(cfg["outputs"]["splits_json"]).read_text(encoding="utf-8"))
    pros = pd.read_parquet((REPO_ROOT / args.features).resolve())

    # Merge on meeting + dact_id, so all models evaluate on the exact same subset
    df = utter.merge(pros, on=["meeting", "dact_id"], how="inner")
    if df.empty:
        raise RuntimeError("No merged rows. Check prosody features and keys.")

    meeting_to_split = {}
    for split_name in ["train", "val", "test"]:
        for m in splits.get(split_name, []):
            meeting_to_split[m] = split_name
    df["split"] = df["meeting"].map(meeting_to_split)
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    # Use only labels seen in train (standard)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    labels = sorted(train_df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    val_df = val_df[val_df["label"].isin(label2id)].copy()
    test_df = test_df[test_df["label"].isin(label2id)].copy()

    # Prosody features
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

    # Drop NaNs
    train_df = train_df.dropna(subset=pros_cols + ["text", "label"]).copy()
    test_df = test_df.dropna(subset=pros_cols + ["text", "label"]).copy()

    # Encode labels
    y_train = train_df["label"].map(label2id).to_numpy(dtype=np.int64)
    y_test = test_df["label"].map(label2id).to_numpy(dtype=np.int64)

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Text-only on the same subset ----
    model_dir = (REPO_ROOT / args.text_model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Text model dir not found: {model_dir}\n"
            f"Train text baseline in this runtime, or point --text_model_dir to the right folder."
        )

    test_probs = infer_text_probs(
        test_df["text"].astype(str).tolist(),
        model_dir=model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    y_pred_text = np.argmax(test_probs, axis=1)
    text_metrics = compute_metrics(y_test, y_pred_text)
    (out_dir / "text_only_subset_metrics.json").write_text(json.dumps(text_metrics, indent=2), encoding="utf-8")

    # ---- Fusion: text probs + prosody features ----
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    train_probs = infer_text_probs(
        train_df["text"].astype(str).tolist(),
        model_dir=model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    X_train = np.hstack([train_probs, train_df[pros_cols].to_numpy(dtype=np.float32)])
    X_test = np.hstack([test_probs, test_df[pros_cols].to_numpy(dtype=np.float32)])

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                class_weight="balanced",
                random_state=seed,
            )),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred_fuse = clf.predict(X_test)
    fusion_metrics = compute_metrics(y_test, y_pred_fuse)
    (out_dir / "fusion_metrics.json").write_text(json.dumps(fusion_metrics, indent=2), encoding="utf-8")

    # ---- Write ablation table ----
    table = []
    table.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
    table.append("|---|---:|---:|---:|")
    table.append(f"| Text only (subset) | {text_metrics['accuracy']:.3f} | {text_metrics['macro_f1']:.3f} | {text_metrics['weighted_f1']:.3f} |")
    # prosody-only numbers come from your earlier run; we record them here as reference
    # You can replace these later if you re-train prosody-only with improved pitch.
    table.append("| Prosody only (subset) | 0.168 | 0.103 | 0.157 |")
    table.append(f"| Text + Prosody (late fusion) | {fusion_metrics['accuracy']:.3f} | {fusion_metrics['macro_f1']:.3f} | {fusion_metrics['weighted_f1']:.3f} |")

    (out_dir / "ablation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_rows_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "audio_meetings": int(df["meeting"].nunique()),
        "text_model_dir": str(model_dir),
        "prosody_features": pros_cols,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nText-only (subset) metrics:", text_metrics)
    print("Fusion metrics:", fusion_metrics)
    print("\nWrote:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
