from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--features", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--run_dir", default="reports/results/prosody_only_mix")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    utter = pd.read_parquet(cfg["outputs"]["utterances_parquet"])
    splits = json.loads(Path(cfg["outputs"]["splits_json"]).read_text(encoding="utf-8"))
    feat = pd.read_parquet((REPO_ROOT / args.features).resolve())

    df = utter.merge(feat, on=["meeting", "dact_id"], how="inner")
    if df.empty:
        raise RuntimeError("No merged rows. Check that prosody features exist and keys match.")

    meeting_to_split = {}
    for split_name in ["train", "val", "test"]:
        for m in splits.get(split_name, []):
            meeting_to_split[m] = split_name
    df["split"] = df["meeting"].map(meeting_to_split)
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    feat_cols = [
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

    df = df.dropna(subset=feat_cols + ["label"]).copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    labels = sorted(train_df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    val_df = val_df[val_df["label"].isin(label2id)].copy()
    test_df = test_df[test_df["label"].isin(label2id)].copy()

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].map(label2id).to_numpy(dtype=np.int64)

    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].map(label2id).to_numpy(dtype=np.int64)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=seed
            )),
        ]
    )

    print("Merged rows:", len(df))
    print("Split rows:", {k: int((df["split"] == k).sum()) for k in ["train", "val", "test"]})
    print("Audio meetings:", int(df["meeting"].nunique()))
    print("Labels:", len(labels))

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")

    rep = classification_report(
        y_test,
        y_pred,
        target_names=[id2label[i] for i in range(len(labels))],
        digits=4,
        zero_division=0,
    )
    (run_dir / "classification_report.txt").write_text(rep, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=[id2label[i] for i in range(len(labels))],
        columns=[id2label[i] for i in range(len(labels))],
    ).to_csv(run_dir / "confusion_matrix.csv", encoding="utf-8")

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "audio_meetings": int(df["meeting"].nunique()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "features": feat_cols,
        "model": "StandardScaler + multinomial LogisticRegression (class_weight=balanced)",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nTest metrics:")
    print(metrics)
    print("\nWrote:", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
