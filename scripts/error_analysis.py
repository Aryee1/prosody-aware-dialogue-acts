from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402


@torch.no_grad()
def infer_text_probs(
    texts: list[str],
    model_dir: Path,
    max_length: int = 128,
    batch_size: int = 64,
) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        out.append(probs)
    return np.vstack(out)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: dict[int, str],
    k: int = 20,
) -> pd.DataFrame:
    from collections import Counter
    pairs = Counter()
    for a, b in zip(y_true, y_pred):
        a = int(a)
        b = int(b)
        if a != b:
            pairs[(a, b)] += 1
    rows = [{"gold": id2label[a], "pred": id2label[b], "count": int(c)} for (a, b), c in pairs.most_common(k)]
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--text_model_dir", required=True)
    ap.add_argument("--mode", choices=["text_only", "text_plus_prosody", "text_plus_wav2vec_pca"], default="text_plus_prosody")
    ap.add_argument("--prosody", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--wav2vec", default="features/cache/wav2vec/ami_mix_w2v.parquet")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out_dir", default="reports/error_analysis")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    utter = pd.read_parquet(cfg["outputs"]["utterances_parquet"])
    splits = json.loads(Path(cfg["outputs"]["splits_json"]).read_text(encoding="utf-8"))

    df = utter.copy()

    if args.mode in ["text_plus_prosody", "text_plus_wav2vec_pca"]:
        pros = pd.read_parquet((REPO_ROOT / args.prosody).resolve())
        df = df.merge(pros, on=["meeting", "dact_id"], how="inner")

    if args.mode == "text_plus_wav2vec_pca":
        w2v = pd.read_parquet((REPO_ROOT / args.wav2vec).resolve())
        df = df.merge(w2v, on=["meeting", "dact_id"], how="inner")

    meeting_to_split = {m: s for s in ["train", "val", "test"] for m in splits.get(s, [])}
    df["split"] = df["meeting"].map(meeting_to_split)
    df = df[df["split"].isin(["train", "test"])].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    model_dir = (REPO_ROOT / args.text_model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    label2id = dict(m.config.label2id) if m.config.label2id else None
    id2label = {int(k): v for k, v in dict(m.config.id2label).items()} if m.config.id2label else None

    if not label2id or not id2label:
        labels = sorted(train_df["label"].unique().tolist())
        label2id = {lab: i for i, lab in enumerate(labels)}
        id2label = {i: lab for lab, i in label2id.items()}

    train_df = train_df[train_df["label"].isin(label2id)].copy()
    test_df = test_df[test_df["label"].isin(label2id)].copy()

    y_train = train_df["label"].map(label2id).to_numpy(np.int64)
    y_test = test_df["label"].map(label2id).to_numpy(np.int64)

    train_probs = infer_text_probs(train_df["text"].tolist(), model_dir, args.max_length, args.batch_size)
    test_probs = infer_text_probs(test_df["text"].tolist(), model_dir, args.max_length, args.batch_size)

    if args.mode == "text_only":
        y_pred = np.argmax(test_probs, axis=1)
        y_proba = test_probs
        mode_name = "text_only"

    elif args.mode == "text_plus_prosody":
        pros_cols = [
            "duration", "word_count", "words_per_sec", "rms_mean", "rms_std", "silence_ratio",
            "longest_silence_sec", "peaks_per_sec", "voiced_ratio", "f0_mean", "f0_std", "f0_min", "f0_max"
        ]
        X_train = np.hstack([train_probs, train_df[pros_cols].to_numpy(np.float32)])
        X_test = np.hstack([test_probs, test_df[pros_cols].to_numpy(np.float32)])

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        clf = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=seed)),
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        mode_name = "text_plus_prosody"

    else:
        Xw_train = np.vstack(train_df["w2v_mean"].to_list()).astype(np.float32)
        Xw_test = np.vstack(test_df["w2v_mean"].to_list()).astype(np.float32)

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        sc = StandardScaler()
        Xw_train_s = sc.fit_transform(Xw_train)
        Xw_test_s = sc.transform(Xw_test)

        pca = PCA(n_components=int(args.pca_dim), random_state=seed)
        Xw_train_p = pca.fit_transform(Xw_train_s)
        Xw_test_p = pca.transform(Xw_test_s)

        X_train = np.hstack([train_probs, Xw_train_p])
        X_test = np.hstack([test_probs, Xw_test_p])

        clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        mode_name = f"text_plus_wav2vec_pca{args.pca_dim}"

    metrics = compute_metrics(y_test, y_pred)

    out_dir = (REPO_ROOT / args.out_dir / mode_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    conf = top_confusions(y_test, y_pred, id2label, k=20)
    conf.to_csv(out_dir / "top_confusions.csv", index=False)

    pred_conf = np.max(y_proba, axis=1) if y_proba is not None else np.ones_like(y_pred, dtype=float)

    mis = test_df.copy()
    mis["gold_id"] = y_test
    mis["pred_id"] = y_pred
    mis["gold"] = [id2label[int(i)] for i in y_test]
    mis["pred"] = [id2label[int(i)] for i in y_pred]
    mis["pred_conf"] = pred_conf

    mis = mis[mis["gold_id"] != mis["pred_id"]].copy()
    mis = mis.sort_values(["pred_conf"], ascending=True).head(int(args.n))

    keep = ["meeting", "dact_id", "gold", "pred", "pred_conf", "text"]
    for c in ["duration", "silence_ratio", "rms_mean", "f0_mean", "word_count", "words_per_sec"]:
        if c in mis.columns:
            keep.append(c)

    mis[keep].to_csv(out_dir / "misclassified_examples.csv", index=False)

    print("Mode:", mode_name)
    print("Test metrics:", metrics)
    print("\nTop confusions:")
    if len(conf):
        print(conf.head(10).to_string(index=False))
    print("\nSaved to:", out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
