from __future__ import annotations
from pathlib import Path
import sys, argparse, json
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

@torch.no_grad()
def infer_text_probs(texts, model_dir: Path, max_length=128, batch_size=64):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(outs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab.yaml")
    ap.add_argument("--prosody", default="features/cache/prosody/ami_mix_prosody.parquet")
    ap.add_argument("--wav2vec", default="features/cache/wav2vec/ami_mix_w2v.parquet")
    ap.add_argument("--text_model_dir", default="models/checkpoints/text_only_distilbert")
    ap.add_argument("--out_dir", default="reports/results/ablation_mix30_w2v_pca")
    ap.add_argument("--pca_dim", type=int, default=128)
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

    df = utter.merge(pros, on=["meeting","dact_id"], how="inner").merge(w2v, on=["meeting","dact_id"], how="inner")
    meeting_to_split = {m:s for s in ["train","val","test"] for m in splits.get(s, [])}
    df["split"] = df["meeting"].map(meeting_to_split)
    df = df[df["split"].isin(["train","test"])].copy()

    train_df = df[df["split"]=="train"].copy()
    test_df = df[df["split"]=="test"].copy()

    labels = sorted(train_df["label"].unique().tolist())
    label2id = {lab:i for i,lab in enumerate(labels)}
    train_df = train_df[train_df["label"].isin(label2id)].copy()
    test_df = test_df[test_df["label"].isin(label2id)].copy()
    y_train = train_df["label"].map(label2id).to_numpy(np.int64)
    y_test = test_df["label"].map(label2id).to_numpy(np.int64)

    pros_cols = [
        "duration","word_count","words_per_sec","rms_mean","rms_std","silence_ratio",
        "longest_silence_sec","peaks_per_sec","voiced_ratio","f0_mean","f0_std","f0_min","f0_max"
    ]

    model_dir = (REPO_ROOT / args.text_model_dir).resolve()
    train_probs = infer_text_probs(train_df["text"].astype(str).tolist(), model_dir, args.max_length, args.batch_size)
    test_probs  = infer_text_probs(test_df["text"].astype(str).tolist(),  model_dir, args.max_length, args.batch_size)

    Xw_train = np.vstack(train_df["w2v_mean"].to_list()).astype(np.float32)
    Xw_test  = np.vstack(test_df["w2v_mean"].to_list()).astype(np.float32)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # PCA on wav2vec only (fit on train)
    scaler_w = StandardScaler()
    Xw_train_s = scaler_w.fit_transform(Xw_train)
    Xw_test_s  = scaler_w.transform(Xw_test)

    pca = PCA(n_components=int(args.pca_dim), random_state=seed)
    Xw_train_p = pca.fit_transform(Xw_train_s)
    Xw_test_p  = pca.transform(Xw_test_s)

    def fit_eval(Xtr, Xte):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y_train)
        yp = clf.predict(Xte)
        return compute_metrics(y_test, yp)

    # Text only on subset
    text_only = compute_metrics(y_test, np.argmax(test_probs, axis=1))

    # Text + wav2vec(PCA)
    m_text_w2v = fit_eval(np.hstack([train_probs, Xw_train_p]), np.hstack([test_probs, Xw_test_p]))

    # Text + prosody + wav2vec(PCA)
    Xp_train = train_df[pros_cols].to_numpy(np.float32)
    Xp_test  = test_df[pros_cols].to_numpy(np.float32)
    m_all = fit_eval(np.hstack([train_probs, Xp_train, Xw_train_p]), np.hstack([test_probs, Xp_test, Xw_test_p]))

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    table = []
    table.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
    table.append("|---|---:|---:|---:|")
    table.append(f"| Text only (subset) | {text_only['accuracy']:.3f} | {text_only['macro_f1']:.3f} | {text_only['weighted_f1']:.3f} |")
    table.append(f"| Text + wav2vec (PCA{args.pca_dim}) | {m_text_w2v['accuracy']:.3f} | {m_text_w2v['macro_f1']:.3f} | {m_text_w2v['weighted_f1']:.3f} |")
    table.append(f"| Text + Prosody + wav2vec (PCA{args.pca_dim}) | {m_all['accuracy']:.3f} | {m_all['macro_f1']:.3f} | {m_all['weighted_f1']:.3f} |")
    (out_dir / "ablation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    (out_dir / "metrics.json").write_text(json.dumps({
        "text_only_subset": text_only,
        "text_plus_wav2vec_pca": m_text_w2v,
        "text_plus_prosody_plus_wav2vec_pca": m_all,
        "pca_dim": int(args.pca_dim)
    }, indent=2), encoding="utf-8")

    print((out_dir / "ablation_table.md").read_text())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
