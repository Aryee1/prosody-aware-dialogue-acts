from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from datetime import datetime
import inspect

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so `import src...` works when running scripts directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402


def _read_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_merged_config(exp_config_path: str | Path) -> dict:
    exp_path = Path(exp_config_path)
    if not exp_path.is_absolute():
        exp_path = (REPO_ROOT / exp_path).resolve()

    exp_cfg = _read_yaml(exp_path)
    base_rel = exp_cfg.get("base_config")
    if not base_rel:
        raise ValueError("Experiment config must include base_config: ../default.yaml")

    base_path = (exp_path.parent / base_rel).resolve()
    base_cfg = load_config(base_path)

    exp_cfg = dict(exp_cfg)
    exp_cfg.pop("base_config", None)
    merged = _deep_update(base_cfg, exp_cfg)
    return merged


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _load_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    utter_path = Path(cfg["outputs"]["utterances_parquet"])
    splits_path = Path(cfg["outputs"]["splits_json"])

    df = pd.read_parquet(utter_path)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))

    meeting_to_split: dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        for m in splits.get(split_name, []):
            meeting_to_split[m] = split_name

    df["split"] = df["meeting"].map(meeting_to_split)
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df, splits


def _select_split(df: pd.DataFrame, split: str, max_n: int | None, seed: int) -> pd.DataFrame:
    sdf = df[df["split"] == split].copy()
    if max_n is None or len(sdf) <= max_n:
        return sdf
    return sdf.sample(n=max_n, random_state=seed).copy()


def _make_training_args(**kwargs):
    """
    transformers v5 uses eval_strategy; older versions use evaluation_strategy.
    We remap safely and only pass supported kwargs.
    """
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    # Remap evaluation arg name if needed
    if "eval_strategy" in allowed and "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    ignored = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if ignored:
        print("\nNote: ignored TrainingArguments keys not supported by your transformers version:")
        for k in ignored:
            print(f"  - {k}")

    # If load_best_model_at_end is True, ensure eval and save strategies exist and match
    if filtered.get("load_best_model_at_end", False):
        eval_key = "evaluation_strategy" if "evaluation_strategy" in filtered else ("eval_strategy" if "eval_strategy" in filtered else None)
        save_key = "save_strategy" if "save_strategy" in filtered else None

        if not eval_key or not save_key:
            print("\nNote: cannot set load_best_model_at_end=True (missing eval/save strategy support). Disabling it.")
            filtered["load_best_model_at_end"] = False
        else:
            if str(filtered.get(eval_key)) != str(filtered.get(save_key)):
                print("\nNote: load_best_model_at_end requires eval and save strategies to match. Disabling it.")
                filtered["load_best_model_at_end"] = False

    return TrainingArguments(**filtered)


def _make_trainer(**kwargs):
    """
    Newer transformers versions removed `tokenizer=` from Trainer.
    We pass only supported kwargs.
    """
    from transformers import Trainer
    sig = inspect.signature(Trainer.__init__)
    allowed = set(sig.parameters.keys())

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    ignored = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if ignored:
        print("\nNote: ignored Trainer keys not supported by your transformers version:")
        for k in ignored:
            print(f"  - {k}")
    return Trainer(**filtered)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/text_only.yaml")
    args = ap.parse_args()

    cfg = load_merged_config(args.config)

    seed = int(cfg.get("experiment", {}).get("seed", cfg.get("project", {}).get("seed", 1337)))
    set_seed(seed)

    df, splits = _load_data(cfg)

    tcfg = cfg["text"]
    df_train = _select_split(df, "train", tcfg.get("max_train_samples"), seed)
    df_val = _select_split(df, "val", tcfg.get("max_val_samples"), seed)
    df_test = _select_split(df, "test", tcfg.get("max_test_samples"), seed)

    labels = sorted(df_train["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    # keep only labels seen in train
    df_val = df_val[df_val["label"].isin(label2id)].copy()
    df_test = df_test[df_test["label"].isin(label2id)].copy()

    df_train["labels"] = df_train["label"].map(label2id).astype(int)
    df_val["labels"] = df_val["label"].map(label2id).astype(int)
    df_test["labels"] = df_test["label"].map(label2id).astype(int)

    run_dir = _resolve(cfg["outputs"]["run_dir"])
    model_dir = _resolve(cfg["outputs"]["model_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    (run_dir / "splits_summary.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    meta = {
        "experiment": cfg.get("experiment", {}),
        "text": cfg.get("text", {}),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "n_labels": int(len(labels)),
        "repo_root": str(REPO_ROOT),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        set_seed as hf_set_seed,
    )
    import transformers
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    print(f"\ntransformers version: {transformers.__version__}")

    hf_set_seed(seed)

    model_name = tcfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=int(tcfg["max_length"]),
        )

    dtrain = Dataset.from_pandas(df_train[["text", "labels"]].reset_index(drop=True))
    dval = Dataset.from_pandas(df_val[["text", "labels"]].reset_index(drop=True))
    dtest = Dataset.from_pandas(df_test[["text", "labels"]].reset_index(drop=True))

    dtrain = dtrain.map(tok, batched=True, remove_columns=["text"])
    dval = dval.map(tok, batched=True, remove_columns=["text"])
    dtest = dtest.map(tok, batched=True, remove_columns=["text"])

    cols = ["input_ids", "attention_mask", "labels"]
    dtrain.set_format(type="torch", columns=cols)
    dval.set_format(type="torch", columns=cols)
    dtest.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        }

    import torch
    fp16 = bool(tcfg.get("fp16_if_cuda", True) and torch.cuda.is_available())

    print("\nSplit sizes:")
    print(f"  train: {len(df_train):,}")
    print(f"  val:   {len(df_val):,}")
    print(f"  test:  {len(df_test):,}")

    training_args = _make_training_args(
        output_dir=str(model_dir),
        num_train_epochs=float(tcfg["epochs"]),
        per_device_train_batch_size=int(tcfg["train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["eval_batch_size"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        gradient_accumulation_steps=int(tcfg["grad_accum_steps"]),
        evaluation_strategy="epoch",  # remaps to eval_strategy if needed
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        fp16=fp16,
        dataloader_num_workers=0,
        seed=seed,
        report_to=[],
    )

    trainer = _make_trainer(
        model=model,
        args=training_args,
        train_dataset=dtrain,
        eval_dataset=dval,
        tokenizer=tokenizer,  # will be ignored automatically if unsupported
        compute_metrics=compute_metrics,
    )

    print("\nTraining text baseline...")
    trainer.train()

    print("\nEvaluating on test split...")
    preds = trainer.predict(dtest)
    test_metrics = compute_metrics((preds.predictions, preds.label_ids))

    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(labels))],
        digits=4,
        zero_division=0,
    )
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=[id2label[i] for i in range(len(labels))],
        columns=[id2label[i] for i in range(len(labels))],
    ).to_csv(run_dir / "confusion_matrix.csv", encoding="utf-8")

    # Save model + tokenizer explicitly (so it works even if Trainer API changes)
    try:
        trainer.save_model(str(model_dir))
    except Exception:
        pass
    try:
        tokenizer.save_pretrained(str(model_dir))
    except Exception:
        pass

    print("\nTest metrics:")
    print(test_metrics)
    print(f"\nWrote results to: {run_dir}")
    print(f"Saved model to:   {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())