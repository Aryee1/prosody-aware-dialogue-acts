# Prosody-Aware Dialogue Act Modeling (AMI)

## Project Overview
This project builds a reproducible, end-to-end pipeline for **dialogue act classification** in multiparty meetings. It combines:
- **Text** (DistilBERT fine-tuning)
- **Audio prosody** (energy, silence, speaking-rate proxy, pitch proxies) extracted from AMI Mix-Headset audio
- **wav2vec2 utterance embeddings** (with and without PCA compression)

The goal is not just a single score, but a clear **unimodal vs multimodal comparison**, with ablations and a short error analysis that explains where models struggle in real meeting dialogue.

No raw AMI audio/video is committed to the repo. Only scripts, configs, and small "commit-safe" result files are tracked.

---

## Dataset
AMI Meeting Corpus resources:
- Download page: https://groups.inf.ed.ac.uk/ami/download/
- Dialogue act manual: https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/dialogue_acts_manual_1.0.pdf
- Mix-Headset audio mirror: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/HeadsetAudio/

---

## Results (commit-safe)
All reported numbers are stored in `reports/results_public/`.

### Full AMI (text-only baseline)
| Setup | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Text only (DistilBERT) | 0.653 | 0.457 | 0.638 |

File: `reports/results_public/text_only_full/test_metrics.json`

### Audio Subset Ablations (30 meetings with Mix-Headset audio)
These comparisons are computed on the same subset: utterances that have extracted audio segments.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Text only (subset) | 0.647 | 0.439 | 0.630 |
| Prosody only (subset) | 0.168 | 0.103 | 0.157 |
| Text + Prosody (late fusion) | 0.603 | 0.477 | 0.616 |
| Text + wav2vec (late fusion) | 0.549 | 0.407 | 0.563 |
| Text + Prosody + wav2vec (late fusion) | 0.547 | 0.405 | 0.562 |
| Text + wav2vec (PCA128) | 0.581 | 0.430 | 0.597 |
| Text + Prosody + wav2vec (PCA128) | 0.581 | 0.438 | 0.596 |

Files:
- `reports/results_public/ablation_mix30/ablation_table.md`
- `reports/results_public/ablation_mix30_w2v/ablation_table.md`
- `reports/results_public/ablation_mix30_w2v_pca/ablation_table.md`

**Takeaways:**
- Prosody improves Macro F1 over text-only (0.439 → 0.477).
- Raw wav2vec embeddings can overfit; PCA compression improves stability.

---

## Repo Layout
```
configs/       # configs for Windows and Colab
scripts/       # runnable entry points
src/           # shared utilities
reports/
  results_public/   # small, commit-safe metrics and tables
```

---

## How to Reproduce on Colab (GPU)

### 1) Setup
In Colab: **Runtime → Change runtime type → GPU**

```bash
!git clone https://github.com/Aryee1/prosody-aware-dialogue-acts.git
%cd prosody-aware-dialogue-acts
!pip -q install -r requirements.txt
!nvidia-smi
```

### 2) Build utterances from AMI manual annotations

```bash
!python scripts/download_ami_annotations.py --config configs/colab.yaml
!python scripts/prepare_ami_subset.py --config configs/colab.yaml
```

### 3) Train the text baseline

```bash
!python scripts/train_text_baseline.py --config configs/experiments/text_only.yaml
```

### 4) Download audio subset and run prosody pipeline

```bash
!python scripts/download_ami_mix_headset.py --config configs/colab.yaml --max_meetings 30
!python scripts/extract_audio_prosody.py --config configs/colab.yaml --audio_dir data/raw/ami_audio_mix
!python scripts/train_prosody_baseline.py --config configs/colab.yaml
!python scripts/run_mix_ablation.py --config configs/colab.yaml
```

### 5) wav2vec2 utterance embeddings + ablations

```bash
!python scripts/extract_wav2vec_utterance.py --config configs/colab.yaml
!python scripts/run_mix_ablation_w2v.py --config configs/colab.yaml
!python scripts/run_mix_ablation_w2v_pca.py --config configs/colab.yaml --pca_dim 128
```

---

## Error Analysis (audio subset)

Generate a small error report (top confusions + misclassified examples):

```bash
!python scripts/error_analysis.py \
  --config configs/colab.yaml \
  --text_model_dir models/checkpoints/text_only_distilbert \
  --mode text_plus_prosody \
  --n 30
```

Outputs are written to:
- `reports/error_analysis/text_plus_prosody/top_confusions.csv`
- `reports/error_analysis/text_plus_prosody/misclassified_examples.csv`

> Note: `reports/error_analysis/` is ignored by git (generated outputs).

---

## References
- AMI Meeting Corpus (Carletta et al.)
- DistilBERT (Sanh et al.)
- wav2vec 2.0 (Baevski et al.)
- AMI dialogue act annotation manual
