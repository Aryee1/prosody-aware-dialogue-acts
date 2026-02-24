# Prosody-Aware Dialogue Act Modeling
Multimodal dialogue act classification on the AMI Meeting Corpus using text, audio prosody, and wav2vec2 embeddings.

This repo builds an end-to-end, reproducible pipeline:
- Build an utterance-level dataset from AMI manual annotations (dialogue acts + words)
- Train a text baseline (DistilBERT)
- Download Mix-Headset audio for a subset of meetings, extract prosody features, and train a prosody baseline
- Compare unimodal vs multimodal fusion with an ablation table
- Run a small error analysis script

No raw AMI audio/video is committed to the repo.

## Dataset
AMI Meeting Corpus.
- Download page (manual annotations and signals): https://groups.inf.ed.ac.uk/ami/download/
- Dialogue act manual: https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/dialogue_acts_manual_1.0.pdf
- Mix-Headset audio mirror: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/HeadsetAudio/

## Results (commit-safe)
All numbers are stored in `reports/results_public/`.

### Full AMI (text-only baseline)
| Setup | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Text only (DistilBERT) | 0.653 | 0.457 | 0.638 |

File: `reports/results_public/text_only_full/test_metrics.json`

### Audio subset ablations (30 meetings with Mix-Headset audio)
These comparisons are computed on the same subset: utterances with available audio segments.

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

## Repo layout
- `configs/` configs for Windows and Colab
- `scripts/` runnable entry points
- `src/` shared utilities
- `reports/results_public/` small, commit-safe results

## How to reproduce on Colab (GPU)
1) Set GPU: Runtime → Change runtime type → GPU
2) Run these cells:

```bash
!git clone https://github.com/Aryee1/prosody-aware-dialogue-acts.git
%cd prosody-aware-dialogue-acts
!pip -q install -r requirements.txt
!nvidia-smi
