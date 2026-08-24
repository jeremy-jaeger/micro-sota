# tiny-sota

The lowest-parameter (and lowest on-disk footprint) model that still exceeds **90% accuracy** on the SST-2 binary sentiment classification task from the GLUE benchmark.

[![Verification workflow](https://img.shields.io/badge/CI-verify_SST--2-0f766e)](.github/workflows/verify.yml)
<!-- ACCURACY_BADGE:START -->
![SST-2 accuracy](https://img.shields.io/badge/SST--2-90.71%25-brightgreen)
<!-- ACCURACY_BADGE:END -->
[![threshold](https://img.shields.io/badge/threshold-90%25-0f172a)](metrics.json)
[![license](https://img.shields.io/badge/license-MIT-informational)](LICENSE)
[![author](https://img.shields.io/badge/author-jeremy--jaeger-0f172a)](https://github.com/jeremy-jaeger)

**Author:** [Jeremy Jaeger](https://github.com/jeremy-jaeger) (sole author).

This repository **establishes** that claim with trainable candidates, and **continuously verifies** it: a seed-controlled evaluation harness loads the champion weights, runs the official GLUE SST-2 validation split, and fails CI if accuracy falls below 90%.

## Current champion

<!-- CHAMPION_TABLE:START -->

**Current champion:** `bert-mini-sst2-distilled`

| Parameters | Disk size | Accuracy | F1 | Latency / ex. | Verified |
| ---: | ---: | ---: | ---: | ---: | --- |
| 11,171,074 (11.1711 M) | 21.9960 MB | **90.71%** (791/872) | 0.9107 | 1.95 ms | 2026-08-24T09:01:20Z |

<!-- CHAMPION_TABLE:END -->

Live view: [leaderboard/index.html](leaderboard/index.html). Machine-readable record: [`metrics.json`](metrics.json). Shields.io endpoint: [`badge.json`](badge.json).

Among models with accuracy ≥ 90%, rank is **ascending parameter count**, then **ascending on-disk size**. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Exact reproduction

Python 3.10+ (3.11 or 3.12 recommended). From a clean clone:

```bash
python -m pip install -r requirements.txt
# GPU: install a CUDA build of torch first, then the remaining requirements.

# Rebuild the published champion from the public BERT-Mini SST-2 checkpoint.
python scripts/fetch_champion.py --dtype fp16

# Independent evaluation on the official validation split. Exits 1 if acc < 90%.
python src/evaluate.py \
  --model-dir models/champion \
  --task sst2 \
  --split validation \
  --seed 42 \
  --output-json /tmp/tiny-sota-eval.json
```

Verify the current committed champion without retraining:

```bash
python src/evaluate.py --model-dir models/champion --seed 42
```

Other training paths:

```bash
# From-scratch micro-Transformer (< 1 M parameters; may not clear 90%).
python src/train.py --config configs/micro_2l_64.yaml --output-dir models/runs/micro

# Fine-tune a public encoder (BERT-Tiny, ~4.4 M).
python src/train.py --config configs/hf_bert_tiny.yaml --output-dir models/runs/bert-tiny

# Distill DistilBERT-SST-2 into a micro student.
python src/distill.py --config configs/distill_micro.yaml --output-dir models/runs/distilled

# Quantize (INT8 dynamic, ONNX INT8, weight-only INT4).
python src/quantize.py --model-dir models/runs/bert-tiny --output-dir models/runs/bert-tiny-int8 --scheme int8-onnx
python src/quantize.py --model-dir models/runs/micro --output-dir models/runs/micro-int4 --scheme int4-weight
```

Refresh the public numbers after an evaluation:

```bash
python scripts/update_leaderboard.py --from-eval /tmp/tiny-sota-eval.json --promote
python scripts/plot_pareto.py
python scripts/serve_leaderboard.py --port 43147
```

## Methodology

**Dataset.** GLUE SST-2 is the binary (positive / negative) sentence-level formulation of the Stanford Sentiment Treebank (Socher et al., 2013; Wang et al., 2018). The Hugging Face `glue` / `nyu-mll/glue` `sst2` config is used. The **train** split (67,349 labeled sentences) is the only data seen by `train.py` and `distill.py`. Model selection uses a stratified internal holdout carved from train (default 5%). Reported numbers use the official **validation** split (872 labeled examples). The GLUE test split is unlabeled in public releases and is **not** used.

**Threshold.** A candidate passes if `accuracy >= 0.90` as a real number, i.e. at least **785 / 872** correct. Rounding 89.9% up to 90% is not accepted. `src/evaluate.py` exits with status 1 on failure.

**Seeds.** Default seed `42` is applied to `random`, NumPy, Python hash seeding, and PyTorch (including `use_deterministic_algorithms(True, warn_only=True)` and cuDNN deterministic mode). sklearn estimators receive the same `random_state`.

**Metrics.** Accuracy (primary), binary F1 / precision / recall (positive class = 1), Matthews correlation, wall-clock **milliseconds per example** (sequential, after warmup), parameter count, and serialized disk size of inference files.

**Hardware baseline.** Latency in `metrics.json` is measured on the machine that last ran the harness. CI uses GitHub-hosted `ubuntu-latest` CPU runners; a CUDA matrix job runs the same harness when a GPU is present and skips cleanly otherwise. Re-run locally with `--device cpu` or `--device cuda` to compare.

**Parameter accounting.** Neural models: `sum(p.numel() for p in model.parameters())`. Linear models: logistic coefficients + intercept + TF-IDF IDF vector (+ NB log-ratio if used). Tokenizer files are included in **disk size** but not in parameter count, matching how BERT embedding matrices *are* counted as parameters while `tokenizer.json` is not.

**Quantization.** INT8: PyTorch dynamic quantization (`nn.Linear` → qint8) and ONNX Runtime `quantize_dynamic`. 4-bit: per-row absmax weight packing (`int4-weight`) and, when CUDA + bitsandbytes are available, NF4 (`int4-bnb`). Quantization does not change parameter *count*; it is reported as a smaller **disk** footprint. The BERT-Mini ONNX INT8 export currently drops well below 90% in this harness, so the published champion is IEEE float16 (same accuracy as fp32, ~22 MB on disk).

**Verified sweep (seed 42, official SST-2 validation).** These numbers were produced by `src/evaluate.py` in this repository, not copied from model cards:

| Model | Parameters | Disk | Accuracy | ≥ 90% |
| --- | ---: | ---: | ---: | :---: |
| TF-IDF word+char LR | 0.18 M | 1.11 MB | 83.26% | no |
| NBSVM word+char | 0.30 M | 1.70 MB | 84.17% | no |
| BERT-Tiny distilled | 4.39 M | 17.4 MB | 83.37% | no |
| **BERT-Mini distilled fp16 (champion)** | **11.17 M** | **22.0 MB** | **90.71%** | **yes** |
| MiniLM-L6 SST-2 | 22.71 M | 87.3 MB | 90.14% | yes |
| DistilBERT SST-2 | 66.96 M | 256 MB | 91.06% | yes |

BERT-Mini is the smallest model in the sweep that clears the bar. Beating it requires `accuracy >= 0.90` and strictly fewer parameters (or equal parameters and a smaller file).

## Repository map

```
tiny-sota/
├── src/train.py          # linear | micro | HF fine-tune
├── src/distill.py        # teacher → student logits
├── src/quantize.py       # int8-dynamic | int8-onnx | int4-weight | int4-bnb
├── src/evaluate.py       # independent harness (CI entry point)
├── src/model_utils.py    # seeds, GLUE I/O, loaders, metrics schema
├── configs/              # YAML for each candidate family
├── models/champion/      # currently verified weights
├── scripts/update_leaderboard.py
├── scripts/fetch_champion.py
├── scripts/plot_pareto.py
├── .github/workflows/verify.yml
├── metrics.json
└── leaderboard/          # static page that reads metrics.json
```

## How to submit a new champion

Documented in [CONTRIBUTING.md](CONTRIBUTING.md). Summary: train without the official validation labels, run `src/evaluate.py` until it prints `PASS`, open a PR with `models/candidates/<name>/`, and let CI re-run the harness. A candidate is promoted only if it is ≥ 90% accurate **and** smaller (parameters, then disk) than the current champion.

## Citation

If you use this repository as a baseline, please cite the underlying dataset and this claim-verification setup:

```
Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality
Over a Sentiment Treebank. EMNLP.
Wang, A., et al. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform
for Natural Language Understanding. EMNLP Workshop.
Jaeger, J. tiny-sota: continuously verified lowest-parameter SST-2 model ≥ 90% accuracy. https://github.com/jeremy-jaeger/tiny-sota
```

## License

Copyright (c) 2026 Jeremy Jaeger. MIT. See [LICENSE](LICENSE).
