# Champion weights

`models/champion/` holds the **current verified champion**: the lowest-parameter
model in this repository that still exceeds 90% accuracy on GLUE SST-2
validation (seed 42).

The published champion is **BERT-Mini** (4 layers, hidden 256, 11,171,074
parameters) stored as float16, independently re-evaluated at **90.71%**
(791/872). Source checkpoint: `moshew/bert-mini-sst2-distilled`.

CI (`.github/workflows/verify.yml`) loads this directory, runs
`src/evaluate.py`, and fails the build if accuracy drops below 0.90. If the
weight file is absent, CI runs `scripts/fetch_champion.py`.

## Layout

A legal champion (or candidate) directory contains:

| File | Role |
| --- | --- |
| `model_spec.json` | Loader dispatch (`format`: `sklearn` / `micro` / `transformers` / `onnx`) |
| weights | `model.joblib`, `pytorch_model.pt` / `model.safetensors`, or `*.onnx` |
| tokenizer | Hugging Face tokenizer files, or embedded in the sklearn pipeline |
| `README.md` | Model card (intended use, metrics, seed) |

Training runs belong in `models/runs/` (gitignored). PR candidates belong in
`models/candidates/<name>/`.

## Replacing the champion

See [CONTRIBUTING.md](../CONTRIBUTING.md). In short: beat the ranking rule,
open a PR, wait for CI, then:

```bash
python src/evaluate.py --model-dir models/candidates/yourname --output-json /tmp/eval.json
python scripts/update_leaderboard.py --from-eval /tmp/eval.json --promote
```

## Download / rebuild

```bash
python scripts/fetch_champion.py --dtype fp16
python src/evaluate.py --model-dir models/champion --seed 42
```

Linear baselines, from-scratch micro-Transformers, DistilBERT fine-tunes, and
distillation students use the YAML files under `configs/`. They are recorded
in `metrics.json` comparisons when they are evaluated; they replace the
champion only if they pass the ranking rule.

