# Contributing to tiny-sota

This repository maintains a **public, continuously verified claim**:

> The lowest-parameter (and lowest on-disk footprint) model that still exceeds
> 90% accuracy on GLUE SST-2.

We welcome candidates that beat the current champion on the ranking rule below.
Please keep training, evaluation, and reporting strictly separated.

## Ranking rule

A candidate **replaces the champion** if and only if all of the following hold
on the official GLUE SST-2 **validation** split (872 labeled examples), seed 42:

1. `accuracy >= 0.90` (at least 785/872 correct).
2. Strictly fewer **parameters** than the current champion, or equal
   parameters and a strictly smaller **on-disk footprint**.
3. Ties on both size metrics are broken by higher accuracy, then earlier
   `verified_at`.

Parameter count is the number of numeric values required at inference
(neural weights, or logistic coefficients + IDF + intercept for linear
models). On-disk size is the total byte size of files needed to load the
model (weights, tokenizer, config) — not the model card.

## Submitting a candidate (PR process)

1. **Train** with a fixed seed using one of the repo scripts. Do **not** peek
   at the official validation split for model selection; use the internal
   holdout that `train.py` / `distill.py` already carves from `train`.

   ```bash
   python src/train.py --config configs/linear_ngram.yaml --output-dir models/candidates/yourname
   # or
   python src/distill.py --config configs/distill_micro.yaml --output-dir models/candidates/yourname
   python src/quantize.py --model-dir models/candidates/yourname --output-dir models/candidates/yourname-int8 --scheme int8-onnx
   ```

2. **Evaluate** with the independent harness. This is the only number that
   counts. The process must exit 0:

   ```bash
   python src/evaluate.py \
     --model-dir models/candidates/yourname \
     --task sst2 --split validation --seed 42 \
     --output-json models/candidates/yourname/eval.json
   ```

3. Open a pull request that contains:

   - `models/candidates/<yourname>/` — weights, tokenizer, `model_spec.json`,
     model card, and `eval.json`
   - The training config you used (or a new file under `configs/`)
   - A short write-up: architecture, parameter count, disk size, seed,
     hardware, wall-clock, and any distillation teacher
   - Confirmation that you did not train or tune on the official validation
     labels

4. CI (`.github/workflows/verify.yml`) re-runs `src/evaluate.py` on the
   candidate (use `workflow_dispatch` with `model_dir`, or temporarily point
   the workflow at your path in the PR if you are proposing a new champion).
   The job **fails** if accuracy is below 90%.

5. A maintainer promotes the candidate with:

   ```bash
   python scripts/update_leaderboard.py --from-eval models/candidates/yourname/eval.json --promote
   ```

   That copies weights to `models/champion/`, rewrites `metrics.json`, and
   refreshes the README table.

## Code standards

- Python 3.10+, type hints, docstrings on public functions.
- No hardcoded absolute paths. Resolve relative to the repo root.
- `set_seed(42)` (or the seed in the config) at the start of every entry point.
- Evaluation code must stay independent of training: `evaluate.py` may load
  weights but must not import a training loop.
- Prefer adding a YAML config over adding flags that exist only in your fork.

```bash
python -m pytest tests -q
ruff check src tests scripts
```

## Additional GLUE tasks

The harness already understands `cola` (`--task cola`). Additional tasks should
land as a new entry in `TASK_REGISTRY` (`src/model_utils.py`) plus a config
under `configs/`. They do **not** affect the SST-2 90% claim unless the README
is explicitly updated.

## License

By contributing you agree that your work is released under the MIT License.
