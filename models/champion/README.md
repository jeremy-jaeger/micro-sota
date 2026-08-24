# micro-sota champion · BERT-Mini SST-2 (distilled, float16)

This artifact is part of **micro-sota**: the lowest-parameter model that
still exceeds 90% accuracy on GLUE SST-2, continuously verified in CI.

## Reported metrics

```json
{
  "source": "moshew/bert-mini-sst2-distilled",
  "architecture": "BERT-Mini (L=4, H=256, A=4)",
  "parameter_count": 11171074,
  "dtype": "float16",
  "task": "sst2",
  "seed": 42
}
```

## Intended use

Binary sentiment classification of English movie-review sentences
(Stanford Sentiment Treebank, GLUE SST-2 formulation). Not intended
for general-domain sentiment, toxicity, or multilingual text.

## Evaluation

Accuracy is measured on the official GLUE SST-2 **validation** split
(872 labeled examples) with a fixed seed. The unlabeled GLUE test
split is not used; labels for that split are not public.

## Architecture

BERT-Mini (Turc et al., *Well-Read Students Learn Better*): 4 layers, hidden size 256,
4 attention heads. Task-specific weights from `moshew/bert-mini-sst2-distilled`.
This repository stores IEEE **float16** tensors; `src/evaluate.py` upcasts to
float32 on CPU so results match the fp32 checkpoint (90.71% on SST-2 validation).

## How this champion was selected

Independent evaluation of public checkpoints and trained baselines with
`src/evaluate.py` (seed 42, official validation split):

| Model | Params | Accuracy |
| --- | ---: | ---: |
| TF-IDF + LR | 0.18 M | 83.3% |
| NBSVM | 0.30 M | 84.2% |
| BERT-Tiny distilled | 4.39 M | 83.4% |
| **BERT-Mini distilled (this)** | **11.17 M** | **90.71%** |
| MiniLM-L6 SST-2 | 22.71 M | 90.14% |
| DistilBERT SST-2 | 66.96 M | 91.06% |

BERT-Mini is the **smallest** model in this sweep that clears the 90% bar.
ONNX INT8 export of the same graph currently falls well below 90% and is **not**
used as the champion.

Rebuild with `python scripts/fetch_champion.py`.

Generated 2026-08-24T09:01:15Z.
