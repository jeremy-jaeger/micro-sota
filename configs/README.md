# Configs for tiny-sota training, distillation, and additional GLUE tasks.
#
# SST-2 (the public claim)
#   linear_ngram.yaml      TF-IDF word+char LogisticRegression
#   linear_nbsvm.yaml      Same features with Wang & Manning NB ratio
#   micro_2l_64.yaml       From-scratch micro-Transformer (~0.3 M params)
#   micro_4l_128.yaml      Larger micro-Transformer with factorized embeddings
#   hf_bert_tiny.yaml      Fine-tune prajjwal1/bert-tiny (~4.4 M)
#   hf_bert_mini.yaml      Fine-tune prajjwal1/bert-mini (~11.2 M)
#   hf_distilbert.yaml     Fine-tune DistilBERT (reference, ~66 M)
#   distill_micro.yaml     DistilBERT teacher → micro student
#   distill_bert_tiny.yaml DistilBERT teacher → BERT-Tiny student
#
# Other GLUE (harness is task-extensible; no 90% claim)
#   cola_linear.yaml
#
# Train:
#   python src/train.py --config configs/linear_ngram.yaml --output-dir models/runs/linear
# Distill:
#   python src/distill.py --config configs/distill_micro.yaml --output-dir models/runs/distilled
