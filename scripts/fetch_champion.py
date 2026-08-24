"""Download and materialize the published champion weights into models/champion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from model_utils import count_parameters_torch, utc_now_iso, write_model_card, write_model_spec  # noqa: E402

DEFAULT_SOURCE = "moshew/bert-mini-sst2-distilled"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Hugging Face model id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "champion")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Fetching {args.source} …")
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    model = AutoModelForSequenceClassification.from_pretrained(args.source)
    n_params = count_parameters_torch(model)
    if args.dtype == "fp16":
        model = model.half()
    output_dir: Path = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_model_spec(
        output_dir,
        {
            "format": "transformers",
            "architecture": "bert-mini",
            "framework": "pytorch",
            "task": "sst2",
            "seed": 42,
            "model_name": "bert-mini-sst2-distilled",
            "parameter_count": n_params,
            "max_length": 128,
            "labels": {"0": "negative", "1": "positive"},
            "source": args.source,
            "quantization": args.dtype,
            "dtype": "float16" if args.dtype == "fp16" else "float32",
            "fetched_at": utc_now_iso(),
        },
    )
    write_model_card(
        output_dir,
        title="tiny-sota champion · BERT-Mini SST-2 (distilled)",
        metrics={"parameter_count": n_params, "source": args.source, "dtype": args.dtype},
        extra_markdown=(
            f"Public checkpoint `{args.source}` (BERT-Mini: 4 layers, hidden 256), "
            "independently re-evaluated by `src/evaluate.py` on GLUE SST-2 validation. "
            "Weights are stored in IEEE float16; the evaluation harness upcasts to "
            "float32 on CPU."
        ),
    )
    print(f"Wrote {output_dir} ({n_params:,} parameters, dtype={args.dtype})")


if __name__ == "__main__":
    main()
