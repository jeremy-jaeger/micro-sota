"""Run a saved candidate on free-form sentences so you can see it classify live."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import (  # noqa: E402
    TASK_REGISTRY,
    get_device,
    load_predictor,
    read_model_spec,
    repo_root,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Classify text with a micro-sota model directory (champion by default).",
)
LOGGER = logging.getLogger("micro_sota.predict")
console = Console()

DEMO_SENTENCES = [
    "a stirring, funny and finally transporting re-imagining of beauty and the beast",
    "this is a painfully amateurish movie",
    "the acting is wooden but the plot kept me watching",
    "I loved every minute of it",
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_model_dir(model_dir: Path) -> Path:
    if model_dir.exists():
        return model_dir.resolve()
    alt = repo_root() / model_dir
    if alt.exists():
        return alt.resolve()
    raise typer.BadParameter(f"Model directory not found: {model_dir}")


def _label_names(model_dir: Path, task: str) -> dict[int, str]:
    spec = read_model_spec(model_dir)
    raw = spec.get("labels") or spec.get("id2label") or {}
    names: dict[int, str] = {}
    for key, value in raw.items():
        names[int(key)] = str(value)
    if names:
        return names
    registry = TASK_REGISTRY.get(task, {})
    fallback = registry.get("id2label") or {0: "negative", 1: "positive"}
    return {int(k): str(v) for k, v in fallback.items()}


def classify_texts(
    model_dir: Path,
    texts: list[str],
    *,
    device: str = "cpu",
    task: str = "sst2",
) -> list[dict[str, object]]:
    """Return one record per sentence: label, probabilities, and logits."""
    predictor = load_predictor(model_dir, device=device)
    names = _label_names(model_dir, task)
    logits = np.asarray(predictor.predict_logits(texts), dtype=np.float64)
    proba = predictor.predict_proba(texts)
    preds = np.argmax(logits, axis=-1).astype(np.int64)
    records: list[dict[str, object]] = []
    for text, pred, row_logits, row_proba in zip(texts, preds, logits, proba, strict=True):
        label_id = int(pred)
        records.append(
            {
                "text": text,
                "label_id": label_id,
                "label": names.get(label_id, str(label_id)),
                "confidence": float(row_proba[label_id]),
                "probabilities": {
                    names.get(i, str(i)): float(p) for i, p in enumerate(row_proba)
                },
                "logits": [float(x) for x in row_logits],
            }
        )
    predictor.close()
    return records


def print_records(records: list[dict[str, object]], model_name: str) -> None:
    table = Table(title=f"Live inference · {model_name}")
    table.add_column("Sentiment", style="bold")
    table.add_column("Confidence")
    table.add_column("P(neg)")
    table.add_column("P(pos)")
    table.add_column("Text")
    for rec in records:
        probs = rec["probabilities"]
        assert isinstance(probs, dict)
        table.add_row(
            str(rec["label"]),
            f"{float(rec['confidence']):.1%}",
            f"{float(probs.get('negative', list(probs.values())[0])):.3f}",
            f"{float(list(probs.values())[-1] if 'positive' not in probs else probs['positive']):.3f}",
            str(rec["text"]),
        )
    console.print(table)


@app.command()
def main(
    texts: Optional[list[str]] = typer.Option(
        None,
        "--text",
        "-t",
        help="Sentence to classify. Repeat the flag for several examples.",
    ),
    model_dir: Path = typer.Option(
        Path("models/champion"),
        "--model-dir",
        "-m",
        help="Directory containing weights + model_spec.json.",
    ),
    task: str = typer.Option("sst2", "--task"),
    device: str = typer.Option("auto", "--device"),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Run a handful of built-in movie-review sentences.",
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", "-o", help="Write predictions as JSON."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Classify ``--text`` sentences, or ``--demo`` examples, with the loaded model."""
    _setup_logging(verbose)
    model_dir = _resolve_model_dir(model_dir)
    resolved_device = get_device(device)
    payload_texts = list(texts or [])
    if demo:
        payload_texts.extend(DEMO_SENTENCES)
    if not payload_texts:
        console.print(
            "[dim]No --text given; classifying built-in demo sentences. "
            "Pass -t 'your sentence' to try your own.[/dim]"
        )
        payload_texts = list(DEMO_SENTENCES)

    spec = read_model_spec(model_dir)
    records = classify_texts(
        model_dir,
        payload_texts,
        device=resolved_device,
        task=task,
    )
    print_records(records, str(spec.get("model_name") or model_dir.name))
    console.print_json(json.dumps(records, indent=2))
    if output_json is not None:
        out = output_json if output_json.is_absolute() else (repo_root() / output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    app()
