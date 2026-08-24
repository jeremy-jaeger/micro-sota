"""Independent, seed-controlled evaluation harness for micro-sota candidates.

This module does not import training code paths other than model loaders. CI
verifies any submitted weight directory by pointing ``--model-dir`` at it.
"""

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
    EvalMetrics,
    bytes_to_mb,
    classification_metrics,
    directory_inference_size_bytes,
    dump_json,
    get_device,
    load_glue_split,
    load_predictor,
    measure_latency_ms,
    repo_root,
    set_seed,
    utc_now_iso,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evaluate a candidate on a GLUE split and enforce the accuracy threshold.",
)
LOGGER = logging.getLogger("micro_sota.evaluate")
console = Console()


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


def evaluate_model(
    model_dir: Path,
    *,
    task: str = "sst2",
    split: str = "validation",
    seed: int = 42,
    device: str = "cpu",
    threshold: float | None = None,
    batch_size: int = 32,
) -> EvalMetrics:
    """Run the official split and return a structured metrics record."""
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task {task!r}")
    set_seed(seed)
    resolved_device = get_device(device)
    meta = TASK_REGISTRY[task]
    if threshold is None:
        threshold = float(meta["threshold"] if meta["threshold"] is not None else 0.0)

    predictor = load_predictor(model_dir, device=resolved_device)
    texts, labels = load_glue_split(task, split)
    LOGGER.info(
        "Evaluating %s on %s/%s (%d examples) device=%s",
        predictor.name,
        task,
        split,
        len(texts),
        resolved_device,
    )

    preds: list[int] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch_pred = np.asarray(predictor.predict(chunk), dtype=np.int64).tolist()
        preds.extend(batch_pred)
    if len(preds) != len(labels):
        raise RuntimeError(f"Prediction count {len(preds)} != label count {len(labels)}")

    clf = classification_metrics(labels, preds)
    latency = measure_latency_ms(predictor.predict, texts)
    n_params = int(predictor.parameter_count())
    disk_bytes = directory_inference_size_bytes(model_dir)
    accuracy = clf["accuracy"]
    meets = bool(accuracy >= threshold) if threshold > 0 else True

    try:
        rel_dir = str(model_dir.resolve().relative_to(repo_root()))
    except ValueError:
        rel_dir = str(model_dir.resolve())

    metrics = EvalMetrics(
        task=task,
        split=split,
        accuracy=float(accuracy),
        f1=float(clf["f1"]),
        precision=float(clf["precision"]),
        recall=float(clf["recall"]),
        matthews_correlation=float(clf["matthews_correlation"]),
        n_examples=int(clf["n_examples"]),
        n_correct=int(clf["n_correct"]),
        parameter_count=n_params,
        parameter_count_millions=round(n_params / 1e6, 6),
        disk_size_bytes=disk_bytes,
        disk_size_mb=bytes_to_mb(disk_bytes),
        latency_ms_per_example=float(latency),
        device=resolved_device,
        seed=seed,
        model_name=str(predictor.name),
        model_format=str(predictor.format),
        model_dir=rel_dir,
        meets_threshold=meets,
        threshold=float(threshold),
        verified_at=utc_now_iso(),
        extras={
            "batch_size": batch_size,
            "primary_metric": meta["primary_metric"],
            "display_name": meta["display_name"],
        },
    )
    predictor.close()
    return metrics


def print_human_table(metrics: EvalMetrics) -> None:
    payload = metrics.to_dict()
    table = Table(title=f"{payload['display_name']} · {metrics.split} · seed={metrics.seed}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    rows = [
        ("model", metrics.model_name),
        ("format", metrics.model_format),
        ("parameters", f"{metrics.parameter_count:,} ({metrics.parameter_count_millions:.4f} M)"),
        ("disk_size_mb", f"{metrics.disk_size_mb:.4f} MB"),
        ("accuracy", f"{metrics.accuracy:.4%}  ({metrics.n_correct}/{metrics.n_examples})"),
        ("f1", f"{metrics.f1:.4f}"),
        ("precision", f"{metrics.precision:.4f}"),
        ("recall", f"{metrics.recall:.4f}"),
        ("matthews_correlation", f"{metrics.matthews_correlation:.4f}"),
        ("latency_ms_per_example", f"{metrics.latency_ms_per_example:.4f} ms"),
        ("device", metrics.device),
        ("threshold", f"{metrics.threshold:.2%}"),
        ("meets_threshold", "YES" if metrics.meets_threshold else "NO"),
        ("verified_at", metrics.verified_at),
    ]
    for key, value in rows:
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def main(
    model_dir: Path = typer.Option(
        Path("models/champion"),
        "--model-dir",
        "-m",
        help="Directory containing weights + model_spec.json.",
    ),
    task: str = typer.Option("sst2", "--task"),
    split: str = typer.Option("validation", "--split"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device"),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Minimum accuracy (default: 0.90 for sst2). Ignored when <= 0.",
    ),
    batch_size: int = typer.Option(32, "--batch-size"),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", "-o", help="Write the metrics record to this path."
    ),
    fail_below_threshold: bool = typer.Option(
        True,
        "--fail-below-threshold/--no-fail-below-threshold",
        help="Exit with status 1 when accuracy is below the threshold.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate ``model_dir`` and print JSON + a human-readable table."""
    _setup_logging(verbose)
    model_dir = _resolve_model_dir(model_dir)
    if task == "sst2" and threshold is None:
        threshold = float(TASK_REGISTRY["sst2"]["threshold"])
    elif threshold is None:
        threshold = 0.0

    metrics = evaluate_model(
        model_dir,
        task=task,
        split=split,
        seed=seed,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
    )
    payload = metrics.to_dict()
    print_human_table(metrics)
    console.print_json(json.dumps(payload, indent=2))

    if output_json is not None:
        out = output_json if output_json.is_absolute() else (repo_root() / output_json)
        dump_json(out, payload)
        LOGGER.info("Wrote %s", out)

    if fail_below_threshold and threshold > 0 and not metrics.meets_threshold:
        console.print(
            f"[bold red]FAIL[/] accuracy {metrics.accuracy:.4%} "
            f"< threshold {metrics.threshold:.2%}"
        )
        raise typer.Exit(code=1)
    console.print("[bold green]PASS[/] accuracy threshold satisfied.")


if __name__ == "__main__":
    app()
