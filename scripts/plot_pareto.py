"""Generate a parameter-count vs accuracy Pareto plot from leaderboard history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_points(history: Path, metrics: Path) -> list[dict]:
    rows: list[dict] = []
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if metrics.exists():
        blob = json.loads(metrics.read_text(encoding="utf-8"))
        if "champion" in blob:
            rows.append(blob["champion"])
        for row in blob.get("comparisons") or []:
            rows.append(row)
    # Dedupe by (name, params, accuracy)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        if "accuracy" not in row or "parameter_count" not in row:
            continue
        key = (
            row.get("model_name"),
            int(row["parameter_count"]),
            round(float(row["accuracy"]), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _pareto_front(points: list[dict]) -> list[dict]:
    """Models that are not dominated on (minimize params, maximize accuracy)."""
    front: list[dict] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_or_equal_params = int(q["parameter_count"]) <= int(p["parameter_count"])
            better_or_equal_acc = float(q["accuracy"]) >= float(p["accuracy"])
            strictly_better = int(q["parameter_count"]) < int(p["parameter_count"]) or float(
                q["accuracy"]
            ) > float(p["accuracy"])
            if better_or_equal_params and better_or_equal_acc and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda r: int(r["parameter_count"]))
    return front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "leaderboard" / "history.jsonl")
    parser.add_argument("--metrics", type=Path, default=ROOT / "metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "leaderboard" / "pareto.png")
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    points = _load_points(args.history, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)
    ax.axhline(args.threshold, color="#b45309", linestyle="--", linewidth=1.0, label="90% threshold")
    if not points:
        ax.set_title("No evaluations recorded yet")
    else:
        xs = [int(p["parameter_count"]) for p in points]
        ys = [float(p["accuracy"]) for p in points]
        ax.scatter(xs, ys, s=42, c="#0f766e", alpha=0.85, label="evaluated models", zorder=3)
        front = _pareto_front(points)
        if front:
            ax.plot(
                [int(p["parameter_count"]) for p in front],
                [float(p["accuracy"]) for p in front],
                color="#111827",
                linewidth=1.2,
                marker="o",
                label="Pareto front",
                zorder=4,
            )
        for p in points:
            ax.annotate(
                str(p.get("model_name") or ""),
                (int(p["parameter_count"]), float(p["accuracy"])),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="#374151",
            )
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("SST-2 validation accuracy")
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("micro-sota · parameter count vs. SST-2 accuracy")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
