"""Update metrics.json, badge.json, the README champion table, and history."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from model_utils import dump_json, load_json, utc_now_iso  # noqa: E402

CHAMPION_START = "<!-- CHAMPION_TABLE:START -->"
CHAMPION_END = "<!-- CHAMPION_TABLE:END -->"
BADGE_START = "<!-- ACCURACY_BADGE:START -->"
BADGE_END = "<!-- ACCURACY_BADGE:END -->"


def _load_eval(path: Path) -> dict:
    data = load_json(path)
    required = {
        "accuracy",
        "f1",
        "parameter_count",
        "disk_size_mb",
        "latency_ms_per_example",
        "task",
        "n_examples",
        "n_correct",
    }
    missing = required - set(data)
    if missing:
        raise SystemExit(f"{path} is missing keys: {sorted(missing)}")
    return data


def _rank_tuple(row: dict) -> tuple:
    """Lower params, then lower disk, then higher accuracy is better (among ≥90%)."""
    return (
        int(row["parameter_count"]),
        float(row.get("disk_size_bytes") or row.get("disk_size_mb", 0) * 1024 * 1024),
        -float(row["accuracy"]),
    )


def _is_better(candidate: dict, champion: dict | None, threshold: float) -> bool:
    if candidate.get("task") != "sst2":
        return False
    if float(candidate["accuracy"]) < threshold:
        return False
    if champion is None:
        return True
    if float(champion.get("accuracy", 0)) < threshold:
        return True
    return _rank_tuple(candidate) < _rank_tuple(champion)


def _champion_markdown(metrics: dict) -> str:
    acc = float(metrics["accuracy"])
    params = int(metrics["parameter_count"])
    disk = float(metrics["disk_size_mb"])
    f1 = float(metrics["f1"])
    latency = float(metrics["latency_ms_per_example"])
    verified = metrics.get("verified_at", "")
    name = metrics.get("model_name", "champion")
    millions = params / 1e6
    return "\n".join(
        [
            CHAMPION_START,
            "",
            f"**Current champion:** `{name}`",
            "",
            "| Parameters | Disk size | Accuracy | F1 | Latency / ex. | Verified |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
            f"| {params:,} ({millions:.4f} M) | {disk:.4f} MB | **{acc:.2%}** "
            f"({metrics['n_correct']}/{metrics['n_examples']}) | {f1:.4f} | "
            f"{latency:.2f} ms | {verified} |",
            "",
            CHAMPION_END,
        ]
    )


def _badge_markdown(metrics: dict) -> str:
    acc = float(metrics["accuracy"]) * 100
    color = "brightgreen" if acc >= 90 else "red"
    message = f"{acc:.2f}%".replace("%", "%25")
    url = f"https://img.shields.io/badge/SST--2-{message}-{color}"
    return "\n".join(
        [
            BADGE_START,
            f"![SST-2 accuracy]({url})",
            BADGE_END,
        ]
    )


def _patch_readme(readme: Path, metrics: dict) -> None:
    text = readme.read_text(encoding="utf-8")
    if CHAMPION_START in text and CHAMPION_END in text:
        text = re.sub(
            re.escape(CHAMPION_START) + r".*?" + re.escape(CHAMPION_END),
            _champion_markdown(metrics),
            text,
            count=1,
            flags=re.S,
        )
    if BADGE_START in text and BADGE_END in text:
        text = re.sub(
            re.escape(BADGE_START) + r".*?" + re.escape(BADGE_END),
            _badge_markdown(metrics),
            text,
            count=1,
            flags=re.S,
        )
    readme.write_text(text, encoding="utf-8")


def _append_history(row: dict) -> None:
    history = ROOT / "leaderboard" / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_badge_json(metrics: dict, path: Path) -> None:
    acc = float(metrics["accuracy"])
    dump_json(
        path,
        {
            "schemaVersion": 1,
            "label": "SST-2 accuracy",
            "message": f"{acc:.2%}",
            "color": "brightgreen" if acc >= 0.90 else "red",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-eval",
        type=Path,
        required=True,
        help="JSON file produced by src/evaluate.py --output-json",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "metrics.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="If the candidate beats the champion, copy its weights to models/champion.",
    )
    parser.add_argument(
        "--force-champion",
        action="store_true",
        help="Write this evaluation as champion even if it does not rank better.",
    )
    args = parser.parse_args()

    eval_path = args.from_eval if args.from_eval.is_absolute() else (ROOT / args.from_eval)
    candidate = _load_eval(eval_path)
    metrics_path = args.metrics_out if args.metrics_out.is_absolute() else (ROOT / args.metrics_out)

    current = None
    if metrics_path.exists():
        current = load_json(metrics_path)
        if "champion" in current:
            current = current["champion"]

    promote = args.force_champion or _is_better(candidate, current, args.threshold)
    comparisons = []
    if metrics_path.exists():
        blob = load_json(metrics_path)
        comparisons = list(blob.get("comparisons") or [])
    comparisons.append(
        {
            "model_name": candidate.get("model_name"),
            "parameter_count": candidate.get("parameter_count"),
            "disk_size_mb": candidate.get("disk_size_mb"),
            "accuracy": candidate.get("accuracy"),
            "f1": candidate.get("f1"),
            "verified_at": candidate.get("verified_at"),
            "model_dir": candidate.get("model_dir"),
        }
    )
    # Keep a short unique history of comparisons by model_dir.
    dedup: dict[str, dict] = {}
    for row in comparisons:
        dedup[str(row.get("model_dir") or row.get("model_name"))] = row
    comparisons = list(dedup.values())

    champion = candidate if promote else (current or candidate)
    payload = {
        "task": "sst2",
        "threshold": args.threshold,
        "claim": (
            "The lowest-parameter (and lowest on-disk footprint) model that still "
            "exceeds 90% accuracy on the SST-2 binary sentiment classification "
            "task from the GLUE benchmark."
        ),
        "updated_at": utc_now_iso(),
        "champion": champion,
        "comparisons": comparisons,
    }
    dump_json(metrics_path, payload)
    dump_json(ROOT / "leaderboard" / "metrics.json", payload)
    _write_badge_json(champion, ROOT / "badge.json")
    _append_history({**candidate, "recorded_at": utc_now_iso(), "promoted": promote})
    _patch_readme(ROOT / "README.md", champion)

    if promote and args.promote:
        src = ROOT / candidate["model_dir"] if not Path(candidate["model_dir"]).is_absolute() else Path(
            candidate["model_dir"]
        )
        dest = ROOT / "models" / "champion"
        if src.resolve() != dest.resolve() and src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            print(f"Promoted {src} -> {dest}")

    status = "NEW CHAMPION" if promote else "recorded (not champion)"
    print(
        f"{status}: {champion.get('model_name')}  "
        f"acc={float(champion['accuracy']):.4%}  "
        f"params={int(champion['parameter_count']):,}  "
        f"disk={float(champion['disk_size_mb']):.4f} MB"
    )


if __name__ == "__main__":
    main()
