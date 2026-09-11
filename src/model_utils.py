"""Shared utilities for micro-sota: seeds, paths, dataset loading, metrics I/O."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

TaskName = Literal["sst2", "cola"]

TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "sst2": {
        "glue_config": "sst2",
        "text_field": "sentence",
        "label_field": "label",
        "num_labels": 2,
        "display_name": "GLUE SST-2",
        "threshold": 0.90,
        "primary_metric": "accuracy",
        "id2label": {0: "negative", 1: "positive"},
    },
    "cola": {
        "glue_config": "cola",
        "text_field": "sentence",
        "label_field": "label",
        "num_labels": 2,
        "display_name": "GLUE CoLA",
        "threshold": None,
        "primary_metric": "matthews_correlation",
        "id2label": {0: "unacceptable", 1: "acceptable"},
    },
}

INFERENCE_WEIGHT_SUFFIXES = (
    ".joblib",
    ".pkl",
    ".npz",
    ".onnx",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".int4.npz",
)
INFERENCE_TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "spiece.model",
}
INFERENCE_CONFIG_NAMES = {
    "config.json",
    "model_spec.json",
    "micro_config.json",
    "quantize_config.json",
    "preprocessor_config.json",
}
EXCLUDE_FROM_DISK = {
    "README.md",
    "MODEL_CARD.md",
    "eval.json",
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "train_results.json",
    "holdout_metrics.json",
    "linear_stats.json",
}


def repo_root() -> Path:
    """Return the repository root (directory that contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a user-supplied path relative to ``base`` (default: repo root)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ((base or repo_root()) / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping, got {type(data)}")
    return data


def dump_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=False)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(requested: str = "auto") -> str:
    """Resolve ``cpu``, ``cuda``, ``mps``, or ``auto`` to a concrete device string."""
    requested = requested.lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return requested


def count_parameters_torch(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def directory_inference_size_bytes(model_dir: str | Path) -> int:
    """Bytes of files required to load and run the serialized model."""
    root = Path(model_dir)
    if root.is_file():
        return root.stat().st_size
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in EXCLUDE_FROM_DISK or name.startswith("checkpoint"):
            continue
        rel_suffix_ok = path.suffix in INFERENCE_WEIGHT_SUFFIXES
        if (
            name in INFERENCE_TOKENIZER_NAMES
            or name in INFERENCE_CONFIG_NAMES
            or rel_suffix_ok
            or path.suffix in {".json", ".txt", ".model"}
        ):
            total += path.stat().st_size
    return total


def bytes_to_mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 4)


@dataclass
class EvalMetrics:
    """Machine-readable evaluation record written by ``evaluate.py``."""

    task: str
    split: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    matthews_correlation: float | None
    n_examples: int
    n_correct: int
    parameter_count: int
    parameter_count_millions: float
    disk_size_bytes: int
    disk_size_mb: float
    latency_ms_per_example: float
    device: str
    seed: int
    model_name: str
    model_format: str
    model_dir: str
    meets_threshold: bool
    threshold: float
    verified_at: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extras = payload.pop("extras") or {}
        payload.update(extras)
        return payload


def write_model_spec(model_dir: str | Path, spec: dict[str, Any]) -> Path:
    path = Path(model_dir) / "model_spec.json"
    dump_json(path, spec)
    return path


def read_model_spec(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "model_spec.json"
    if not path.exists():
        return infer_model_spec(Path(model_dir))
    return load_json(path)


def infer_model_spec(model_dir: Path) -> dict[str, Any]:
    """Best-effort spec inference so submitted HF folders still evaluate."""
    if (model_dir / "model.joblib").exists() or (model_dir / "linear_model.joblib").exists():
        return {"format": "sklearn", "task": "sst2", "architecture": "tfidf_logistic"}
    if list(model_dir.glob("*.onnx")):
        return {"format": "onnx", "task": "sst2", "architecture": "onnx"}
    if (model_dir / "micro_config.json").exists():
        return {"format": "micro", "task": "sst2", "architecture": "micro_transformer"}
    if (model_dir / "config.json").exists():
        return {"format": "transformers", "task": "sst2", "architecture": "hf"}
    raise FileNotFoundError(
        f"Could not infer model format in {model_dir}. Add a model_spec.json."
    )


def load_glue_split(
    task: str = "sst2",
    split: str = "validation",
    *,
    cache_dir: str | Path | None = None,
) -> tuple[list[str], list[int]]:
    """Load a GLUE split and drop unlabeled rows (label == -1)."""
    if task not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task!r}. Known: {sorted(TASK_REGISTRY)}")
    from datasets import load_dataset

    meta = TASK_REGISTRY[task]
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        dataset = load_dataset("nyu-mll/glue", meta["glue_config"], split=split, **kwargs)
    except Exception:
        dataset = load_dataset("glue", meta["glue_config"], split=split, **kwargs)

    text_field = meta["text_field"]
    label_field = meta["label_field"]
    texts: list[str] = []
    labels: list[int] = []
    for row in dataset:
        label = int(row[label_field])
        if label < 0:
            continue
        texts.append(str(row[text_field]))
        labels.append(label)
    if not texts:
        raise RuntimeError(f"No labeled examples in {task}/{split}")
    return texts, labels


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="binary", pos_label=1)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0)
        ),
        "matthews_correlation": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "n_examples": int(y_true_arr.size),
        "n_correct": int((y_true_arr == y_pred_arr).sum()),
    }


def measure_latency_ms(
    predict_fn,
    texts: Sequence[str],
    *,
    warmup: int = 8,
    max_timed: int = 128,
) -> float:
    """Median wall-clock milliseconds per example for sequential inference."""
    if not texts:
        return 0.0
    n_warmup = min(warmup, len(texts))
    for i in range(n_warmup):
        predict_fn([texts[i]])
    timed = texts[: min(max_timed, len(texts))]
    started = time.perf_counter()
    for text in timed:
        predict_fn([text])
    elapsed = time.perf_counter() - started
    return round(1000.0 * elapsed / max(len(timed), 1), 4)


def write_model_card(
    model_dir: str | Path,
    *,
    title: str,
    metrics: dict[str, Any] | None = None,
    extra_markdown: str = "",
) -> Path:
    """Write a Hugging Face-style model card next to the weights."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}
    lines = [
        f"# {title}",
        "",
        "This artifact is part of **micro-sota**: the lowest-parameter model that",
        "still exceeds 90% accuracy on GLUE SST-2, continuously verified in CI.",
        "",
        "## Reported metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "## Intended use",
        "",
        "Binary sentiment classification of English movie-review sentences",
        "(Stanford Sentiment Treebank, GLUE SST-2 formulation). Not intended",
        "for general-domain sentiment, toxicity, or multilingual text.",
        "",
        "## Evaluation",
        "",
        "Accuracy is measured on the official GLUE SST-2 **validation** split",
        "(872 labeled examples) with a fixed seed. The unlabeled GLUE test",
        "split is not used; labels for that split are not public.",
        "",
        extra_markdown.strip(),
        "",
        f"Generated {utc_now_iso()}.",
        "",
    ]
    path = model_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class Predictor:
    """Minimal inference protocol used by ``evaluate.py`` and ``predict.py``."""

    format: str = "unknown"
    name: str = "unknown"

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        logits = self.predict_logits(texts)
        return np.argmax(np.asarray(logits), axis=-1).astype(np.int64)

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        logits = np.asarray(self.predict_logits(texts), dtype=np.float64)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def parameter_count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return None


def load_predictor(model_dir: str | Path, *, device: str = "cpu") -> Predictor:
    """Dispatch loader based on ``model_spec.json`` (or inferred format)."""
    model_dir = Path(model_dir)
    spec = read_model_spec(model_dir)
    fmt = spec.get("format", "transformers")
    if fmt in {"sklearn", "linear", "sklearn_pipeline"}:
        from linear_model import SklearnPredictor

        return SklearnPredictor(model_dir, spec)
    if fmt in {"micro", "micro_transformer"}:
        from micro_model import MicroPredictor

        return MicroPredictor(model_dir, spec, device=device)
    if fmt in {"onnx", "onnxruntime"}:
        return _OnnxPredictor(model_dir, spec)
    if fmt in {"transformers", "hf", "pytorch"}:
        return _TransformersPredictor(model_dir, spec, device=device)
    if fmt in {"int4", "int4-weight"}:
        return _Int4Predictor(model_dir, spec, device=device)
    raise ValueError(f"Unsupported model format {fmt!r} in {model_dir}")


class _TransformersPredictor(Predictor):
    format = "transformers"

    def __init__(self, model_dir: Path, spec: dict[str, Any], *, device: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = spec.get("model_name") or model_dir.name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_length = int(spec.get("max_length", 128))
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(device)
        if device == "cpu":
            self.model = self.model.float()
        self.model.eval()
        self._spec = spec

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            return logits.detach().cpu().numpy()

    def parameter_count(self) -> int:
        return count_parameters_torch(self.model)


class _OnnxPredictor(Predictor):
    format = "onnx"

    def __init__(self, model_dir: Path, spec: dict[str, Any]) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = spec.get("files", {}).get("weights")
        if onnx_path:
            weights = model_dir / onnx_path
        else:
            candidates = list(model_dir.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No ONNX weights in {model_dir}")
            weights = candidates[0]
        tokenizer_dir = model_dir
        if (model_dir / "tokenizer").is_dir():
            tokenizer_dir = model_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_length = int(spec.get("max_length", 128))
        self.session = ort.InferenceSession(
            str(weights),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.name = spec.get("model_name") or model_dir.name
        self._n_params = int(spec.get("parameter_count") or 0)
        self._spec = spec

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feeds = {}
        for name in self.input_names:
            if name in encoded:
                feeds[name] = encoded[name]
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(encoded["input_ids"])
        outputs = self.session.run(None, feeds)
        return np.asarray(outputs[0])

    def parameter_count(self) -> int:
        if self._n_params:
            return self._n_params
        # Fall back to on-disk weight file size as a last resort is wrong;
        # require spec to carry the fp32 parameter count when possible.
        return int(self._spec.get("parameter_count", 0))


class _Int4Predictor(Predictor):
    """Weight-only 4-bit packed tensors, dequantized to fp32 at load time."""

    format = "int4-weight"

    def __init__(self, model_dir: Path, spec: dict[str, Any], *, device: str) -> None:
        from micro_model import MicroTransformer, load_micro_config

        pack_path = model_dir / spec.get("files", {}).get("weights", "weights_int4.npz")
        payload = np.load(pack_path, allow_pickle=False)
        config = load_micro_config(model_dir)
        import torch

        self.device = device
        self.model = MicroTransformer(**config)
        state = {}
        for key in payload.files:
            if key.endswith(".__scale"):
                continue
            if key.endswith(".__packed"):
                name = key[: -len(".__packed")]
                packed = payload[key]
                scale = payload[f"{name}.__scale"]
                state[name] = torch.from_numpy(_dequant_int4(packed, scale))
            else:
                state[key] = torch.from_numpy(payload[key])
        self.model.load_state_dict(state, strict=True)
        self.model.to(device)
        self.model.eval()
        from transformers import AutoTokenizer

        tok_dir = model_dir / "tokenizer" if (model_dir / "tokenizer").is_dir() else model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.max_length = int(spec.get("max_length", config.get("max_length", 128)))
        self.name = spec.get("model_name") or model_dir.name

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
        with torch.no_grad():
            logits = self.model(**encoded)
            return logits.detach().cpu().numpy()

    def parameter_count(self) -> int:
        return count_parameters_torch(self.model)


def _dequant_int4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Unpack uint8 nibbles stored as signed int4 in [-8, 7] and rescale."""
    packed = packed.astype(np.uint8)
    hi = (packed >> 4).astype(np.int16)
    lo = (packed & 0x0F).astype(np.int16)
    # stored as unsigned 0..15 representing -8..7
    hi = hi - 8
    lo = lo - 8
    flat = np.empty(packed.size * 2, dtype=np.float32)
    flat[0::2] = lo.reshape(-1)
    flat[1::2] = hi.reshape(-1)
    n = int(scale.shape[0]) if scale.ndim else 1
    # scale is per-row (out_features,) for linear weights
    if scale.ndim == 0:
        return (flat * float(scale)).astype(np.float32)
    width = flat.size // n
    return (flat.reshape(n, width) * scale.reshape(n, 1)).astype(np.float32)


def pack_int4_tensor(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group-wise (per-output-row) absmax int4 pack. Pads to even length."""
    weight = np.asarray(weight, dtype=np.float32)
    original_shape = weight.shape
    if weight.ndim == 1:
        rows = weight.reshape(1, -1)
    else:
        rows = weight.reshape(weight.shape[0], -1)
    absmax = np.maximum(np.max(np.abs(rows), axis=1), 1e-8)
    scale = (absmax / 7.0).astype(np.float32)
    q = np.clip(np.round(rows / scale[:, None]), -8, 7).astype(np.int16)
    q_u = (q + 8).astype(np.uint8)
    n_cols = q_u.shape[1]
    if n_cols % 2 == 1:
        q_u = np.pad(q_u, ((0, 0), (0, 1)), constant_values=8)
    lo = q_u[:, 0::2]
    hi = q_u[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    packed = packed.reshape(-1)
    return packed, scale.astype(np.float32)
