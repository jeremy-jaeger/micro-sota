"""Unit tests that do not require the GLUE download or trained champion weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from linear_model import (  # noqa: E402
    build_linear_pipeline,
    count_linear_parameters,
    save_sklearn_model,
)
from model_utils import (  # noqa: E402
    EvalMetrics,
    classification_metrics,
    pack_int4_tensor,
    set_seed,
    utc_now_iso,
)


def test_set_seed_is_deterministic() -> None:
    set_seed(42)
    a = np.random.rand(8)
    set_seed(42)
    b = np.random.rand(8)
    assert np.allclose(a, b)


def test_classification_metrics_perfect() -> None:
    y = [0, 1, 0, 1, 1]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["n_correct"] == 5
    assert m["f1"] == pytest.approx(1.0)


def test_linear_pipeline_fits_toy_data(tmp_path: Path) -> None:
    set_seed(42)
    texts = [
        "a wonderful delightful brilliant movie",
        "excellent superb fantastic acting",
        "terrible awful horrible waste",
        "dreadful boring disappointing mess",
        "loved every beautiful moment",
        "hated this stupid film",
    ] * 8
    labels = [1, 1, 0, 0, 1, 0] * 8
    cfg = {
        "use_word": True,
        "use_char": False,
        "word_tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 200},
        "classifier": "logreg",
        "C": 4.0,
        "max_iter": 200,
    }
    pipe = build_linear_pipeline(cfg, seed=42)
    pipe.fit(texts, labels)
    preds = pipe.predict(texts)
    acc = float((np.asarray(preds) == np.asarray(labels)).mean())
    assert acc >= 0.9
    n_params = count_linear_parameters(pipe)
    assert n_params > 0
    save_sklearn_model(pipe, tmp_path, spec_extra={"task": "sst2", "model_name": "toy"})
    spec = json.loads((tmp_path / "model_spec.json").read_text(encoding="utf-8"))
    assert spec["format"] == "sklearn"
    assert (tmp_path / "model.joblib").exists()


def test_int4_pack_roundtrip() -> None:
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(4, 8)).astype(np.float32)
    packed, scale = pack_int4_tensor(weight)
    assert packed.dtype == np.uint8
    assert scale.shape == (4,)


def test_micro_transformer_under_one_million_params() -> None:
    from micro_model import MicroTransformer
    from model_utils import count_parameters_torch

    model = MicroTransformer(
        vocab_size=4096,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_size=128,
        max_length=96,
        embedding_size=64,
    )
    n = count_parameters_torch(model)
    assert n < 1_000_000
    import torch

    ids = torch.randint(0, 4096, (2, 16))
    mask = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=mask)
    assert logits.shape == (2, 2)


def test_eval_metrics_json_roundtrip() -> None:
    m = EvalMetrics(
        task="sst2",
        split="validation",
        accuracy=0.91,
        f1=0.90,
        precision=0.90,
        recall=0.90,
        matthews_correlation=0.82,
        n_examples=872,
        n_correct=794,
        parameter_count=1000,
        parameter_count_millions=0.001,
        disk_size_bytes=1024,
        disk_size_mb=0.001,
        latency_ms_per_example=0.5,
        device="cpu",
        seed=42,
        model_name="toy",
        model_format="sklearn",
        model_dir="models/champion",
        meets_threshold=True,
        threshold=0.9,
        verified_at=utc_now_iso(),
    )
    payload = m.to_dict()
    assert payload["accuracy"] == 0.91
    assert payload["meets_threshold"] is True
