"""TF-IDF / NBSVM linear baselines for SST-2 (and other binary GLUE tasks)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from model_utils import Predictor, dump_json, write_model_spec


class NaiveBayesRatio(BaseEstimator, TransformerMixin):
    """Wang & Manning (2012) Naive Bayes log-count ratio transform.

    Fits class-conditional interpolated counts on a (usually TF-IDF) feature
    matrix ``X`` and multiplies each example by ``r = log(p / q)``.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        x_pos = X[y == 1]
        x_neg = X[y == 0]
        p = np.asarray(x_pos.sum(axis=0)).ravel() + self.alpha
        q = np.asarray(x_neg.sum(axis=0)).ravel() + self.alpha
        p /= p.sum()
        q /= q.sum()
        self.r_ = np.log(p / q)
        return self

    def transform(self, X):
        return X.multiply(self.r_) if hasattr(X, "multiply") else X * self.r_


def build_linear_pipeline(cfg: dict[str, Any], seed: int = 42) -> Pipeline:
    """Construct a sklearn pipeline from a linear-* config mapping."""
    word_cfg = cfg.get("word_tfidf", {})
    char_cfg = cfg.get("char_tfidf", {})
    transformers = []
    if cfg.get("use_word", True):
        transformers.append(
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(word_cfg.get("ngram_range", [1, 3])),
                    min_df=int(word_cfg.get("min_df", 2)),
                    max_df=float(word_cfg.get("max_df", 0.95)),
                    max_features=word_cfg.get("max_features", 50000),
                    sublinear_tf=bool(word_cfg.get("sublinear_tf", True)),
                    lowercase=True,
                    strip_accents="unicode",
                ),
            )
        )
    if cfg.get("use_char", True):
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer=char_cfg.get("analyzer", "char_wb"),
                    ngram_range=tuple(char_cfg.get("ngram_range", [3, 6])),
                    min_df=int(char_cfg.get("min_df", 2)),
                    max_df=float(char_cfg.get("max_df", 0.95)),
                    max_features=char_cfg.get("max_features", 40000),
                    sublinear_tf=bool(char_cfg.get("sublinear_tf", True)),
                    lowercase=True,
                ),
            )
        )
    if not transformers:
        raise ValueError("linear model config must enable word and/or char TF-IDF")

    steps: list[tuple[str, Any]] = [("features", FeatureUnion(transformers, n_jobs=None))]
    if cfg.get("variant", "tfidf_lr") == "nbsvm":
        steps.append(("nbsvm", NaiveBayesRatio(alpha=float(cfg.get("nb_alpha", 1.0)))))

    classifier = cfg.get("classifier", "logreg")
    C = float(cfg.get("C", 2.0))
    if classifier == "linearsvc":
        clf: Any = LinearSVC(
            C=C,
            max_iter=int(cfg.get("max_iter", 5000)),
            dual=False,
            random_state=seed,
        )
    else:
        clf = LogisticRegression(
            C=C,
            max_iter=int(cfg.get("max_iter", 2000)),
            solver=cfg.get("solver", "liblinear"),
            random_state=seed,
        )
    steps.append(("clf", clf))
    return Pipeline(steps)


def count_linear_parameters(pipeline: Pipeline) -> int:
    """Numeric parameters used at inference: classifier weights + intercept + idf."""
    total = 0
    clf = pipeline.named_steps["clf"]
    coef = np.asarray(clf.coef_)
    total += int(coef.size)
    if hasattr(clf, "intercept_") and clf.intercept_ is not None:
        total += int(np.asarray(clf.intercept_).size)
    features = pipeline.named_steps["features"]
    for _, vectorizer in features.transformer_list:
        idf = getattr(vectorizer, "idf_", None)
        if idf is not None:
            total += int(np.asarray(idf).size)
    if "nbsvm" in pipeline.named_steps:
        total += int(np.asarray(pipeline.named_steps["nbsvm"].r_).size)
    return total


def vocabulary_size(pipeline: Pipeline) -> int:
    features = pipeline.named_steps["features"]
    size = 0
    for _, vectorizer in features.transformer_list:
        vocab = getattr(vectorizer, "vocabulary_", None)
        if vocab is not None:
            size += len(vocab)
    return size


def save_sklearn_model(
    pipeline: Pipeline,
    output_dir: str | Path,
    *,
    spec_extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / "model.joblib"
    joblib.dump(pipeline, weights, compress=("xz", 4))
    n_params = count_linear_parameters(pipeline)
    spec = {
        "format": "sklearn",
        "architecture": "tfidf_logistic",
        "framework": "scikit-learn",
        "files": {"weights": "model.joblib"},
        "parameter_count": n_params,
        "vocabulary_size": vocabulary_size(pipeline),
        "labels": {"0": "negative", "1": "positive"},
    }
    if spec_extra:
        spec.update(spec_extra)
    write_model_spec(output_dir, spec)
    dump_json(
        output_dir / "linear_stats.json",
        {
            "parameter_count": n_params,
            "vocabulary_size": vocabulary_size(pipeline),
            "classifier": type(pipeline.named_steps["clf"]).__name__,
        },
    )
    return output_dir


class SklearnPredictor(Predictor):
    format = "sklearn"

    def __init__(self, model_dir: Path, spec: dict[str, Any]) -> None:
        weights_name = spec.get("files", {}).get("weights", "model.joblib")
        path = model_dir / weights_name
        if not path.exists():
            fallback = model_dir / "linear_model.joblib"
            if fallback.exists():
                path = fallback
            else:
                raise FileNotFoundError(f"sklearn weights not found in {model_dir}")
        self.pipeline: Pipeline = joblib.load(path)
        self.name = spec.get("model_name") or model_dir.name
        self._n_params = int(spec.get("parameter_count") or count_linear_parameters(self.pipeline))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.pipeline.predict(list(texts)), dtype=np.int64)

    def parameter_count(self) -> int:
        return self._n_params
