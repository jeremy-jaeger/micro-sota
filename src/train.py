"""Train linear, micro-Transformer, or Hugging Face encoder models on GLUE tasks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

# Allow `python src/train.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear_model import build_linear_pipeline, save_sklearn_model  # noqa: E402
from micro_model import (  # noqa: E402
    MicroTransformer,
    save_micro_model,
    train_wordpiece_tokenizer,
)
from model_utils import (  # noqa: E402
    TASK_REGISTRY,
    classification_metrics,
    count_parameters_torch,
    dump_json,
    get_device,
    load_glue_split,
    load_yaml,
    repo_root,
    set_seed,
    utc_now_iso,
    write_model_card,
    write_model_spec,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Train a tiny-sota candidate on a GLUE binary classification task.",
)
LOGGER = logging.getLogger("tiny_sota.train")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _task_name(cfg: dict[str, Any], override: str | None) -> str:
    task = override or cfg.get("task", "sst2")
    if task not in TASK_REGISTRY:
        raise typer.BadParameter(f"Unknown task {task!r}")
    return task


def _internal_split(
    texts: list[str],
    labels: list[int],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        texts,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config (see configs/)."),
    output_dir: Path = typer.Option(
        Path("models/runs/latest"),
        "--output-dir",
        "-o",
        help="Directory for weights, tokenizer, and model card.",
    ),
    seed: int = typer.Option(42, "--seed", help="Global RNG seed."),
    device: str = typer.Option("auto", "--device", help="cpu | cuda | mps | auto"),
    task: Optional[str] = typer.Option(None, "--task", help="Override config task (sst2, cola)."),
    max_train_samples: Optional[int] = typer.Option(
        None, "--max-train-samples", help="Optional subsample of the train split."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train a model described by a YAML config. Never reads the official val split."""
    _setup_logging(verbose)
    config_path = Path(config)
    if not config_path.exists():
        config_path = (Path.cwd() / config).resolve()
    if not config_path.exists():
        config_path = (repo_root() / config).resolve()
    if not config_path.exists():
        raise typer.BadParameter(f"Config not found: {config}")

    cfg = load_yaml(config_path)
    cfg.setdefault("seed", seed)
    seed = int(cfg.get("seed", seed))
    set_seed(seed)
    task_name = _task_name(cfg, task)
    resolved_device = get_device(device)
    output_dir = output_dir if output_dir.is_absolute() else (repo_root() / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading %s train split (seed=%s)", task_name, seed)
    train_texts, train_labels = load_glue_split(task_name, "train")
    if max_train_samples is not None and max_train_samples < len(train_texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_texts), size=max_train_samples, replace=False)
        idx.sort()
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", cfg.get("type", "linear"))
    LOGGER.info("Model type=%s device=%s n_train=%d", model_type, resolved_device, len(train_texts))

    if model_type in {"linear", "sklearn", "nbsvm"}:
        _train_linear(cfg, model_cfg, train_texts, train_labels, output_dir, seed, task_name, config_path)
    elif model_type in {"micro", "micro_transformer"}:
        _train_micro(
            cfg,
            model_cfg,
            train_texts,
            train_labels,
            output_dir,
            seed,
            task_name,
            resolved_device,
            config_path,
        )
    elif model_type in {"hf", "transformers", "huggingface"}:
        _train_hf(
            cfg,
            model_cfg,
            train_texts,
            train_labels,
            output_dir,
            seed,
            task_name,
            resolved_device,
            config_path,
        )
    else:
        raise typer.BadParameter(f"Unknown model.type {model_type!r}")

    typer.echo(f"Saved candidate to {output_dir}")


def _holdout_metrics(
    predict, texts: list[str], labels: list[int]
) -> dict[str, float]:
    preds = np.asarray(predict(texts), dtype=np.int64)
    return classification_metrics(labels, preds.tolist())


def _train_linear(
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    texts: list[str],
    labels: list[int],
    output_dir: Path,
    seed: int,
    task_name: str,
    config_path: Path,
) -> None:
    val_ratio = float(cfg.get("train", {}).get("internal_val_ratio", 0.05))
    x_train, x_val, y_train, y_val = _internal_split(texts, labels, val_ratio=val_ratio, seed=seed)
    pipeline = build_linear_pipeline(model_cfg, seed=seed)
    LOGGER.info("Fitting linear pipeline on %d examples", len(x_train))
    pipeline.fit(x_train, y_train)
    holdout = _holdout_metrics(pipeline.predict, x_val, y_val)
    LOGGER.info("Internal holdout accuracy=%.4f f1=%.4f", holdout["accuracy"], holdout["f1"])

    # Refit on the full training split for the submitted candidate.
    pipeline.fit(texts, labels)
    save_sklearn_model(
        pipeline,
        output_dir,
        spec_extra={
            "task": task_name,
            "seed": seed,
            "model_name": model_cfg.get("name", "tfidf-nbsvm" if model_cfg.get("variant") == "nbsvm" else "tfidf-lr"),
            "config": str(config_path.relative_to(repo_root())) if _is_relative_to(config_path, repo_root()) else str(config_path),
            "trained_at": utc_now_iso(),
            "internal_holdout": holdout,
        },
    )
    dump_json(output_dir / "holdout_metrics.json", holdout)
    write_model_card(
        output_dir,
        title=f"tiny-sota linear candidate ({task_name})",
        metrics=holdout,
        extra_markdown=(
            "Linear TF-IDF (word + character n-grams) classifier, optionally with "
            "Wang & Manning (2012) Naive Bayes feature scaling. Tokenizer/vocabulary "
            "is stored inside `model.joblib`.\n\n"
            "Official GLUE validation numbers are produced only by `src/evaluate.py`."
        ),
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _train_micro(
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    texts: list[str],
    labels: list[int],
    output_dir: Path,
    seed: int,
    task_name: str,
    device: str,
    config_path: Path,
) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer

    train_cfg = cfg.get("train", {})
    val_ratio = float(train_cfg.get("internal_val_ratio", 0.05))
    x_train, x_val, y_train, y_val = _internal_split(texts, labels, val_ratio=val_ratio, seed=seed)

    tok_name = model_cfg.get("tokenizer")
    vocab_size = int(model_cfg.get("vocab_size", 4096))
    max_length = int(model_cfg.get("max_length", train_cfg.get("max_length", 128)))
    if tok_name:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        vocab_size = len(tokenizer)
        pad_id = int(tokenizer.pad_token_id)
    else:
        LOGGER.info("Training WordPiece tokenizer vocab_size=%d", vocab_size)
        tokenizer = train_wordpiece_tokenizer(
            x_train, vocab_size=vocab_size, model_max_length=max_length
        )
        vocab_size = len(tokenizer)
        pad_id = int(tokenizer.pad_token_id)

    model = MicroTransformer(
        vocab_size=vocab_size,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        ffn_size=int(model_cfg.get("ffn_size", 256)),
        max_length=max_length,
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_labels=int(TASK_REGISTRY[task_name]["num_labels"]),
        embedding_size=model_cfg.get("embedding_size"),
        pad_token_id=pad_id,
    ).to(device)
    LOGGER.info("Micro-Transformer parameters: %s", f"{count_parameters_torch(model):,}")

    class _Txt(Dataset):
        def __init__(self, xs: list[str], ys: list[int]) -> None:
            self.xs = xs
            self.ys = ys

        def __len__(self) -> int:
            return len(self.xs)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            enc = tokenizer(
                self.xs[idx],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.ys[idx], dtype=torch.long),
            }

    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 5))
    lr = float(train_cfg.get("lr", 3e-4))
    loader = DataLoader(_Txt(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_Txt(x_val, y_val), batch_size=batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    total_steps = max(len(loader) * epochs, 1)
    warmup = int(train_cfg.get("warmup_ratio", 0.06) * total_steps)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * float(step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        import math

        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    global_step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            for pg in opt.param_groups:
                pg["lr"] = lr_at(global_step)
            opt.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            global_step += 1
        model.eval()
        preds: list[int] = []
        gold: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds.extend(logits.argmax(-1).cpu().tolist())
                gold.extend(batch["labels"].tolist())
        metrics = classification_metrics(gold, preds)
        LOGGER.info(
            "epoch %d/%d loss=%.4f holdout_acc=%.4f",
            epoch + 1,
            epochs,
            running / max(len(loader), 1),
            metrics["accuracy"],
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    save_micro_model(
        model,
        tokenizer,
        output_dir,
        spec_extra={
            "task": task_name,
            "seed": seed,
            "model_name": model_cfg.get("name", "micro-transformer"),
            "trained_at": utc_now_iso(),
            "internal_holdout_accuracy": best_acc,
        },
    )
    write_model_card(
        output_dir,
        title=f"tiny-sota micro-Transformer ({task_name})",
        metrics={"internal_holdout_accuracy": best_acc, "parameters": count_parameters_torch(model)},
        extra_markdown=(
            "Trained from scratch on the task train split with a WordPiece tokenizer "
            "fitted only on training text (or a configured public tokenizer). "
            "The official GLUE validation split is reserved for `src/evaluate.py`."
        ),
    )


def _train_hf(
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    texts: list[str],
    labels: list[int],
    output_dir: Path,
    seed: int,
    task_name: str,
    device: str,
    config_path: Path,
) -> None:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed as hf_set_seed,
    )

    hf_set_seed(seed)
    pretrained = model_cfg.get("pretrained") or model_cfg.get("name")
    if not pretrained:
        raise typer.BadParameter("hf model config requires model.pretrained")
    num_labels = int(TASK_REGISTRY[task_name]["num_labels"])
    id2label = {int(k): v for k, v in TASK_REGISTRY[task_name]["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    train_cfg = cfg.get("train", {})
    max_length = int(model_cfg.get("max_length", train_cfg.get("max_length", 128)))
    val_ratio = float(train_cfg.get("internal_val_ratio", 0.05))
    x_train, x_val, y_train, y_val = _internal_split(texts, labels, val_ratio=val_ratio, seed=seed)

    def tokenize_rows(xs: list[str], ys: list[int]) -> Dataset:
        enc = tokenizer(xs, truncation=True, max_length=max_length)
        enc["labels"] = ys
        return Dataset.from_dict(enc)

    train_ds = tokenize_rows(x_train, y_train)
    val_ds = tokenize_rows(x_val, y_val)

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=-1)
        return classification_metrics(gold, preds)

    use_fp16 = bool(train_cfg.get("fp16", False)) and device == "cuda"
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer_out"),
        seed=seed,
        data_seed=seed,
        learning_rate=float(train_cfg.get("lr", 2e-5)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 32)),
        num_train_epochs=float(train_cfg.get("epochs", 3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        fp16=use_fp16,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    write_model_spec(
        output_dir,
        {
            "format": "transformers",
            "architecture": pretrained,
            "framework": "pytorch",
            "task": task_name,
            "seed": seed,
            "model_name": model_cfg.get("name", pretrained),
            "parameter_count": count_parameters_torch(trainer.model),
            "max_length": max_length,
            "labels": {str(k): v for k, v in id2label.items()},
            "trained_at": utc_now_iso(),
            "internal_holdout": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        },
    )
    write_model_card(
        output_dir,
        title=f"tiny-sota HF fine-tune: {pretrained}",
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        extra_markdown=(
            f"Fine-tuned `{pretrained}` on the {task_name} training split with Hugging Face "
            "Trainer. Best checkpoint selected on an internal holdout, never on the "
            "official GLUE validation set."
        ),
    )
    # Drop bulky trainer checkpoints from the candidate folder.
    trainer_out = output_dir / "trainer_out"
    if trainer_out.exists():
        import shutil

        shutil.rmtree(trainer_out, ignore_errors=True)


if __name__ == "__main__":
    app()
