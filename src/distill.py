"""Task-level knowledge distillation from a public teacher into a tiny student."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from micro_model import MicroTransformer, save_micro_model, train_wordpiece_tokenizer  # noqa: E402
from model_utils import (  # noqa: E402
    TASK_REGISTRY,
    classification_metrics,
    count_parameters_torch,
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
    help="Distill a teacher classifier into a micro-Transformer or HF student.",
)
LOGGER = logging.getLogger("micro_sota.distill")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_config(config: Path) -> Path:
    if config.exists():
        return config.resolve()
    for candidate in ((Path.cwd() / config), (repo_root() / config)):
        if candidate.exists():
            return candidate.resolve()
    raise typer.BadParameter(f"Config not found: {config}")


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="YAML distillation config."),
    output_dir: Path = typer.Option(Path("models/runs/distilled"), "--output-dir", "-o"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device"),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Distill class logits from a teacher into a smaller student.

    Intermediate-layer matching is intentionally omitted so the student may use
    a different tokenizer, width, and depth. The loss is

        L = (1 - alpha) * CE(student, y) + alpha * T^2 * KL(student/T, teacher/T)
    """
    _setup_logging(verbose)
    cfg = load_yaml(_resolve_config(config))
    seed = int(cfg.get("seed", seed))
    set_seed(seed)
    task_name = cfg.get("task", "sst2")
    resolved_device = get_device(device)
    output_dir = output_dir if output_dir.is_absolute() else (repo_root() / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_name = cfg.get("teacher", {}).get(
        "pretrained", "distilbert-base-uncased-finetuned-sst-2-english"
    )
    student_cfg = cfg.get("student", {})
    train_cfg = cfg.get("train", {})
    temperature = float(cfg.get("temperature", 2.0))
    alpha = float(cfg.get("alpha", 0.7))
    max_length = int(train_cfg.get("max_length", student_cfg.get("max_length", 128)))

    LOGGER.info("Loading teacher %s", teacher_name)
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name)
    teacher.to(resolved_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    texts, labels = load_glue_split(task_name, "train")
    if max_train_samples is not None and max_train_samples < len(texts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), size=max_train_samples, replace=False)
        idx.sort()
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]

    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=float(train_cfg.get("internal_val_ratio", 0.05)),
        random_state=seed, stratify=labels,
    )

    student_type = student_cfg.get("type", "micro")
    if student_type == "hf":
        student, student_tok, fmt = _init_hf_student(student_cfg, task_name, resolved_device)
    else:
        student, student_tok, fmt = _init_micro_student(
            student_cfg, x_train, task_name, resolved_device, max_length
        )
    LOGGER.info("Student parameters: %s", f"{count_parameters_torch(student):,}")

    class _Pair(Dataset):
        def __init__(self, xs: list[str], ys: list[int]) -> None:
            self.xs = xs
            self.ys = ys

        def __len__(self) -> int:
            return len(self.xs)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
            return {"text": self.xs[idx], "labels": torch.tensor(self.ys[idx], dtype=torch.long)}

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        xs = [b["text"] for b in batch]
        ys = torch.stack([b["labels"] for b in batch])
        t_enc = teacher_tok(
            xs, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        s_enc = student_tok(
            xs, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        return {"teacher": t_enc, "student": s_enc, "labels": ys}

    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 4))
    lr = float(train_cfg.get("lr", 5e-4))
    loader = DataLoader(_Pair(x_train, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(_Pair(x_val, y_val), batch_size=batch_size, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01)))

    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    for epoch in range(epochs):
        student.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            t_inputs = {k: v.to(resolved_device) for k, v in batch["teacher"].items()}
            s_inputs = {k: v.to(resolved_device) for k, v in batch["student"].items()}
            y = batch["labels"].to(resolved_device)
            with torch.no_grad():
                t_logits = teacher(**t_inputs).logits
            s_out = student(
                input_ids=s_inputs["input_ids"],
                attention_mask=s_inputs.get("attention_mask"),
            )
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
            hard = F.cross_entropy(s_logits, y)
            soft = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = (1.0 - alpha) * hard + alpha * soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
        acc = _eval_student(student, val_loader, resolved_device)
        LOGGER.info(
            "epoch %d/%d loss=%.4f holdout_acc=%.4f",
            epoch + 1,
            epochs,
            running / max(len(loader), 1),
            acc,
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    student.to(resolved_device)
    extra = {
        "task": task_name,
        "seed": seed,
        "model_name": student_cfg.get("name", "distilled-student"),
        "teacher": teacher_name,
        "temperature": temperature,
        "alpha": alpha,
        "trained_at": utc_now_iso(),
        "internal_holdout_accuracy": best_acc,
        "parameter_count": count_parameters_torch(student),
    }
    if fmt == "micro":
        save_micro_model(student, student_tok, output_dir, spec_extra=extra)
    else:
        student.save_pretrained(output_dir)
        student_tok.save_pretrained(output_dir)
        write_model_spec(output_dir, {"format": "transformers", "architecture": student_cfg.get("pretrained"), **extra})
    write_model_card(
        output_dir,
        title=f"micro-sota distilled student ({task_name})",
        metrics={"internal_holdout_accuracy": best_acc, "parameters": count_parameters_torch(student)},
        extra_markdown=(
            f"Task-level distillation from `{teacher_name}`.\n\n"
            f"Loss: `(1 - α) CE + α T² KL` with α={alpha}, T={temperature}."
        ),
    )
    typer.echo(f"Saved distilled student to {output_dir}")


def _init_micro_student(student_cfg, texts, task_name, device, max_length):
    tok_name = student_cfg.get("tokenizer")
    if tok_name:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        vocab_size = len(tokenizer)
        pad_id = int(tokenizer.pad_token_id)
    else:
        vocab_size = int(student_cfg.get("vocab_size", 4096))
        tokenizer = train_wordpiece_tokenizer(texts, vocab_size=vocab_size, model_max_length=max_length)
        vocab_size = len(tokenizer)
        pad_id = int(tokenizer.pad_token_id)
    model = MicroTransformer(
        vocab_size=vocab_size,
        hidden_size=int(student_cfg.get("hidden_size", 128)),
        num_layers=int(student_cfg.get("num_layers", 2)),
        num_heads=int(student_cfg.get("num_heads", 4)),
        ffn_size=int(student_cfg.get("ffn_size", 256)),
        max_length=max_length,
        dropout=float(student_cfg.get("dropout", 0.1)),
        num_labels=int(TASK_REGISTRY[task_name]["num_labels"]),
        embedding_size=student_cfg.get("embedding_size"),
        pad_token_id=pad_id,
    ).to(device)
    return model, tokenizer, "micro"


def _init_hf_student(student_cfg, task_name, device):
    pretrained = student_cfg.get("pretrained")
    if not pretrained:
        raise typer.BadParameter("student.pretrained is required when student.type=hf")
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    id2label = {int(k): v for k, v in TASK_REGISTRY[task_name]["id2label"].items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=int(TASK_REGISTRY[task_name]["num_labels"]),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    ).to(device)
    return model, tokenizer, "hf"


def _eval_student(student, loader, device) -> float:
    student.eval()
    preds: list[int] = []
    gold: list[int] = []
    with torch.no_grad():
        for batch in loader:
            s_inputs = {k: v.to(device) for k, v in batch["student"].items()}
            out = student(input_ids=s_inputs["input_ids"], attention_mask=s_inputs.get("attention_mask"))
            logits = out.logits if hasattr(out, "logits") else out
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(batch["labels"].tolist())
    return classification_metrics(gold, preds)["accuracy"]


if __name__ == "__main__":
    app()
