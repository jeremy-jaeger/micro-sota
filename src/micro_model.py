"""From-scratch micro-Transformer for binary sequence classification."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import Predictor, count_parameters_torch, dump_json, write_model_spec


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with GELU FFN."""

    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.ln2(x))


class MicroTransformer(nn.Module):
    """Tiny encoder: factorized embeddings, N layers, mean-pool classifier."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_size: int = 256,
        max_length: int = 128,
        dropout: float = 0.1,
        num_labels: int = 2,
        embedding_size: int | None = None,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        embedding_size = int(embedding_size or hidden_size)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_size = int(ffn_size)
        self.max_length = int(max_length)
        self.dropout_p = float(dropout)
        self.num_labels = int(num_labels)
        self.embedding_size = embedding_size
        self.pad_token_id = int(pad_token_id)

        self.token_emb = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_token_id)
        self.pos_emb = nn.Embedding(self.max_length, embedding_size)
        self.emb_proj: nn.Module = (
            nn.Identity() if embedding_size == hidden_size else nn.Linear(embedding_size, hidden_size)
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, ffn_size, dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._reset()

    def _reset(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def config_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_size": self.ffn_size,
            "max_length": self.max_length,
            "dropout": self.dropout_p,
            "num_labels": self.num_labels,
            "embedding_size": self.embedding_size,
            "pad_token_id": self.pad_token_id,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        seqlen = input_ids.size(1)
        if seqlen > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_length]
            seqlen = self.max_length
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_proj(x))
        key_padding_mask = None if attention_mask is None else attention_mask.eq(0)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.final_ln(x)
        if attention_mask is None:
            pooled = x[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)
        if labels is None:
            return logits
        return F.cross_entropy(logits, labels), logits


def load_micro_config(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "micro_config.json"
    if not path.exists():
        raise FileNotFoundError(f"micro_config.json missing in {model_dir}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_micro_model(
    model: MicroTransformer,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    spec_extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.config_dict()
    dump_json(output_dir / "micro_config.json", config)
    torch.save(model.state_dict(), output_dir / "pytorch_model.pt")
    try:
        from safetensors.torch import save_file

        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_file(cpu_state, str(output_dir / "model.safetensors"))
    except Exception:
        pass
    tokenizer.save_pretrained(output_dir / "tokenizer")
    tokenizer.save_pretrained(output_dir)
    spec = {
        "format": "micro",
        "architecture": "micro_transformer",
        "framework": "pytorch",
        "files": {"weights": "pytorch_model.pt", "config": "micro_config.json"},
        "parameter_count": count_parameters_torch(model),
        "max_length": config["max_length"],
        "labels": {"0": "negative", "1": "positive"},
    }
    if spec_extra:
        spec.update(spec_extra)
    write_model_spec(output_dir, spec)
    return output_dir


def load_micro_model(model_dir: str | Path, *, device: str = "cpu") -> MicroTransformer:
    model_dir = Path(model_dir)
    config = load_micro_config(model_dir)
    model = MicroTransformer(**config)
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device=device)
        model.load_state_dict(state)
    else:
        state = torch.load(model_dir / "pytorch_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def train_wordpiece_tokenizer(
    texts: Sequence[str],
    *,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    model_max_length: int = 128,
):
    """Train a WordPiece tokenizer on task text and wrap it as a fast tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordPiece(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    special = [pad_token, unk_token, cls_token, sep_token]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special,
    )
    tokenizer.train_from_iterator(list(texts), trainer=trainer)
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    cls_id = tokenizer.token_to_id(cls_token)
    sep_id = tokenizer.token_to_id(sep_token)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B {sep_token}",
        special_tokens=[(cls_token, cls_id), (sep_token, sep_id)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token=pad_token,
        cls_token=cls_token,
        sep_token=sep_token,
        model_max_length=model_max_length,
    )


class MicroPredictor(Predictor):
    format = "micro"

    def __init__(self, model_dir: Path, spec: dict[str, Any], *, device: str) -> None:
        from transformers import AutoTokenizer

        self.device = device
        self.model = load_micro_model(model_dir, device=device)
        tok_dir = model_dir / "tokenizer" if (model_dir / "tokenizer").is_dir() else model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.max_length = int(spec.get("max_length", self.model.max_length))
        self.name = spec.get("model_name") or model_dir.name

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(logits, tuple):
                logits = logits[-1]
            return logits.detach().cpu().numpy()

    def parameter_count(self) -> int:
        return count_parameters_torch(self.model)
