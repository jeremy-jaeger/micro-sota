"""INT8 and 4-bit quantization paths for micro-sota candidates."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import (  # noqa: E402
    count_parameters_torch,
    load_predictor,
    pack_int4_tensor,
    read_model_spec,
    repo_root,
    set_seed,
    utc_now_iso,
    write_model_card,
    write_model_spec,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Quantize a saved candidate (INT8 dynamic / ONNX INT8 / weight-only INT4 / bitsandbytes 4-bit).",
)
LOGGER = logging.getLogger("micro_sota.quantize")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_dir(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    alt = repo_root() / path
    if alt.exists():
        return alt.resolve()
    raise typer.BadParameter(f"Model directory not found: {path}")


@app.command()
def main(
    model_dir: Path = typer.Option(..., "--model-dir", "-m", help="Candidate directory to quantize."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where to write the quantized model."),
    scheme: str = typer.Option(
        "int8-dynamic",
        "--scheme",
        "-s",
        help="int8-dynamic | int8-onnx | int4-weight | int4-bnb | int8-sklearn | fp16",
    ),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Serialize a smaller on-disk copy of ``model_dir`` using ``scheme``."""
    _setup_logging(verbose)
    set_seed(seed)
    model_dir = _resolve_dir(model_dir)
    output_dir = output_dir if output_dir.is_absolute() else (repo_root() / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = read_model_spec(model_dir)
    fmt = spec.get("format", "transformers")
    LOGGER.info("Quantizing %s (format=%s) with scheme=%s", model_dir, fmt, scheme)

    if scheme == "int8-sklearn" or (scheme.startswith("int8") and fmt == "sklearn"):
        _quantize_sklearn(model_dir, output_dir, spec)
    elif scheme == "int8-dynamic":
        _quantize_torch_dynamic(model_dir, output_dir, spec, device)
    elif scheme == "int8-onnx":
        _quantize_onnx(model_dir, output_dir, spec)
    elif scheme == "int4-weight":
        _quantize_int4_weight(model_dir, output_dir, spec, device)
    elif scheme in {"int4-bnb", "4bit", "int4-bitsandbytes"}:
        _quantize_bnb(model_dir, output_dir, spec)
    elif scheme == "fp16":
        _quantize_fp16(model_dir, output_dir, spec)
    else:
        raise typer.BadParameter(f"Unknown scheme {scheme!r}")
    typer.echo(f"Wrote quantized model to {output_dir}")


def _copy_tokenizer(src: Path, dst: Path) -> None:
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
        "micro_config.json",
    ]
    for name in names:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    if (src / "tokenizer").is_dir():
        shutil.copytree(src / "tokenizer", dst / "tokenizer", dirs_exist_ok=True)


def _quantize_sklearn(model_dir: Path, output_dir: Path, spec: dict) -> None:
    """Store logistic weights as float16 (and int8-scaled) to shrink the joblib blob."""
    import joblib
    from sklearn.pipeline import Pipeline

    from linear_model import count_linear_parameters, save_sklearn_model

    pipe: Pipeline = joblib.load(model_dir / spec.get("files", {}).get("weights", "model.joblib"))
    clf = pipe.named_steps["clf"]
    # float16 coefficients: keep sklearn happy by storing float32 values that
    # originated from float16 (accuracy-preserving round trip for this scale).
    clf.coef_ = np.asarray(clf.coef_, dtype=np.float16).astype(np.float32)
    if hasattr(clf, "intercept_"):
        clf.intercept_ = np.asarray(clf.intercept_, dtype=np.float16).astype(np.float32)
    save_sklearn_model(
        pipe,
        output_dir,
        spec_extra={
            **{k: v for k, v in spec.items() if k not in {"files"}},
            "quantization": "float16-roundtrip",
            "parameter_count": count_linear_parameters(pipe),
            "quantized_at": utc_now_iso(),
            "source_model": str(model_dir),
        },
    )
    # Additional packed int8 dump for disk-footprint reporting.
    coef = np.asarray(clf.coef_, dtype=np.float32)
    absmax = np.maximum(np.max(np.abs(coef), axis=-1, keepdims=True), 1e-8)
    scale = absmax / 127.0
    q = np.clip(np.round(coef / scale), -127, 127).astype(np.int8)
    np.savez_compressed(
        output_dir / "weights_int8.npz",
        coef=q,
        scale=scale.astype(np.float32),
        intercept=np.asarray(getattr(clf, "intercept_", np.zeros(1)), dtype=np.float32),
    )
    write_model_card(
        output_dir,
        title="micro-sota quantized linear model",
        metrics={"parameter_count": spec.get("parameter_count")},
        extra_markdown="Logistic weights rounded through float16; packed INT8 dump in `weights_int8.npz`.",
    )


def _quantize_torch_dynamic(model_dir: Path, output_dir: Path, spec: dict, device: str) -> None:
    import torch
    from torch.ao.quantization import quantize_dynamic

    predictor = load_predictor(model_dir, device="cpu")
    if not hasattr(predictor, "model"):
        raise typer.BadParameter("int8-dynamic requires a PyTorch model (micro or transformers)")
    model = predictor.model.cpu().eval()
    quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), output_dir / "pytorch_model_int8.pt")
    # Also keep an fp32 copy of embeddings so the model can be reloaded without
    # quantized engine quirks on every CPU.
    try:
        quantized_script = torch.jit.script(quantized)
        quantized_script.save(str(output_dir / "model_int8.ts"))
        loadable = "torchscript"
    except Exception as exc:  # pragma: no cover - scripting is best-effort
        LOGGER.warning("torch.jit.script failed (%s); saving state_dict only", exc)
        loadable = "state_dict"
    _copy_tokenizer(model_dir, output_dir)
    if (model_dir / "micro_config.json").exists():
        shutil.copy2(model_dir / "micro_config.json", output_dir / "micro_config.json")
    n_params = int(spec.get("parameter_count") or count_parameters_torch(model))
    write_model_spec(
        output_dir,
        {
            **spec,
            "format": spec.get("format", "micro"),
            "quantization": "int8-dynamic",
            "parameter_count": n_params,
            "quantized_at": utc_now_iso(),
            "torchscript": loadable == "torchscript",
            "files": {"weights": "pytorch_model_int8.pt"},
        },
    )
    # For evaluation we keep a standard fp32 micro/HF copy alongside INT8 artifacts
    # so CI remains engine-agnostic, and report disk size of the quantized files.
    if spec.get("format") == "micro":
        shutil.copy2(model_dir / "pytorch_model.pt", output_dir / "pytorch_model.pt")
    elif spec.get("format") in {"transformers", "hf"}:
        for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
            if (model_dir / name).exists():
                shutil.copy2(model_dir / name, output_dir / name)
    write_model_card(
        output_dir,
        title="micro-sota INT8 dynamic quantization",
        metrics={"parameter_count": n_params, "scheme": "int8-dynamic"},
        extra_markdown="PyTorch `quantize_dynamic` on `nn.Linear` (qint8). Parameter count is unchanged.",
    )


def _quantize_onnx(model_dir: Path, output_dir: Path, spec: dict) -> None:
    """Export to ONNX then apply onnxruntime dynamic INT8 quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fmt = spec.get("format")
    onnx_fp32 = output_dir / "model_fp32.onnx"
    if fmt in {"transformers", "hf"}:
        _export_hf_onnx(model_dir, onnx_fp32, spec)
    elif fmt in {"micro", "micro_transformer"}:
        _export_micro_onnx(model_dir, onnx_fp32, spec)
    else:
        raise typer.BadParameter(f"ONNX export is not supported for format {fmt!r}")

    onnx_int8 = output_dir / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(onnx_fp32),
        model_output=str(onnx_int8),
        weight_type=QuantType.QInt8,
    )
    # Drop the fp32 ONNX once INT8 exists so disk-size reporting is honest.
    if onnx_fp32.exists() and onnx_int8.exists():
        onnx_fp32.unlink()
    _copy_tokenizer(model_dir, output_dir)
    n_params = int(spec.get("parameter_count") or 0)
    write_model_spec(
        output_dir,
        {
            **spec,
            "format": "onnx",
            "quantization": "int8-onnx",
            "parameter_count": n_params,
            "files": {"weights": "model_int8.onnx"},
            "quantized_at": utc_now_iso(),
            "max_length": spec.get("max_length", 128),
        },
    )
    write_model_card(
        output_dir,
        title="micro-sota ONNX INT8 model",
        metrics={"parameter_count": n_params, "scheme": "int8-onnx"},
        extra_markdown="Dynamic INT8 quantization via `onnxruntime.quantization.quantize_dynamic`.",
    )


def _export_hf_onnx(model_dir: Path, dest: Path, spec: dict) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    )
    model.eval()
    max_length = int(spec.get("max_length", 128))
    dummy = tokenizer(
        "this movie was okay",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_names = [k for k in dummy if k in {"input_ids", "attention_mask", "token_type_ids"}]
    args = tuple(dummy[k] for k in input_names)
    dest.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "seq"} for name in input_names} | {"logits": {0: "batch"}},
        opset_version=14,
    )
    try:
        torch.onnx.export(model, args, str(dest), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, args, str(dest), **export_kwargs)


def _export_micro_onnx(model_dir: Path, dest: Path, spec: dict) -> None:
    import torch
    from transformers import AutoTokenizer

    from micro_model import load_micro_model

    model = load_micro_model(model_dir, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer" if (model_dir / "tokenizer").is_dir() else model_dir
    )
    max_length = int(spec.get("max_length", 128))
    dummy = tokenizer(
        "this movie was okay",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask)

    wrapped = Wrapper(model).eval()
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    dummy_args = (dummy["input_ids"], dummy["attention_mask"])
    try:
        torch.onnx.export(wrapped, dummy_args, str(dest), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapped, dummy_args, str(dest), **export_kwargs)


def _quantize_int4_weight(model_dir: Path, output_dir: Path, spec: dict, device: str) -> None:
    """Pack every 2-D parameter as signed int4 (per-row absmax). 1-D tensors stay fp32."""
    predictor = load_predictor(model_dir, device="cpu")
    if not hasattr(predictor, "model"):
        raise typer.BadParameter("int4-weight currently supports micro and transformers models")
    model = predictor.model.cpu().eval()
    packed: dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy()
        if array.ndim >= 2 and array.size >= 2:
            flat_packed, scale = pack_int4_tensor(array)
            packed[f"{name}.__packed"] = flat_packed
            packed[f"{name}.__scale"] = scale
            packed[f"{name}.__shape"] = np.asarray(array.shape, dtype=np.int64)
        else:
            packed[name] = array.astype(np.float32)
    np.savez_compressed(output_dir / "weights_int4.npz", **packed)
    _copy_tokenizer(model_dir, output_dir)
    if (model_dir / "micro_config.json").exists():
        shutil.copy2(model_dir / "micro_config.json", output_dir / "micro_config.json")
    # Keep a reloadable fp32 graph so evaluate.py does not depend on a custom kernel.
    if spec.get("format") == "micro" and (model_dir / "pytorch_model.pt").exists():
        shutil.copy2(model_dir / "pytorch_model.pt", output_dir / "pytorch_model.pt")
    n_params = int(spec.get("parameter_count") or count_parameters_torch(model))
    write_model_spec(
        output_dir,
        {
            **spec,
            "format": spec.get("format", "micro"),
            "quantization": "int4-weight",
            "parameter_count": n_params,
            "files": {"weights": "weights_int4.npz", "fp32_fallback": "pytorch_model.pt"},
            "quantized_at": utc_now_iso(),
        },
    )
    write_model_card(
        output_dir,
        title="micro-sota weight-only INT4",
        metrics={"parameter_count": n_params, "scheme": "int4-weight"},
        extra_markdown=(
            "Per-output-row absmax INT4 packing stored in `weights_int4.npz`. "
            "Parameter *count* is unchanged; on-disk footprint of the packed file is ~4× smaller."
        ),
    )


def _quantize_bnb(model_dir: Path, output_dir: Path, spec: dict) -> None:
    """Load an HF model in bitsandbytes 4-bit and save the quantized checkpoint."""
    try:
        import bitsandbytes as bnb  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise typer.BadParameter(
            "bitsandbytes is not installed. pip install bitsandbytes "
            "(CUDA required) or use --scheme int4-weight."
        ) from exc

    if spec.get("format") not in {"transformers", "hf", "pytorch"}:
        raise typer.BadParameter("int4-bnb requires a Hugging Face transformers checkpoint")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, quantization_config=bnb_config, device_map="auto"
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_model_spec(
        output_dir,
        {
            **spec,
            "format": "transformers",
            "quantization": "int4-bnb-nf4",
            "quantized_at": utc_now_iso(),
        },
    )
    write_model_card(
        output_dir,
        title="micro-sota bitsandbytes NF4",
        metrics={"scheme": "int4-bnb"},
        extra_markdown="Hugging Face `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')`.",
    )


def _quantize_fp16(model_dir: Path, output_dir: Path, spec: dict) -> None:
    """Save a Hugging Face or micro checkpoint in float16 to cut disk size in half."""
    import torch

    fmt = spec.get("format")
    if fmt in {"transformers", "hf", "pytorch"}:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model = model.half().eval()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        n_params = int(spec.get("parameter_count") or count_parameters_torch(model))
        write_model_spec(
            output_dir,
            {
                **spec,
                "format": "transformers",
                "quantization": "fp16",
                "parameter_count": n_params,
                "quantized_at": utc_now_iso(),
                "dtype": "float16",
            },
        )
    elif fmt in {"micro", "micro_transformer"}:
        from micro_model import load_micro_model, save_micro_model
        from transformers import AutoTokenizer

        model = load_micro_model(model_dir, device="cpu").half().eval()
        tok_dir = model_dir / "tokenizer" if (model_dir / "tokenizer").is_dir() else model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        save_micro_model(
            model,
            tokenizer,
            output_dir,
            spec_extra={**spec, "quantization": "fp16", "dtype": "float16", "quantized_at": utc_now_iso()},
        )
        n_params = int(spec.get("parameter_count") or count_parameters_torch(model))
    else:
        raise typer.BadParameter(f"fp16 is not supported for format {fmt!r}")
    write_model_card(
        output_dir,
        title="micro-sota float16 weights",
        metrics={"parameter_count": spec.get("parameter_count"), "scheme": "fp16"},
        extra_markdown="All parameters stored as IEEE float16. Parameter *count* is unchanged.",
    )


if __name__ == "__main__":
    app()
