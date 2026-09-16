"""SFT / LoRA / QLoRA training CLI for moss-tts-lite.

Usage:
    python -m moss_tts_lite.train train --model-dir ... --train-jsonl ... --mode qlora ...
    python -m moss_tts_lite.train merge --adapter-dir ... --model-dir ... --output-dir ...

(`train` is the default subcommand: `python -m moss_tts_lite.train --mode lora ...`
also works.)

Single GPU (or CPU for lora/full smoke tests). peft / bitsandbytes are lazy
imports; install with `pip install '.[train]'`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .bpe import QwenBPE
from .data import MossTTSTrainDataset, load_jsonl
from .model import N_VQ
from .nn import TrainableMossTTS
from .nn_local import TrainableMossTTSLocal

__all__ = ["main", "cmd_train", "cmd_merge", "write_safetensors"]

DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
ASSET_FILES = (
    "vocab.json", "merges.txt", "tokenizer.json", "added_tokens.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "config.json", "generation_config.json", "processor_config.json",
)
MAX_SHARD_BYTES = 4 * 2**30


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m moss_tts_lite.train",
        description="Single-GPU SFT / LoRA / QLoRA training for MOSS-TTS-v1.5.")
    p.add_argument("--model-dir", required=True,
                   help="Base model dir (MOSS-TTS-v1.5 safetensors + tokenizer files).")
    p.add_argument("--train-jsonl", required=True,
                   help="JSONL produced by moss_tts_lite.prepare_data (audio_codes ...).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=("full", "lora", "qlora"), default="qlora")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-targets", type=str, default=DEFAULT_LORA_TARGETS,
                   help="Comma-separated LoRA target module names.")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Default 1e-4 for lora/qlora, 1e-5 for full.")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler", choices=("cosine", "linear", "constant"),
                   default="cosine")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Optimizer steps; overrides --num-epochs when set.")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true",
                   help="bf16 params/autograd on CUDA (fp32 fallback elsewhere).")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--channelwise-loss-weight", type=str, default="1,32",
                   help="n_vq+1 values, or 2 values (text, total-audio).")
    p.add_argument("--max-audio-frames", type=int, default=3000,
                   help="Drop records with more target audio frames (~25 s at 24 kHz).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=0,
                   help="0: only save at the end of training.")
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', ...")
    p.add_argument("--merge-and-export", action="store_true",
                   help="After training, merge LoRA and export a lite-CLI-ready "
                        "model dir (output-dir/merged).")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batch-tokens", type=int, default=0,
                   help="Token-budget batching: group similar-length records so "
                        "padded max_len*bs stays <= budget (long clips get "
                        "smaller batches). 0 = fixed --per-device-batch-size.")
    p.add_argument("--fused-optimizer", action="store_true",
                   help="Fused AdamW optimizer step (CUDA only).")
    p.add_argument("--torch-compile", action="store_true",
                   help="torch.compile the wrapped model (experimental with "
                        "NF4/peft; try with --max-batch-tokens for stable "
                        "shapes).")
    return p


def build_merge_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m moss_tts_lite.train merge",
        description="Merge a LoRA adapter into the base model and export a "
                    "lite-CLI-ready model dir.")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--bf16", action="store_true",
                   help="Save merged weights in bf16 (default is fp32).")
    p.add_argument("--qlora-exact-base", action="store_true", default=None,
                   help="Quantize the base to NF4 before merging (replicates the "
                        "training-time forward exactly; needs bitsandbytes+CUDA). "
                        "Auto-detected from train_args.json when possible.")
    p.add_argument("--no-qlora-exact-base", dest="qlora_exact_base", action="store_false")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moss_tts_lite.train",
        description="Train (SFT/LoRA/QLoRA) or merge adapters for MOSS-TTS-v1.5.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", parents=[build_train_parser()], add_help=False)
    sub.add_parser("merge", parents=[build_merge_parser()], add_help=False)
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "merge"):
        argv = ["train"] + argv  # default subcommand: train (incl. --help)
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        return cmd_merge(args)
    return cmd_train(args)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_tokenizer(model_dir: str) -> QwenBPE:
    d = Path(model_dir)
    tok_json = d / "tokenizer.json"
    added = str(tok_json) if tok_json.exists() else str(d / "added_tokens.json")
    tok = QwenBPE(str(d / "vocab.json"), str(d / "merges.txt"), added)
    # v1 (ModelScope) and v2 (HF) assign different ids to audio_start/end;
    # prefer the model's config.json, fall back to the v1 defaults.
    cfg = {}
    cfg_path = d / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    for token, cfg_key, default, is_slot in (
            ("<|im_start|>", "im_start_token_id", 151644, False),
            ("<|im_end|>", "im_end_token_id", 151645, False),
            ("<|audio_start|>", "audio_start_token_id", 151652, False),
            ("<|audio_end|>", "audio_end_token_id", 151653, False),
            ("<|audio_user_slot|>", "audio_user_slot_token_id", 151654, True),
            ("<|audio_assistant_gen_slot|>", "audio_assistant_gen_slot_token_id",
             151656, True),
            ("<|audio_assistant_delay_slot|>", None, 151662, True)):
        want = int(cfg.get(cfg_key, default)) if cfg_key else default
        got = tok.encode(token)
        if got == [want]:
            continue
        if is_slot and tok.id_to_added_token(want) is not None:
            # v2 (local-transformer) tokenizers have no dedicated slot strings;
            # slots reuse the vision/video-pad tokens and are injected by id.
            continue
        raise ValueError(f"tokenizer in {model_dir} maps {token!r} to {got}, "
                         f"expected [{want}]")
    return tok


def parse_channelwise_loss_weight(spec: Optional[str], n_heads: int) -> list[float]:
    values = [float(item.strip()) for item in (spec or "").split(",") if item.strip()]
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_w, total_audio_w = values
        resolved = [text_w] + [total_audio_w / (n_heads - 1)] * (n_heads - 1)
    else:
        raise ValueError(
            f"--channelwise-loss-weight expects {n_heads} values or 2 values "
            f"(text,total_audio), got {len(values)}.")
    if sum(resolved) <= 0:
        raise ValueError("--channelwise-loss-weight must sum to a positive value.")
    return resolved


def resolve_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_model_class(model_dir: str) -> tuple[bool, type]:
    """Pick TrainableMossTTSLocal vs TrainableMossTTS from config.json."""
    cfg_path = Path(model_dir) / "config.json"
    is_local_arch = False
    if cfg_path.exists():
        try:
            is_local_arch = json.loads(
                cfg_path.read_text("utf-8")).get("model_type") == "moss_tts_local"
        except json.JSONDecodeError:
            pass
    model_cls = TrainableMossTTSLocal if is_local_arch else TrainableMossTTS
    return is_local_arch, model_cls


def copy_assets(model_dir: str, output_dir: Path) -> list[str]:
    copied = []
    for name in ASSET_FILES:
        src = Path(model_dir) / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    return copied


# --- safetensors writing (mirror of st_loader reader) ----------------------

_DT_REV: dict[torch.dtype, tuple[str, Any]] = {
    torch.float64: ("F64", "float64"),
    torch.float32: ("F32", "float32"),
    torch.float16: ("F16", "float16"),
    torch.bfloat16: ("BF16", None),
    torch.int64: ("I64", "int64"),
    torch.int32: ("I32", "int32"),
    torch.int16: ("I16", "int16"),
    torch.int8: ("I8", "int8"),
    torch.uint8: ("U8", "uint8"),
    torch.bool: ("BOOL", "bool"),
}


def write_safetensors(state_dict: dict[str, torch.Tensor], out_path: str | Path,
                      max_shard_bytes: int = MAX_SHARD_BYTES) -> list[str]:
    """Write (optionally sharded) safetensors readable by st_loader/HF.

    Single shard -> `model.safetensors`; multi shard -> `model-0000X-of-0000N.safetensors`
    + `model.safetensors.index.json`.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    entries = [(name, t.detach().cpu().contiguous()) for name, t in state_dict.items()]

    shards: list[list[tuple[str, torch.Tensor]]] = [[]]
    shard_bytes = 0
    for name, t in entries:
        nbytes = t.numel() * t.element_size()
        if shards[-1] and shard_bytes + nbytes > max_shard_bytes:
            shards.append([])
            shard_bytes = 0
        shards[-1].append((name, t))
        shard_bytes += nbytes

    weight_map: dict[str, str] = {}
    n = len(shards)
    written: list[str] = []
    for idx, shard in enumerate(shards, start=1):
        if n == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{idx:05d}-of-{n:05d}.safetensors"
        _write_single_safetensors(shard, out_path / filename)
        for name, _t in shard:
            weight_map[name] = filename
        written.append(filename)

    if n > 1:
        total = sum(t.numel() * t.element_size() for _n, t in entries)
        with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f,
                      indent=2)
    return written


def _write_single_safetensors(entries: list[tuple[str, torch.Tensor]],
                              path: Path) -> None:
    import struct
    import numpy as np

    header: dict[str, Any] = {}
    offset = 0
    for name, t in entries:
        if t.dtype not in _DT_REV:
            raise TypeError(f"cannot serialize dtype {t.dtype} for {name!r}")
        dt_str, np_dtype = _DT_REV[t.dtype]
        nbytes = t.numel() * t.element_size()
        header[name] = {"dtype": dt_str, "shape": list(t.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _name, t in entries:
            _dt_str, np_dtype = _DT_REV[t.dtype]
            if np_dtype is None:  # bf16
                raw = t.view(torch.uint16).numpy().tobytes()
            else:
                raw = t.numpy().astype(np_dtype, copy=False).tobytes()
            f.write(raw)


def export_model_dir(model: torch.nn.Module, model_dir: str, output_dir: Path,
                     dtype: Optional[torch.dtype] = None) -> list[str]:
    """Save a full state_dict with original key names + tokenizer/config assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {k: (v.to(dtype) if dtype is not None else v)
             for k, v in model.state_dict().items()}
    written = write_safetensors(state, output_dir)
    copy_assets(model_dir, output_dir)
    # trust_remote_code architectures also need their modeling .py files,
    # which ASSET_FILES (v1 asset names) does not cover.
    for src in Path(model_dir).iterdir():
        if src.suffix in (".py", ".jinja") and src.is_file():
            shutil.copy2(src, output_dir / src.name)
    return written


# --- LoRA / QLoRA -----------------------------------------------------------

def quantize_model_4bit(model: torch.nn.Module,
                         target_device: torch.device | None = None) -> None:
    """Swap language_model linears for bitsandbytes NF4 Linear4bit (in place).

    Embeddings and lm_heads stay in their loaded dtype (standard QLoRA layout,
    like transformers' default llm_int8_skip_modules=["lm_head"]). The quantized
    weights must live on CUDA. When target_device is given, CPU-resident
    linears are moved there one at a time before quantizing, so a full bf16
    copy of the model never sits on the GPU at once (peak VRAM saver for
    single-24GB QLoRA). Remaining non-linear modules stay where they are;
    callers should model.to(target_device) afterwards.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit, Params4bit
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "QLoRA requires bitsandbytes (CUDA, Linux): pip install bitsandbytes"
        ) from exc

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, torch.nn.Linear):
                continue
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            # Both generations keep their qwen3 trunk under one container:
            # language_model.* (v1 delay) / transformer.* (local-transformer).
            # "local_transformer.*" must NOT match: it stays high-precision.
            if not (qualified.startswith("language_model")
                    or qualified.startswith("transformer.")):
                continue
            weight = child.weight.data
            if target_device is not None and weight.device.type == "cpu":
                weight = weight.to(target_device)
            quant_weight, quant_state = bnb.functional.quantize_4bit(
                weight, quant_type="nf4", compress_statistics=True)
            new_layer = Linear4bit(
                child.in_features, child.out_features, bias=False,
                compute_dtype=weight.dtype, quant_type="nf4",
                compress_statistics=True, device=weight.device)
            new_layer.weight = Params4bit(quant_weight, requires_grad=False,
                                          quant_state=quant_state,
                                          bnb_quantized=True)
            setattr(parent, child_name, new_layer)


def dequantize_model_4bit(model: torch.nn.Module) -> None:
    """Replace bnb Linear4bit layers with plain nn.Linear in compute dtype.

    The reconstructed weights are exactly what the training-time forward
    used (NF4 storage dequantized), so a subsequent plain LoRA merge is
    numerically exact and avoids peft's packed-uint8 merge pitfalls.
    Works both on a bare model and inside a PeftModel (the LoRA wrapper's
    base_layer reference is swapped along with the module tree).
    """
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit

    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, Linear4bit):
                continue
            weight = child.weight.data
            deq = bnb.functional.dequantize_4bit(weight, child.weight.quant_state)
            deq = deq.to(child.compute_dtype)
            new_layer = torch.nn.Linear(child.in_features, child.out_features,
                                        bias=False, device=deq.device,
                                        dtype=deq.dtype)
            new_layer.weight.data.copy_(deq)
            setattr(parent, child_name, new_layer)
            replaced += 1
    if replaced:
        torch.cuda.empty_cache()
        print(f"dequantized {replaced} Linear4bit layers to "
              f"{deq.dtype} for merge")


def wrap_lora(model: torch.nn.Module, rank: int, alpha: int, targets: list[str],
              dropout: float = 0.0) -> torch.nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LoRA requires peft: pip install 'moss-tts-lite[train]' (or pip install peft)"
        ) from exc
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                        lora_dropout=dropout, bias="none")
    return get_peft_model(model, config)


def _print_trainable(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{_ts()}] trainable params: {trainable:,} / {total:,} "
          f"({100.0 * trainable / max(total, 1):.2f}%)")


# --- scheduler ---------------------------------------------------------------

def build_lr_scheduler(optimizer: torch.optim.Optimizer, name: str,
                       warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if name == "linear":
            return 1.0 - progress
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TokenBudgetBatchSampler:
    """Length-grouped, token-budget batch sampler for variable-length records.

    Keeps padding waste low (similar lengths share a batch) and caps the
    worst-case padded batch (max_len * bs <= max_tokens), so bigger
    --per-device-batch-size no longer risks OOM on long clips. Re-shuffles
    every epoch; batch order is shuffled too.
    """

    def __init__(self, lengths: list[int], batch_size: int, max_tokens: int,
                 seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.seed = seed
        self.epoch = 0

    def _compose(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        # Stable sort keeps the random order within equal lengths.
        idx.sort(key=lambda i: self.lengths[i])
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in idx:
            n = self.lengths[i]
            new_max = max(cur_max, n)
            if cur and (len(cur) + 1 > self.batch_size
                        or new_max * (len(cur) + 1) > self.max_tokens):
                batches.append(cur)
                cur, cur_max = [i], n
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._compose()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._compose())


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    # Fragmentation-resistant allocation for variable-length batches; must be
    # set before the first CUDA allocation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    if args.mode == "qlora" and device.type != "cuda":
        raise SystemExit("--mode qlora needs CUDA (bitsandbytes 4-bit); "
                         "use --mode lora for CPU testing.")

    dtype = torch.float32
    if args.bf16:
        if device.type == "cuda":
            dtype = torch.bfloat16
        else:
            print(f"[{_ts()}] WARNING: --bf16 ignored on {device.type}; using fp32")
    elif args.mode == "qlora":
        print(f"[{_ts()}] WARNING: qlora without --bf16 keeps fp32 master weights "
              f"(NF4 storage still applies); pass --bf16 for the recommended "
              f"NF4 + bf16-compute setup")

    lr = args.learning_rate
    if lr is None:
        lr = 1e-5 if args.mode == "full" else 1e-4

    records = load_jsonl(args.train_jsonl)
    if not records:
        raise SystemExit(f"No records in {args.train_jsonl}")
    if args.max_audio_frames and args.max_audio_frames > 0:
        kept = [r for r in records
                if len(r["audio_codes"]) <= args.max_audio_frames]
        dropped = len(records) - len(kept)
        if dropped:
            print(f"[{_ts()}] dropped {dropped}/{len(records)} records exceeding "
                  f"max-audio-frames={args.max_audio_frames}")
        records = kept
    if not records:
        raise SystemExit("No records left after --max-audio-frames filtering.")

    tokenizer = load_tokenizer(args.model_dir)
    # v2 (local-transformer): slots reuse Qwen vision/video-pad ids; pass
    # them so the dataset builds one-slot-per-frame sequences.
    local_slot_ids = None
    if _detect_model_class(args.model_dir)[0]:
        raw_cfg = json.loads((Path(args.model_dir) / "config.json")
                              .read_text(encoding="utf-8"))
        local_slot_ids = (int(raw_cfg["audio_user_slot_token_id"]),
                          int(raw_cfg["audio_assistant_slot_token_id"]))
    dataset = MossTTSTrainDataset(records, tokenizer, slot_ids=local_slot_ids)
    batch_sampler = None
    if args.max_batch_tokens and args.max_batch_tokens > 0:
        lengths = [int(dataset.pack_record(r)["input_ids"].shape[0])
                   for r in records]
        batch_sampler = TokenBudgetBatchSampler(
            lengths, args.per_device_batch_size, args.max_batch_tokens,
            seed=args.seed)
        srt = sorted(lengths)
        print(f"[{_ts()}] token-budget batching: budget={args.max_batch_tokens} "
              f"max_bs={args.per_device_batch_size} "
              f"batches/epoch={len(batch_sampler)} "
              f"len p50={srt[len(srt) // 2]} p95={srt[min(len(srt) - 1, int(len(srt) * 0.95))]} "
              f"max={srt[-1]}")
    loader_kwargs = dict(
        collate_fn=dataset.collate_fn, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0)
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        loader = DataLoader(dataset, batch_size=args.per_device_batch_size,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed),
                            **loader_kwargs)

    # QLoRA: load on CPU first, then quantize linears onto the GPU one at a
    # time — avoids ever holding a full bf16 copy + NF4 copy on a 24GB card.
    load_device = (torch.device("cpu")
                   if (args.mode == "qlora" and device.type == "cuda") else device)
    print(f"[{_ts()}] loading base model from {args.model_dir} "
          f"(device={load_device}, dtype={dtype})")
    is_local_arch, model_cls = _detect_model_class(args.model_dir)
    print(f"[{_ts()}] architecture: "
          f"{'local-transformer (n_vq=12, 48kHz v2)' if is_local_arch else 'delay (v1)'}")
    model = model_cls.from_pretrained(args.model_dir, dtype=dtype,
                                      device=load_device)
    base_n_vq = model.config.n_vq

    if args.mode == "qlora":
        quantize_model_4bit(model, target_device=device)
        model.to(device)
        print(f"[{_ts()}] quantized language_model linears to NF4")

    if args.mode in ("lora", "qlora"):
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, targets,
                          args.lora_dropout)
        _print_trainable(model)

    if args.gradient_checkpointing:
        # PeftModel delegates unknown attributes to the wrapped base model.
        model.gradient_checkpointing_enable()
        print(f"[{_ts()}] gradient checkpointing enabled")

    if args.torch_compile:
        model = torch.compile(model, dynamic=True)
        print(f"[{_ts()}] torch.compile enabled (dynamic=True)")

    n_heads = 1 + base_n_vq
    channelwise = parse_channelwise_loss_weight(args.channelwise_loss_weight, n_heads)
    print(f"[{_ts()}] channelwise_loss_weight={channelwise}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=args.weight_decay,
                      betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      fused=args.fused_optimizer and device.type == "cuda")

    micro_batches_per_epoch = (
        len(batch_sampler) if batch_sampler is not None
        else math.ceil(len(dataset) / args.per_device_batch_size))
    update_steps_per_epoch = math.ceil(micro_batches_per_epoch
                                       / args.gradient_accumulation_steps)
    max_train_steps = args.max_steps or args.num_epochs * update_steps_per_epoch
    warmup_steps = math.ceil(max_train_steps * args.warmup_ratio)
    scheduler = build_lr_scheduler(optimizer, args.lr_scheduler, warmup_steps,
                                   max_train_steps)
    print(f"[{_ts()}] scheduler={args.lr_scheduler} warmup_steps={warmup_steps} "
          f"micro_batches/epoch={micro_batches_per_epoch} "
          f"optimizer_steps/epoch={update_steps_per_epoch} "
          f"max_train_steps={max_train_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_args = {k: v for k, v in vars(args).items() if k != "command"}
    train_args.update({"learning_rate": lr,
                       "resolved_channelwise_loss_weight": channelwise,
                       "max_train_steps": max_train_steps,
                       "warmup_steps": warmup_steps,
                       "n_vq": n_heads - 1,
                       "torch_dtype": str(dtype).replace("torch.", ""),
                       "moss_tts_lite_train_version": 1})
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(train_args, f, indent=2, ensure_ascii=False)

    model.train()
    global_step = 0
    losses: list[float] = []
    start_time = time.perf_counter()
    last_log = start_time
    last_logged_step = 0
    done = False

    def _save_adapter(step_dir: Path) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        # torch.compile wraps the model in OptimizedModule; peft's
        # save_pretrained lives on the original module.
        target = getattr(model, "_orig_mod", model)
        target.save_pretrained(str(step_dir))
        with open(step_dir / "train_args.json", "w", encoding="utf-8") as f:
            json.dump(train_args, f, indent=2, ensure_ascii=False)

    def _optimizer_step(micro_loss: float, epoch: int) -> bool:
        """Run one optimizer update. Returns True when max_train_steps is hit."""
        nonlocal global_step, last_log, last_logged_step
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        losses.append(micro_loss * args.gradient_accumulation_steps)

        now = time.perf_counter()
        if global_step % args.logging_steps == 0:
            steps_per_sec = max(global_step - last_logged_step, 1) / (now - last_log)
            eta = (max_train_steps - global_step) / steps_per_sec
            mem = (torch.cuda.max_memory_allocated() / 2**30
                   if device.type == "cuda" else 0.0)
            print(f"[{_ts()}] epoch={epoch} step={global_step}/{max_train_steps} "
                  f"loss={losses[-1]:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"steps_per_sec={steps_per_sec:.3f} "
                  f"eta={_fmt_duration(eta)} peak_mem={mem:.2f}GiB",
                  flush=True)
            last_log = now
            last_logged_step = global_step

        if args.save_steps and global_step % args.save_steps == 0:
            if args.mode in ("lora", "qlora"):
                _save_adapter(output_dir / f"checkpoint-{global_step}")
            else:
                export_model_dir(model, args.model_dir,
                                 output_dir / f"checkpoint-{global_step}",
                                 dtype=dtype)

        return global_step >= max_train_steps

    for epoch in range(args.num_epochs):
        if done:
            break
        for it, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        channelwise_loss_weight=channelwise)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()

            if (it + 1) % args.gradient_accumulation_steps == 0:
                if _optimizer_step(float(loss.detach()), epoch):
                    done = True
                    break
        else:
            # Epoch exhausted without hitting max steps: flush a trailing
            # partial accumulation window (batch counts can vary per epoch
            # under token-budget batching).
            if not done and (it + 1) % args.gradient_accumulation_steps != 0:
                if _optimizer_step(float(loss.detach()), epoch):
                    done = True

    # -- final save -----------------------------------------------------------
    merged_dir = None
    if args.mode in ("lora", "qlora"):
        _save_adapter(output_dir)
        print(f"[{_ts()}] saved adapter to {output_dir}")
        if args.merge_and_export:
            merged_dir = output_dir / "merged"
            if args.mode == "qlora":
                dequantize_model_4bit(model)
            merged = model.merge_and_unload()
            export_model_dir(merged, args.model_dir, merged_dir, dtype=dtype)
            del merged
            print(f"[{_ts()}] merged export saved to {merged_dir}")
    else:
        written = export_model_dir(model, args.model_dir, output_dir, dtype=dtype)
        print(f"[{_ts()}] saved full model: {', '.join(written)}")

    elapsed = time.perf_counter() - start_time
    print(f"[{_ts()}] finished: steps={global_step} elapsed={_fmt_duration(elapsed)} "
          f"final_loss={losses[-1] if losses else float('nan'):.4f}")
    return {"steps": global_step, "losses": losses, "output_dir": str(output_dir),
            "merged_dir": str(merged_dir) if merged_dir else None}


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def cmd_merge(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("merge requires peft: pip install peft") from exc

    device = resolve_device(args.device)
    adapter_dir = Path(args.adapter_dir)
    train_args = {}
    train_args_path = adapter_dir / "train_args.json"
    if train_args_path.exists():
        train_args = json.loads(train_args_path.read_text(encoding="utf-8"))

    qlora_exact = args.qlora_exact_base
    if qlora_exact is None:
        qlora_exact = bool(train_args.get("mode") == "qlora")

    compute_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") \
        else torch.float32
    qlora_on_cuda = qlora_exact and device.type == "cuda"
    load_device = torch.device("cpu") if qlora_on_cuda else device
    is_local_arch, model_cls = _detect_model_class(args.model_dir)
    print(f"[{_ts()}] loading base model from {args.model_dir} "
          f"(device={load_device}, dtype={compute_dtype}, qlora_exact_base={qlora_exact}, "
          f"arch={'local-transformer' if is_local_arch else 'delay'})")
    model = model_cls.from_pretrained(args.model_dir, dtype=compute_dtype,
                                      device=load_device)
    if qlora_exact:
        if not qlora_on_cuda:
            print(f"[{_ts()}] WARNING: qlora-exact-base needs CUDA+bitsandbytes; "
                  "falling back to plain bf16 base merge")
        else:
            quantize_model_4bit(model, target_device=device)
            dequantize_model_4bit(model)
            model.to(device)
            print(f"[{_ts()}] base == NF4-dequantized training-time forward; bf16 merge")

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print(f"[{_ts()}] adapter loaded from {adapter_dir}; merging")
    merged = model.merge_and_unload()

    out_dtype = torch.bfloat16 if args.bf16 else torch.float32
    output_dir = Path(args.output_dir)
    written = export_model_dir(merged, args.model_dir, output_dir, dtype=out_dtype)
    with open(output_dir / "merge_args.json", "w", encoding="utf-8") as f:
        json.dump({"adapter_dir": str(adapter_dir), "model_dir": args.model_dir,
                   "qlora_exact_base": qlora_exact,
                   "torch_dtype": str(out_dtype).replace("torch.", "")},
                  f, indent=2, ensure_ascii=False)
    print(f"[{_ts()}] merged model saved to {output_dir}: {', '.join(written)}")
    return {"output_dir": str(output_dir), "written": written}


if __name__ == "__main__":
    main()
