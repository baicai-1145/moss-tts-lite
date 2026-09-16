"""MOSS-TTS-Local-Transformer-v1.5 (n_vq=12, 48 kHz v2 codec) training model.

Second-generation MOSS-TTS architecture: a global Qwen3 backbone plus a
1-layer NanoGPT2 "local transformer". For every frame the local window is
[global_hidden, audio_embed_0 .. audio_embed_{n_vq-2}] (teacher forcing with
the lower audio channels); text logits come from slot 0 and audio channel c
logits from slot c. Parameter names mirror the official checkpoint keys
(transformer.*, local_transformer.h.0.*, audio_embeddings.*, audio_lm_heads.*,
text_lm_head, local_text_lm_head) and heads are tied to their embeddings
exactly like MossTTSLocalModel.tie_weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (MossBackbone, MossTTSConfig, MossTTSTrainOutput,
                 TrainableMossTTS, rope_tables)
from .st_loader import read_safetensors


@dataclass
class LocalConfig:
    """NanoGPT2 local-transformer hyper-parameters."""

    n_layer: int = 1
    hidden_size: int = 2560
    n_head: int = 32
    n_inner: int = 0  # 0 -> 4 * hidden_size (GPT-2 default)
    activation: str = "gelu_new"  # or "silu" (official v1.5 uses silu)
    layer_norm_eps: float = 1e-5
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_head

    @property
    def inner_size(self) -> int:
        return self.n_inner if self.n_inner > 0 else 4 * self.hidden_size


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    # transformers ACT2FN["gelu_new"]
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


def _interleaved_rope_tables(n: int, head_dim: int, base: float,
                             device: torch.device, dtype: torch.dtype):
    """GPT2-style interleaved RoPE (matches MossTTSNanoGPT2RotaryEmbedding)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                       dtype=torch.float32) / head_dim))
    pos = torch.arange(n, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)                                  # [n, d/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1)                 # [n, d]
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    # [1, 1, n, d] so it broadcasts over [N, H, S, D]
    return (cos[None, None].to(dtype), sin[None, None].to(dtype))


def _apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor,
                            sin: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    rot = torch.stack((-odd, even), dim=-1).reshape_as(x)
    return x * cos + rot * sin


class NanoGPT2Attention(nn.Module):
    """MossTTSNanoGPT2Attention: fused c_attn, interleaved RoPE, causal MHA."""

    def __init__(self, cfg: LocalConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.c_attn = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.c_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.rope_base = cfg.rope_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, S, _ = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        shape = (N, S, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        cos, sin = _interleaved_rope_tables(S, self.head_dim, self.rope_base,
                                            x.device, q.dtype)
        q = _apply_interleaved_rope(q, cos, sin)
        k = _apply_interleaved_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(N, S, -1))


class NanoGPT2MLP(nn.Module):
    def __init__(self, cfg: LocalConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.hidden_size, cfg.inner_size)
        self.fc_out = nn.Linear(cfg.inner_size, cfg.hidden_size)
        self._act = (F.silu if cfg.activation == "silu" else _gelu_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self._act(self.fc_in(x)))


class NanoGPT2Block(nn.Module):
    """Pre-LN GPT2 block (ln_1/attn/ln_2/mlp), names match the checkpoint."""

    def __init__(self, cfg: LocalConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = NanoGPT2Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = NanoGPT2MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class NanoGPT2Local(nn.Module):
    """The local_transformer.* subtree: h.{i} blocks + ln_f (no wte/wpe)."""

    def __init__(self, cfg: LocalConfig):
        super().__init__()
        self.h = nn.ModuleList(NanoGPT2Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class TrainableMossTTSLocal(nn.Module):
    """MOSS-TTS-Local-Transformer-v1.5 with the official SFT loss."""

    def __init__(self, config: MossTTSConfig, local_config: LocalConfig,
                 codebook_sizes: Sequence[int],
                 audio_pad_token_id: int = 1024,
                 audio_assistant_slot_token_id: int = -1,
                 audio_end_token_id: int = -1,
                 use_binary_local_text_head: bool = True):
        super().__init__()
        self.config = config
        self.local_config = local_config
        self.codebook_sizes = list(codebook_sizes)
        if len(self.codebook_sizes) != config.n_vq:
            raise ValueError("codebook_sizes length != n_vq")
        self.audio_pad_token_id = int(audio_pad_token_id)
        self.audio_assistant_slot_token_id = int(audio_assistant_slot_token_id)
        self.audio_end_token_id = int(audio_end_token_id)
        self.use_binary_local_text_head = bool(use_binary_local_text_head)
        self.gradient_checkpointing = False

        self.transformer = MossBackbone(config)
        self.local_transformer = NanoGPT2Local(local_config)
        self.audio_embeddings = nn.ModuleList(
            nn.Embedding(n, config.hidden_size) for n in self.codebook_sizes)
        self.text_lm_head = nn.Linear(config.hidden_size, config.text_vocab,
                                      bias=False)
        self.audio_lm_heads = nn.ModuleList(
            nn.Linear(config.hidden_size, n, bias=False)
            for n in self.codebook_sizes)
        self.local_text_lm_head = (
            nn.Linear(config.hidden_size, 2, bias=False)
            if self.use_binary_local_text_head else None)
        self._tie_weights()

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_dir: str, dtype: torch.dtype | None = None,
                        device: str | torch.device | None = None,
                        ) -> "TrainableMossTTSLocal":
        raw = json.loads((Path(model_dir) / "config.json").read_text("utf-8"))
        q3 = raw.get("qwen3_config", raw)
        g2 = raw.get("gpt2_config", {})
        cfg = MossTTSConfig(
            n_layers=int(q3["num_hidden_layers"]),
            hidden_size=int(q3["hidden_size"]),
            n_heads=int(q3["num_attention_heads"]),
            n_kv_heads=int(q3.get("num_key_value_heads", q3["num_attention_heads"])),
            head_dim=int(q3.get("head_dim", 0)) or
                int(q3["hidden_size"]) // int(q3["num_attention_heads"]),
            intermediate_size=int(q3["intermediate_size"]),
            text_vocab=int(q3["vocab_size"]),
            audio_vocab=int(raw.get("audio_vocab_size", 1024)),
            n_vq=int(raw["n_vq"]),
            rope_theta=float(q3.get("rope_theta", 1e6)),
            rms_eps=float(q3.get("rms_norm_eps", 1e-6)),
        )
        lcfg = LocalConfig(
            n_layer=int(g2.get("n_layer", 1)),
            hidden_size=int(g2.get("n_embd", cfg.hidden_size)),
            n_head=int(g2.get("n_head", cfg.n_heads)),
            n_inner=int(g2.get("n_inner", 0)),
            activation=str(g2.get("activation_function", "gelu_new")),
            layer_norm_eps=float(g2.get("layer_norm_epsilon", 1e-5)),
            rope_base=float(g2.get("rope_base", 10000.0)),
        )
        sizes = [int(n) for n in raw.get("audio_codebook_sizes",
                                         [cfg.audio_vocab] * cfg.n_vq)]
        model = cls(cfg, lcfg, sizes,
                    audio_pad_token_id=int(raw.get("audio_pad_token_id", 1024)),
                    audio_assistant_slot_token_id=int(
                        raw.get("audio_assistant_slot_token_id", -1)),
                    audio_end_token_id=int(raw.get("audio_end_token_id", -1)),
                    use_binary_local_text_head=bool(
                        raw.get("use_binary_local_text_head", True)))
        weights = read_safetensors(str(model_dir), dtype=dtype)
        if dtype is not None:
            model.to(dtype)
        # Tied weights may be absent from the checkpoint; resolve before load.
        model.load_state_dict(weights, strict=False)
        model._fix_tied_and_verify(weights)
        if device is not None:
            model.to(torch.device(device))
        return model

    def _fix_tied_and_verify(self, weights: dict) -> None:
        state = self.state_dict()
        tied_keys = {"text_lm_head.weight":
                     "transformer.embed_tokens.weight"}
        for i in range(self.config.n_vq):
            tied_keys[f"audio_lm_heads.{i}.weight"] = f"audio_embeddings.{i}.weight"
        missing = [k for k in state if k not in weights]
        for k in missing:
            src = tied_keys.get(k)
            if src is not None and src in weights:
                continue  # tied pair present under the other name
            if k == "local_text_lm_head.weight":
                # initialize from the tied text head rows (official behavior)
                slot, end = self.audio_assistant_slot_token_id, self.audio_end_token_id
                if slot >= 0 and end >= 0:
                    with torch.no_grad():
                        self.local_text_lm_head.weight.copy_(
                            self.text_lm_head.weight.index_select(
                                0, torch.tensor([slot, end],
                                                device=self.text_lm_head.weight.device)))
                    continue
            raise KeyError(f"weight {k!r} missing from checkpoint")
        self._tie_weights()

    def _tie_weights(self) -> None:
        self.text_lm_head.weight = self.transformer.embed_tokens.weight
        for embedding, head in zip(self.audio_embeddings, self.audio_lm_heads):
            head.weight = embedding.weight

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    # -- forward ------------------------------------------------------------

    def _compute_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.transformer.embed_tokens(input_ids[..., 0])
        for i, embedding in enumerate(self.audio_embeddings):
            channel = input_ids[..., i + 1]
            valid = channel.ne(self.audio_pad_token_id)
            safe = channel.masked_fill(~valid, 0)
            embeds = embeds + embedding(safe) * valid.unsqueeze(-1)
        return embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        channelwise_loss_weight: Sequence[float] | None = None,
    ) -> MossTTSTrainOutput:
        cfg = self.config
        if input_ids.dim() != 3 or input_ids.shape[-1] != cfg.n_vq + 1:
            raise ValueError(
                f"input_ids must be (B, T, {cfg.n_vq + 1}), got {tuple(input_ids.shape)}")
        bsz, seqlen, _ = input_ids.shape
        # RoPE tables in the embedding dtype: fp32 tables would upcast q/k
        # past the (bf16) value tensor and break SDPA (same as the v1 path).
        cos, sin = rope_tables(
            seqlen, cfg.head_dim, cfg.rope_theta, input_ids.device,
            self.transformer.embed_tokens.weight.dtype)
        attn_mask = TrainableMossTTS._build_attn_bias(
            attention_mask, seqlen, input_ids.device)

        inputs_embeds = self._compute_input_embeddings(input_ids)
        h = self.transformer(inputs_embeds,
                             {"cos": cos, "sin": sin, "mask": attn_mask},
                             gradient_checkpointing=self.gradient_checkpointing)

        text_logits = audio_logits = loss = channel_losses = None
        if labels is not None:
            loss, channel_losses = self._local_loss(
                h, labels, channelwise_loss_weight)
        return MossTTSTrainOutput(text_logits, audio_logits, loss,
                                  channel_losses)

    def _local_loss(
        self,
        global_hidden: torch.Tensor,           # [B, T, H]
        labels: torch.Tensor,                  # [B, T, n_vq+1]
        channelwise_loss_weight: Sequence[float] | None,
    ):
        """Port of finetuning/sft.py compute_supervised_loss_from_hidden."""
        cfg = self.config
        bsz, seqlen, hidden = global_hidden.shape
        n_vq = cfg.n_vq
        if labels.shape[-1] != n_vq + 1:
            raise ValueError(f"labels must have {n_vq + 1} channels")
        weights = list(channelwise_loss_weight) if channelwise_loss_weight \
            else [1.0] * (n_vq + 1)
        if len(weights) != n_vq + 1:
            raise ValueError(f"channelwise weights length {len(weights)} "
                             f"!= {n_vq + 1}")

        flat_hidden = global_hidden.reshape(bsz * seqlen, hidden)
        flat_labels = labels.reshape(bsz * seqlen, n_vq + 1)
        local_dtype = self.local_transformer.ln_f.weight.dtype
        prefix = flat_hidden.to(dtype=local_dtype)
        local_inputs = torch.zeros(bsz * seqlen, n_vq, hidden,
                                   dtype=local_dtype,
                                   device=flat_hidden.device)
        local_inputs[:, 0, :] = prefix

        audio_targets = flat_labels[:, 1:]
        for c in range(n_vq - 1):
            teacher = audio_targets[:, c]
            embedding = self.audio_embeddings[c]
            valid = (teacher >= 0) & (teacher < embedding.num_embeddings)
            safe = teacher.masked_fill(~valid, 0)
            emb = embedding(safe).to(dtype=local_dtype)
            local_inputs[:, c + 1, :] = emb * valid.unsqueeze(-1)

        local_hidden = self.local_transformer(local_inputs)   # [B*T, n_vq, H]

        total = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
        total_weight = 0.0
        text_targets = flat_labels[:, 0]
        per_channel = torch.zeros(n_vq + 1, device=flat_hidden.device,
                                  dtype=torch.float32)

        if self.use_binary_local_text_head and self.local_text_lm_head is not None \
                and self.audio_assistant_slot_token_id >= 0:
            logits = self.local_text_lm_head(local_hidden[:, 0, :])
            binary = torch.full_like(text_targets, -100)
            binary = binary.masked_fill(
                text_targets.eq(self.audio_assistant_slot_token_id), 0)
            binary = binary.masked_fill(
                text_targets.eq(self.audio_end_token_id), 1)
            if (binary != -100).any():
                ce = F.cross_entropy(logits.float(), binary,
                                     ignore_index=-100)
                total = total + float(weights[0]) * ce
                per_channel[0] = ce.detach()
                total_weight += float(weights[0])
        elif (text_targets != -100).any():
            logits = self.text_lm_head(local_hidden[:, 0, :])
            ce = F.cross_entropy(logits.float(), text_targets,
                                 ignore_index=-100)
            total = total + float(weights[0]) * ce
            per_channel[0] = ce.detach()
            total_weight += float(weights[0])

        for c in range(n_vq):
            targets = audio_targets[:, c]
            if not (targets != -100).any():
                continue
            logits = self.audio_lm_heads[c](local_hidden[:, c, :])
            ce = F.cross_entropy(logits.float(), targets, ignore_index=-100)
            total = total + float(weights[c + 1]) * ce
            per_channel[c + 1] = ce.detach()
            total_weight += float(weights[c + 1])

        if total_weight <= 0:
            raise RuntimeError("All labels are ignored; check dataset packing.")
        return total / total_weight, per_channel
