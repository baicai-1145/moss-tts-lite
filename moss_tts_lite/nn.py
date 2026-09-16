"""Trainable MOSS-TTS-v1.5 model for SFT / LoRA / QLoRA.

`TrainableMossTTS` mirrors the inference implementation (`moss_tts_lite.model`)
op-for-op -- fp32 RMSNorm statistics, the same RoPE chain, per-head q/k norms --
so that logits match `MossTTSModel` bit-for-bit-ish, while parameter names
match the MOSS-TTS-v1.5 safetensors keys exactly (strict `load_state_dict`).

Training-only module: importing it never pulls in transformers/peft/bnb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import N_VQ, AUDIO_PAD_CODE, ROPE_THETA, RMS_EPS
from .st_loader import read_safetensors

__all__ = ["MossTTSConfig", "TrainableMossTTS", "MossTTSTrainOutput"]


@dataclass
class MossTTSConfig:
    """Architecture hyper-parameters (inferred from the checkpoint by default)."""

    n_layers: int = 36
    hidden_size: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    text_vocab: int = 155648
    audio_vocab: int = 1024          # codebook size; embeddings/heads use +1 (pad)
    n_vq: int = N_VQ
    rope_theta: float = ROPE_THETA
    rms_eps: float = RMS_EPS

    @classmethod
    def infer_from_weights(cls, weights: dict[str, torch.Tensor]) -> "MossTTSConfig":
        layer_ids = sorted({int(k.split(".")[2]) for k in weights
                            if k.startswith("language_model.layers.")})
        if not layer_ids or layer_ids != list(range(len(layer_ids))):
            raise ValueError(f"unexpected layer ids {layer_ids}")
        emb = weights["language_model.embed_tokens.weight"]
        head_dim = int(weights["language_model.layers.0.self_attn.q_norm.weight"].shape[0])
        n_heads = int(weights["language_model.layers.0.self_attn.q_proj.weight"].shape[0]) // head_dim
        n_kv = int(weights["language_model.layers.0.self_attn.k_proj.weight"].shape[0]) // head_dim
        n_vq = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("emb_ext."))             if any(k.startswith("emb_ext.") for k in weights) else 0
        audio_rows = int(weights["emb_ext.0.weight"].shape[0]) if n_vq else AUDIO_PAD_CODE + 1
        return cls(
            n_layers=len(layer_ids),
            hidden_size=int(emb.shape[1]),
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            intermediate_size=int(weights["language_model.layers.0.mlp.gate_proj.weight"].shape[0]),
            text_vocab=int(emb.shape[0]),
            audio_vocab=audio_rows - 1,
            n_vq=n_vq,
        )


class MossRMSNorm(nn.Module):
    """transformers Qwen3RMSNorm, op-for-op (fp32 stats, cast back, then scale)."""

    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_tables(n: int, head_dim: int, theta: float,
                device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin for `n` positions -- the inference fp32 chain, verbatim."""
    inv = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(n, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb2 = torch.cat((freqs, freqs), dim=-1)
    return emb2.cos().to(dtype), emb2.sin().to(dtype)


class MossAttention(nn.Module):
    def __init__(self, cfg: MossTTSConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = MossRMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = MossRMSNorm(cfg.head_dim, cfg.rms_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_bias: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(bsz, seqlen, cfg.n_heads, cfg.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, cfg.n_kv_heads, cfg.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, cfg.n_kv_heads, cfg.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)                      # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = cfg.n_heads // cfg.n_kv_heads
        if rep > 1:
            k = torch.repeat_interleave(k, rep, dim=1)
            v = torch.repeat_interleave(v, rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(bsz, seqlen, cfg.n_heads * cfg.head_dim)
        return self.o_proj(out)


class MossMLP(nn.Module):
    def __init__(self, cfg: MossTTSConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossDecoderLayer(nn.Module):
    def __init__(self, cfg: MossTTSConfig):
        super().__init__()
        self.self_attn = MossAttention(cfg)
        self.mlp = MossMLP(cfg)
        self.input_layernorm = MossRMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.post_attention_layernorm = MossRMSNorm(cfg.hidden_size, cfg.rms_eps)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_bias: torch.Tensor) -> torch.Tensor:
        residual = h
        h = self.self_attn(self.input_layernorm(h), cos, sin, attn_bias)
        h = residual + h
        residual = h
        h = self.mlp(self.post_attention_layernorm(h))
        return residual + h


class MossBackbone(nn.Module):
    """The `language_model.*` subtree of the checkpoint."""

    def __init__(self, cfg: MossTTSConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.text_vocab, cfg.hidden_size)
        self.layers = nn.ModuleList(MossDecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.norm = MossRMSNorm(cfg.hidden_size, cfg.rms_eps)

    def forward(self, inputs_embeds: torch.Tensor, attn_bias: torch.Tensor,
                gradient_checkpointing: bool = False) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            if gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, attn_bias["cos"], attn_bias["sin"], attn_bias["mask"],
                    use_reentrant=False)
            else:
                h = layer(h, attn_bias["cos"], attn_bias["sin"], attn_bias["mask"])
        return self.norm(h)


class MossTTSTrainOutput(NamedTuple):
    text_logits: torch.Tensor          # [B, T, text_vocab]
    audio_logits: torch.Tensor         # [B, T, n_vq, audio_vocab + 1]
    loss: Optional[torch.Tensor]       # scalar, None when labels is None
    channel_losses: Optional[torch.Tensor]  # [n_vq + 1], None when labels is None


class TrainableMossTTS(nn.Module):
    """MOSS-TTS-v1.5 with training heads; parameter names == checkpoint keys."""

    def __init__(self, config: MossTTSConfig | None = None, **overrides):
        super().__init__()
        base = dict(vars(config)) if config is not None else {}
        base.update(overrides)
        self.config = MossTTSConfig(**{k: v for k, v in base.items()
                                       if k in MossTTSConfig.__dataclass_fields__})
        cfg = self.config
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.gradient_checkpointing = False
        self.language_model = MossBackbone(cfg)
        self.emb_ext = nn.ModuleList(
            nn.Embedding(cfg.audio_vocab + 1, cfg.hidden_size) for _ in range(cfg.n_vq))
        self.lm_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_size, cfg.text_vocab, bias=False)]
            + [nn.Linear(cfg.hidden_size, cfg.audio_vocab + 1, bias=False)
               for _ in range(cfg.n_vq)])

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_dir: str, dtype: torch.dtype | None = None,
                        device: str | torch.device | None = None) -> "TrainableMossTTS":
        weights = read_safetensors(str(model_dir), dtype=dtype)
        model = cls(MossTTSConfig.infer_from_weights(weights))
        if dtype is not None:
            model.to(dtype)  # cast params first so load copies without upcasting
        model.load_state_dict(weights, strict=True)
        if device is not None:
            model.to(torch.device(device))
        return model

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    # -- forward ------------------------------------------------------------

    def _compute_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_tokens(input_ids[..., 0])
        for i, embed_layer in enumerate(self.emb_ext):
            inputs_embeds = inputs_embeds + embed_layer(input_ids[..., i + 1])
        return inputs_embeds

    @staticmethod
    def _build_attn_bias(attention_mask: torch.Tensor | None, seqlen: int,
                         device: torch.device) -> torch.Tensor:
        """[B, 1, T, T] bool mask (True = attend), causal + key padding.

        Query rows that fall in the left padding attend to all valid keys so
        SDPA never sees an all-masked row (which would produce NaNs that
        poison real rows through 0 * NaN in the value path). Those rows are
        excluded from the loss by -100 labels.
        """
        if attention_mask is None:
            attention_mask = torch.ones(1, seqlen, dtype=torch.bool, device=device)
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        key_valid = attention_mask[:, None, None, :]                 # [B,1,1,T]
        causal = torch.ones(seqlen, seqlen, dtype=torch.bool,
                            device=device).tril()[None, None]         # [1,1,T,T]
        pad_query = ~attention_mask[:, None, :, None]                # [B,1,T,1]
        return (causal & key_valid) | (pad_query & key_valid)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        channelwise_loss_weight: Sequence[float] | None = None,
    ) -> MossTTSTrainOutput:
        if input_ids.dim() != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                f"input_ids must be (B, T, {self.config.n_vq + 1}), "
                f"got {tuple(input_ids.shape)}")
        bsz, seqlen, _ = input_ids.shape
        device = input_ids.device

        cos, sin = rope_tables(seqlen, self.config.head_dim, self.config.rope_theta,
                               device, self.language_model.embed_tokens.weight.dtype)
        cos = cos[None, None, :, :]   # [1, 1, T, D] broadcast over [B, H, T, D]
        sin = sin[None, None, :, :]
        attn_bias = {"cos": cos, "sin": sin,
                     "mask": self._build_attn_bias(attention_mask, seqlen, device)}

        inputs_embeds = self._compute_input_embeddings(input_ids)
        h = self.language_model(inputs_embeds, attn_bias,
                                gradient_checkpointing=self.gradient_checkpointing)

        text_logits = self.lm_heads[0](h)
        audio_logits = torch.stack(
            [head(h) for head in self.lm_heads[1:]], dim=2)          # [B, T, n_vq, V+1]
        # The pad code is structural (delay pattern / padding), never a target;
        # mask it exactly like the official modeling file does for heads i > 0.
        audio_logits[..., -1] = float("-inf")

        loss = channel_losses = None
        if labels is not None:
            if labels.dim() != 3 or labels.shape[:2] != (bsz, seqlen):
                raise ValueError(f"labels must be (B, T, n_vq+1), got {tuple(labels.shape)}")
            loss, channel_losses = _multi_head_ce_loss(
                text_logits, audio_logits, labels, channelwise_loss_weight)

        return MossTTSTrainOutput(text_logits, audio_logits, loss, channel_losses)


def _multi_head_ce_loss(
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    labels: torch.Tensor,
    channelwise_loss_weight: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of MossTTSDelayModel.forward loss (modeling_moss_tts.py).

    Numerically identical to a per-head loop of F.cross_entropy: rows are
    independent, so the 32 audio heads (shared vocab) are folded into one CE
    call over [B * n_vq * T, V] and only the text head runs separately.
    Cuts ~33 Python/kernel round-trips per step down to 2.
    """
    bsz = labels.size(0)
    n_heads = labels.size(2)
    all_token_nums = torch.sum(labels != -100, dim=1)                # [B, C]

    text_vocab = text_logits.size(-1)
    text_per_token = F.cross_entropy(
        text_logits.reshape(-1, text_vocab).float(),
        labels[..., 0].contiguous().view(-1),
        reduction="none",
    ).view(bsz, -1)                                                  # [B, T]

    n_vq = audio_logits.size(2)
    audio_vocab = audio_logits.size(-1)
    # audio_logits [B, T, C, V] is contiguous: plain reshape is zero-copy
    # (a permute-based version forced an extra bf16 contiguous copy, which
    # pushed peak activation memory over the edge on 24GB cards).
    # CE rows are independent, so the (B*T*C) row set matches the per-head
    # loop exactly. The per-sample T-sum is mathematically identical; fp32
    # addition order may differ from the loop by ~1e-6 (strided reduction).
    audio_per_token = F.cross_entropy(
        audio_logits.reshape(-1, audio_vocab).float(),
        labels[..., 1:].contiguous().view(-1),
        reduction="none",
    ).view(bsz, -1, n_vq)                                            # [B, T, C]
    audio_sums = audio_per_token.sum(dim=1)                          # [B, C-1]

    all_sum_losses = torch.cat(
        [text_per_token.sum(dim=-1, keepdim=True),
         audio_sums], dim=1)                                         # [B, C]

    if channelwise_loss_weight is not None:
        if len(channelwise_loss_weight) != n_heads:
            raise ValueError(
                f"channelwise_loss_weight length {len(channelwise_loss_weight)} "
                f"!= n_heads {n_heads}")
        w = torch.tensor(list(channelwise_loss_weight), device=all_sum_losses.device,
                         dtype=all_sum_losses.dtype)
        total_loss_per_channel = all_sum_losses.sum(dim=0)
        total_tokens_per_channel = all_token_nums.sum(dim=0).float().clamp(min=1.0)
        channel_losses = total_loss_per_channel / total_tokens_per_channel
        loss = (channel_losses * w).sum() / w.sum()
    else:
        total_tokens = all_token_nums.sum().float().clamp(min=1.0)
        loss = all_sum_losses.sum() / total_tokens
        channel_losses = all_sum_losses.sum(dim=0) / all_token_nums.sum(dim=0).clamp(min=1.0)
    return loss, channel_losses
