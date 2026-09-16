"""TrainableMossTTS vs MossTTSModel parity + loss/backward tests (CPU)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from moss_tts_lite.model import AUDIO_PAD_CODE, N_VQ, MossTTSModel
from moss_tts_lite.nn import TrainableMossTTS
from tests._train_fixtures import TEXT_VOCAB_SMALL, small_weights, write_small_model_dir


@pytest.fixture(scope="module")
def small_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("small_model")
    write_small_model_dir(str(d))
    return str(d)


def _random_ids(seqlen=48, batch=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(batch, seqlen, N_VQ + 1, dtype=torch.long)
    ids[..., 0] = torch.randint(0, TEXT_VOCAB_SMALL, (batch, seqlen), generator=g)
    ids[..., 1:] = torch.randint(0, AUDIO_PAD_CODE, (batch, seqlen, N_VQ), generator=g)
    return ids


def test_from_pretrained_strict(small_dir):
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.float32)
    cfg = model.config
    assert (cfg.n_layers, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads,
            cfg.head_dim) == (2, 64, 4, 2, 16)
    assert cfg.text_vocab == TEXT_VOCAB_SMALL and cfg.n_vq == N_VQ
    weights = small_weights()
    assert set(dict(model.named_parameters()).keys()) == set(weights.keys())
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), weights[name]), name


def test_logits_match_inference(small_dir):
    weights = small_weights()
    ref = MossTTSModel({k: v.clone() for k, v in weights.items()}, device="cpu",
                       dtype=torch.float32, max_seq_len=256)
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.float32)
    model.eval()

    ids = _random_ids(seqlen=40)
    with torch.no_grad():
        hs = ref.prefill(ids)
        text_ref = ref.text_logits(hs)          # [T, V]
        audio_ref = ref.audio_logits(hs)        # [T, 32, 1025]
        out = model(ids)
    assert out.text_logits.shape == (1, 40, TEXT_VOCAB_SMALL)
    assert out.audio_logits.shape == (1, 40, 32, 1025)
    assert torch.allclose(out.text_logits[0], text_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out.audio_logits[0], audio_ref, atol=1e-4, rtol=1e-4)
    assert torch.isinf(out.audio_logits[..., AUDIO_PAD_CODE]).all()
    assert out.loss is None


def test_logits_match_inference_bf16(small_dir):
    """Same parity in bf16 (training dtype) with a bf16-appropriate tolerance."""
    weights = small_weights()
    ref = MossTTSModel({k: v.clone() for k, v in weights.items()}, device="cpu",
                       dtype=torch.bfloat16, max_seq_len=256)
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.bfloat16)
    model.eval()
    ids = _random_ids(seqlen=32)
    with torch.no_grad():
        hs = ref.prefill(ids)
        text_ref = ref.text_logits(hs).float()
        audio_ref = ref.audio_logits(hs).float()
        out = model(ids)
    assert out.text_logits.dtype == torch.bfloat16
    t = out.text_logits[0].float()
    a = out.audio_logits[0].float()
    assert (t - text_ref).abs().max().item() < 0.05
    assert (a[..., :AUDIO_PAD_CODE] - audio_ref[..., :AUDIO_PAD_CODE]).abs().max().item() < 0.05
    assert torch.isinf(a[..., AUDIO_PAD_CODE]).all()


def test_loss_backward_and_grads(small_dir):
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.float32)
    ids = _random_ids(seqlen=24, batch=2, seed=3)
    labels = ids.clone()
    # mask: left-pad rows for sample 1 (shorter by 6), no loss on first 5 steps
    attention = torch.ones(2, 24, dtype=torch.bool)
    attention[1, :6] = False
    labels[1, :7] = -100
    labels[:, :5] = -100
    audio = labels[..., 1:]
    labels[..., 1:] = torch.where(audio == AUDIO_PAD_CODE,
                                  torch.full_like(audio, -100), audio)

    out = model(ids, attention_mask=attention, labels=labels,
                channelwise_loss_weight=[1.0] + [1.0] * N_VQ)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.channel_losses.shape == (N_VQ + 1,)
    out.loss.backward()
    n_with_grad = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        if p.grad.abs().sum() > 0:
            n_with_grad += 1
    assert n_with_grad > 0
    # losses must be finite across channels too
    assert torch.isfinite(out.channel_losses).all()


def test_channelwise_weighting(small_dir):
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.float32)
    model.eval()
    ids = _random_ids(seqlen=16, seed=7)
    labels = ids.clone()
    labels[:, :4] = -100
    labels[:, 5, 1] = -100   # audio channel 0 has one fewer valid token
    with torch.no_grad():
        out = model(ids, labels=labels)
        w = [1.0] + [1.0] * N_VQ
        out_w = model(ids, labels=labels, channelwise_loss_weight=w)
    # equal weights reproduce the mean of channel losses
    assert torch.allclose(out_w.loss, out_w.channel_losses.mean(), atol=1e-6)
    # default (None) is token-weighted, which differs once channel counts differ
    assert not torch.allclose(out_w.loss, out.loss)


def test_gradient_checkpointing(small_dir):
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.train()
    ids = _random_ids(seqlen=16, seed=9)
    labels = ids.clone()
    labels[:, :3] = -100
    out = model(ids, labels=labels)
    out.loss.backward()
    assert torch.isfinite(out.loss)


def test_left_padding_causality(small_dir):
    """Left-padded batch keeps causal: suffix mutation leaves prefix logits."""
    model = TrainableMossTTS.from_pretrained(small_dir, dtype=torch.float32)
    model.eval()
    ids = _random_ids(seqlen=20, batch=2, seed=11)
    attention = torch.ones(2, 20, dtype=torch.bool)
    attention[1, :5] = False
    ids2 = ids.clone()
    ids2[:, 12:] = _random_ids(seqlen=8, batch=2, seed=12)
    with torch.no_grad():
        a = model(ids, attention_mask=attention).text_logits
        b = model(ids2, attention_mask=attention).text_logits
    # row 0 is unpadded; row 1's real tokens start at 5 (pads may legally change)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5), "causality violated"
    assert torch.allclose(a[1, 5:12], b[1, 5:12], atol=1e-5), "causality violated"
    # no NaNs despite the padded rows
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
