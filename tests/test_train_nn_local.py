"""Smoke tests for the MOSS-TTS-Local-Transformer-v1.5 training model."""

import pytest
import torch

from moss_tts_lite.nn import MossTTSConfig
from moss_tts_lite.nn_local import LocalConfig, TrainableMossTTSLocal


def _tiny_model(n_vq: int = 3, binary: bool = True) -> TrainableMossTTSLocal:
    cfg = MossTTSConfig(n_layers=2, hidden_size=64, n_heads=4, n_kv_heads=2,
                        head_dim=16, intermediate_size=128, text_vocab=101,
                        audio_vocab=50, n_vq=n_vq)
    lcfg = LocalConfig(n_layer=1, hidden_size=64, n_head=4)
    return TrainableMossTTSLocal(
        cfg, lcfg, [50] * n_vq, audio_pad_token_id=50,
        audio_assistant_slot_token_id=99, audio_end_token_id=100,
        use_binary_local_text_head=binary)


def _batch(model: TrainableMossTTSLocal, bsz=2, seqlen=12):
    n_vq = model.config.n_vq
    pad = model.audio_pad_token_id
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (bsz, seqlen, n_vq + 1))
    input_ids[..., 0] = torch.randint(0, 101, (bsz, seqlen))
    # left padding on row 0
    input_ids[0, :3] = 0
    input_ids[0, :3, 1:] = pad
    attn = torch.ones(bsz, seqlen, dtype=torch.bool)
    attn[0, :3] = False
    labels = input_ids.clone()
    labels[labels == pad] = -100
    labels[:, :4] = -100  # prompt region
    labels[0, :3] = -100
    # a couple of slot/end targets so the binary head path fires
    labels[0, 6, 0] = 99
    labels[0, 8, 0] = 100
    labels[1, 5, 0] = 99
    return input_ids, attn, labels


def test_forward_loss_finite_and_backward():
    model = _tiny_model()
    input_ids, attn, labels = _batch(model)
    out = model(input_ids, attn, labels,
                channelwise_loss_weight=[1.0, 1.0, 1.0, 1.0])
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.channel_losses.shape == (4,)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    # backbone must receive gradients
    assert model.transformer.layers[0].mlp.gate_proj.weight.grad is not None
    # local transformer + heads (tied to embeddings) must train
    assert model.local_transformer.h[0].attn.c_attn.weight.grad is not None
    assert model.audio_embeddings[0].weight.grad is not None
    if model.local_text_lm_head is not None:
        assert model.local_text_lm_head.weight.grad is not None


def test_binary_head_masking_matches_off():
    """The binary text head only supervises slot/end tokens; with it off the
    full-vocab head supervises all non-ignored text targets instead."""
    torch.manual_seed(1)
    m_on = _tiny_model(binary=True)
    m_off = _tiny_model(binary=False)
    m_off.load_state_dict(m_on.state_dict(), strict=False)
    input_ids, attn, labels = _batch(m_on)
    w = [1.0] * 4
    loss_on, _ = m_on(input_ids, attn, labels, w)[:2] \
        if False else (m_on(input_ids, attn, labels, w).loss,
                       m_on(input_ids, attn, labels, w).channel_losses)
    loss_off = m_off(input_ids, attn, labels, w).loss
    assert torch.isfinite(loss_on) and torch.isfinite(loss_off)
    assert not torch.allclose(loss_on, loss_off)  # different supervision


def test_full_vocab_text_head_path():
    model = _tiny_model(binary=False)
    input_ids, attn, labels = _batch(model)
    out = model(input_ids, attn, labels, [2.0, 1.0, 1.0, 1.0])
    assert torch.isfinite(out.loss)
    assert model.local_text_lm_head is None


def test_invalid_shapes_rejected():
    model = _tiny_model()
    with pytest.raises(ValueError):
        model(torch.zeros(2, 5, 33), None, None)
    input_ids, attn, _ = _batch(model)
    with pytest.raises(ValueError):
        model(input_ids, attn, torch.zeros(2, 5, 9))
