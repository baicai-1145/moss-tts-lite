"""Shared fixtures for training tests: small checkpoints + tiny tokenizer.

The small weight dict uses the *real* MOSS-TTS-v1.5 safetensors key names
(small shapes) so `TrainableMossTTS.load_state_dict(strict=True)` and
`MossTTSModel` both accept it. The tiny QwenBPE keeps the real special-token
ids (>= 151643) so `moss_tts_lite.data` sequence building works unmodified.
"""

from __future__ import annotations

import json
import os

import torch

TEXT_VOCAB_SMALL = 151700  # > max special id 151663, tiny enough for CPU tests

SPECIAL_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_pad|>": 151646,
    "<|audio_user_slot_pad|>": 151647,
    "<|audio_pad2|>": 151648,
    "<|audio_pad3|>": 151649,
    "<|audio_start|>": 151652,
    "<|audio_end|>": 151653,
    "<|audio_user_slot|>": 151654,
    "<|audio_pad4|>": 151655,
    "<|audio_assistant_gen_slot|>": 151656,
    "<|audio_pad5|>": 151657,
    "<|audio_pad6|>": 151658,
    "<|audio_pad7|>": 151659,
    "<|audio_pad8|>": 151660,
    "<|audio_pad9|>": 151661,
    "<|audio_assistant_delay_slot|>": 151662,
}


def small_weights(*, hidden=64, n_layers=2, n_heads=4, n_kv=2, head_dim=16,
                  intermediate=110, text_vocab=TEXT_VOCAB_SMALL, audio_vocab=1024,
                  n_vq=32, seed=0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=0.02):
        return torch.randn(*shape, generator=g) * scale

    w: dict[str, torch.Tensor] = {}
    w["language_model.embed_tokens.weight"] = rand(text_vocab, hidden)
    for i in range(n_layers):
        p = f"language_model.layers.{i}"
        w[f"{p}.self_attn.q_proj.weight"] = rand(n_heads * head_dim, hidden)
        w[f"{p}.self_attn.k_proj.weight"] = rand(n_kv * head_dim, hidden)
        w[f"{p}.self_attn.v_proj.weight"] = rand(n_kv * head_dim, hidden)
        w[f"{p}.self_attn.o_proj.weight"] = rand(hidden, n_heads * head_dim)
        w[f"{p}.self_attn.q_norm.weight"] = 1.0 + rand(head_dim, scale=0.01)
        w[f"{p}.self_attn.k_norm.weight"] = 1.0 + rand(head_dim, scale=0.01)
        w[f"{p}.input_layernorm.weight"] = 1.0 + rand(hidden, scale=0.01)
        w[f"{p}.post_attention_layernorm.weight"] = 1.0 + rand(hidden, scale=0.01)
        w[f"{p}.mlp.gate_proj.weight"] = rand(intermediate, hidden)
        w[f"{p}.mlp.up_proj.weight"] = rand(intermediate, hidden)
        w[f"{p}.mlp.down_proj.weight"] = rand(hidden, intermediate)
    w["language_model.norm.weight"] = 1.0 + rand(hidden, scale=0.01)
    for i in range(n_vq):
        w[f"emb_ext.{i}.weight"] = rand(audio_vocab + 1, hidden)
    w["lm_heads.0.weight"] = rand(text_vocab, hidden)
    for i in range(n_vq):
        w[f"lm_heads.{i + 1}.weight"] = rand(audio_vocab + 1, hidden)
    return w


def write_tiny_tokenizer(model_dir: str) -> None:
    """Byte-level char vocab + a couple merges + real special-token ids."""
    def bytes_to_unicode():
        bs = (list(range(ord("!"), ord("~") + 1))
              + list(range(ord("\xa1"), ord("\xac") + 1))
              + list(range(ord("\xae"), ord("\xff") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, (chr(c) for c in cs)))

    b2u = bytes_to_unicode()
    vocab = {c: i for i, c in enumerate(sorted(set(b2u.values())))}
    merges = [("h", "e"), ("t", "h"), ("th", "e"), ("l", "o"), ("hel", "lo")]
    nxt = len(vocab)
    for a, b in merges:
        vocab[a + b] = nxt
        nxt += 1
    with open(os.path.join(model_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(os.path.join(model_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    with open(os.path.join(model_dir, "added_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(SPECIAL_TOKENS, f)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "moss_tts_delay", "note": "tiny test checkpoint"}, f)


def write_small_model_dir(model_dir: str, weights: dict[str, torch.Tensor] | None = None,
                          **kwargs) -> dict[str, torch.Tensor]:
    """model.safetensors + tiny tokenizer files, loadable by from_pretrained."""
    from moss_tts_lite.train import write_safetensors

    os.makedirs(model_dir, exist_ok=True)
    weights = weights if weights is not None else small_weights(**kwargs)
    write_safetensors(weights, model_dir)
    write_tiny_tokenizer(model_dir)
    return weights


def tiny_tokenizer(model_dir: str):
    from moss_tts_lite.bpe import QwenBPE
    return QwenBPE(os.path.join(model_dir, "vocab.json"),
                   os.path.join(model_dir, "merges.txt"),
                   os.path.join(model_dir, "added_tokens.json"))


def make_records(n=2, frames=8, n_vq=32, seed=1,
                 with_reference=False) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    records = []
    texts = ["Hello world, this is a test.", "The quick brown fox."]
    for i in range(n):
        record = {
            "audio_codes": torch.randint(0, 1024, (frames, n_vq),
                                         generator=g).tolist(),
            "text": texts[i % len(texts)],
            "language": "English",
        }
        if with_reference and i == 0:
            record["ref_audio_codes"] = torch.randint(
                0, 1024, (6, n_vq), generator=g).tolist()
        records.append(record)
    return records
