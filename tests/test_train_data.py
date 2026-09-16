"""Sequence packing / collate alignment tests (CPU).

Structure is verified against the official computing_loss layout by hand, and
-- when transformers + the MOSS tokenizer are reachable -- against the official
`MossTTSDelayProcessor` itself.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from moss_tts_lite.bpe import QwenBPE
from moss_tts_lite.data import (MossTTSTrainDataset, build_computing_loss_ids,
                                build_generation_prompt_ids, get_unified_codes,
                                pad_batch)
from moss_tts_lite.model import (AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID,
                                 AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_PAD_CODE,
                                 AUDIO_START_TOKEN_ID, AUDIO_USER_SLOT_TOKEN_ID,
                                 N_VQ, PAD_TOKEN_ID)
from moss_tts_lite.prompt import apply_delay_pattern
from tests._train_fixtures import (make_records, tiny_tokenizer,
                                   write_small_model_dir)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok_model")
    write_small_model_dir(str(d), hidden=8, n_layers=1, n_vq=N_VQ)
    return tiny_tokenizer(str(d))


def test_generation_prompt_layout(tokenizer):
    record = make_records(n=1, frames=8, seed=5)[0]
    prompt = build_generation_prompt_ids(record, tokenizer)
    text_ch = prompt[:, 0]
    # no reference: no audio block in the prompt, audio channels all pad
    assert not (text_ch == AUDIO_START_TOKEN_ID).any()
    assert (prompt[:, 1:] == AUDIO_PAD_CODE).all()
    assert text_ch[-1] == tokenizer.encode("<|im_start|>assistant\n")[-1]
    # the generation prompt ends with the assistant generation start
    ids = text_ch.tolist()
    assert ids == tokenizer.encode(
        "<|im_start|>user\n" + _user_content(record) + "<|im_end|>\n"
        "<|im_start|>assistant\n")


def _user_content(record):
    from moss_tts_lite.data import build_user_message
    msg = build_user_message(text=record["text"], language=record["language"])
    return msg["content"]


def test_computing_loss_layout_no_reference(tokenizer):
    record = make_records(n=1, frames=8, seed=6)[0]
    ids = build_computing_loss_ids(record, tokenizer)
    text_ch = ids[:, 0]

    start = int((text_ch == AUDIO_START_TOKEN_ID).nonzero()[0])
    end = int((text_ch == AUDIO_END_TOKEN_ID).nonzero()[0])
    T = 8
    # assistant audio block: start + T gen-slots + (n_vq-1) delay-slots + end
    assert end - start == 1 + T + (N_VQ - 1)
    block = text_ch[start + 1: start + 1 + T + N_VQ - 1]
    assert (block[:T] == AUDIO_GEN_SLOT_TOKEN_ID).all()
    assert (block[T:] == AUDIO_DELAY_SLOT_TOKEN_ID).all()

    # audio channels carry the delay pattern of the target codes
    codes = torch.tensor(record["audio_codes"], dtype=torch.long)
    delayed = apply_delay_pattern(codes, AUDIO_PAD_CODE)
    audio_block = ids[start + 1: start + 1 + delayed.shape[0], 1:]
    assert torch.equal(audio_block, delayed)
    # before/after the block the audio channels are structural pad
    assert (ids[:start + 1, 1:] == AUDIO_PAD_CODE).all()
    assert (ids[end:, 1:] == AUDIO_PAD_CODE).all()


def test_computing_loss_layout_with_reference(tokenizer):
    record = make_records(n=1, frames=5, seed=7, with_reference=True)[0]
    ids = build_computing_loss_ids(record, tokenizer)
    text_ch = ids[:, 0]
    starts = (text_ch == AUDIO_START_TOKEN_ID).nonzero().flatten().tolist()
    ends = (text_ch == AUDIO_END_TOKEN_ID).nonzero().flatten().tolist()
    assert len(starts) == 2 and len(ends) == 2

    # user block: user-slot tokens for both gen and delay segments
    ref_codes = torch.tensor(record["ref_audio_codes"], dtype=torch.long)
    ref_delayed = apply_delay_pattern(ref_codes, AUDIO_PAD_CODE)
    user_block = ids[starts[0] + 1: starts[0] + 1 + 6 + N_VQ - 1, :]
    assert (user_block[:6, 0] == AUDIO_USER_SLOT_TOKEN_ID).all()
    assert (user_block[6:, 0] == AUDIO_USER_SLOT_TOKEN_ID).all()
    assert torch.equal(user_block[:, 1:], ref_delayed)

    # assistant block still present after the user block
    assert starts[1] > ends[0]


def test_pack_record_loss_mask(tokenizer):
    records = make_records(n=2, frames=6, seed=8)
    ds = MossTTSTrainDataset(records, tokenizer)
    for record in records:
        item = ds.pack_record(record)
        prompt_len = build_generation_prompt_ids(record, tokenizer).shape[0]
        mask = item["loss_mask"]
        assert mask.shape[0] == item["input_ids"].shape[0] - 1
        assert not mask[: prompt_len - 1].any()
        assert mask[prompt_len - 1:].all()


def test_collate_left_padding(tokenizer):
    records = make_records(n=2, frames=6, seed=9)
    ds = MossTTSTrainDataset(records, tokenizer)
    items = [ds.pack_record(r) for r in records]
    batch = ds.collate_fn(items)

    input_ids, attention, labels = (batch["input_ids"], batch["attention_mask"],
                                    batch["labels"])
    assert input_ids.shape[0] == 2 and labels.shape[1] == input_ids.shape[1]
    assert input_ids.shape[2] == N_VQ + 1

    for b, item in enumerate(items):
        t = item["input_ids"].shape[0]
        pad = input_ids.shape[1] - (t - 1)  # input is truncated by 1
        assert pad >= 0
        # attention: left pad region False
        assert not attention[b, :pad].any() and attention[b, pad:].all()
        if pad:
            assert (input_ids[b, :pad, 0] == PAD_TOKEN_ID).all()
            assert (input_ids[b, :pad, 1:] == AUDIO_PAD_CODE).all()
        # labels equal shifted input ids where the loss mask is on, modulo the
        # structural audio-pad -> -100 masking (official collate behavior)
        mask = item["loss_mask"]
        shifted = item["input_ids"][1:].clone()
        audio = shifted[:, 1:]
        shifted[:, 1:] = torch.where(audio == AUDIO_PAD_CODE,
                                     torch.full_like(audio, -100), audio)
        lab = labels[b, pad:]
        assert torch.equal(lab[mask], shifted[mask])
        # non-loss prefix masked
        assert (lab[: mask.nonzero()[0].item()] == -100).all()
        # structural audio pad positions masked
        audio_lab = lab[:, 1:]
        assert not (audio_lab == AUDIO_PAD_CODE).any()


def test_pad_batch_matches_official_layout(tokenizer):
    a = torch.full((3, 33), 7, dtype=torch.long)
    b = torch.full((5, 33), 9, dtype=torch.long)
    out = pad_batch([a, b])
    padded, attention = out["input_ids"], out["attention_mask"]
    assert padded.shape == (2, 5, 33)
    assert (padded[0, :2, 0] == PAD_TOKEN_ID).all()
    assert (padded[0, :2, 1:] == AUDIO_PAD_CODE).all()
    assert (padded[1] == 9).all()
    assert (attention == torch.tensor([[False, False, True, True, True],
                                       [True] * 5])).all()


# ---------------------------------------------------------------------------
# prepare_data audio helpers
# ---------------------------------------------------------------------------

def test_loudness_normalize_clamp():
    import numpy as np
    from moss_tts_lite.prepare_data import loudness_normalize
    # gain is clamped to +-3 dB around the input level (-20 dBFS target)
    loud = np.full(1000, 0.5, dtype=np.float32)      # -6.0 dBFS
    out = loudness_normalize(loud)
    db = 10.0 * np.log10(float(np.mean(out.astype(np.float64) ** 2)) + 1e-9)
    assert abs(db - (-6.0206 - 3.0)) < 0.1          # gain hit -3 dB
    quiet = np.full(1000, 1e-3, dtype=np.float32)   # -60 dBFS
    out2 = loudness_normalize(quiet)
    db2 = 10.0 * np.log10(float(np.mean(out2.astype(np.float64) ** 2)) + 1e-9)
    assert abs(db2 - (-60.0 + 3.0)) < 0.1           # gain hit +3 dB


def test_load_wav_mono_resample(tmp_path):
    import numpy as np
    import soundfile as sf
    from moss_tts_lite.prepare_data import load_audio_for_codec
    t = np.linspace(0, 0.5, 22050, dtype=np.float32)  # 0.5 s at 44.1 kHz
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "a44k.wav")
    sf.write(path, np.stack([wav, wav], axis=1), 44100, subtype="PCM_16")
    out = load_audio_for_codec(path, 24000)
    assert out.dtype == np.float32 and out.ndim == 1
    assert 11800 < len(out) < 12200                  # 0.5 s -> 12000 samples


def test_resample_numpy_port_matches_torchaudio():
    torchaudio = pytest.importorskip("torchaudio")
    import numpy as np
    from moss_tts_lite.prepare_data import _resample_sinc_numpy
    rng = np.random.default_rng(0)
    for orig, new in ((44100, 24000), (48000, 24000), (16000, 24000),
                      (22050, 24000), (8000, 24000)):
        wav = (rng.standard_normal(16000) * 0.1).astype(np.float32)
        mine = _resample_sinc_numpy(wav, orig, new)
        ref = torchaudio.functional.resample(
            torch.from_numpy(wav.copy()), orig, new).numpy()
        assert mine.shape == ref.shape
        assert np.abs(mine - ref).max() < 1e-6, (orig, new)


# ---------------------------------------------------------------------------
# Official processor parity (optional: transformers + MOSS tokenizer)
# ---------------------------------------------------------------------------

def _load_official():
    """Return (processor, lite_tokenizer) or None."""
    try:
        import transformers  # noqa: F401
        import torchaudio  # noqa: F401
        from transformers import AutoTokenizer
    except Exception:
        return None
    root = os.environ.get("MOSS_TTS_ROOT",
                          os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    workspace = os.path.dirname(os.path.abspath(root))
    official_root = os.path.join(workspace, "MOSS-TTS")
    if not os.path.isdir(os.path.join(official_root, "moss_tts_delay")):
        return None
    sys.path.insert(0, official_root)
    try:
        from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig
        from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor
    except Exception:
        return None

    model_dir = os.path.join(root, "models", "MOSS-TTS-v1.5")
    if not os.path.exists(os.path.join(model_dir, "vocab.json")):
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(
                "OpenMOSS-Team/MOSS-TTS-v1.5",
                allow_patterns=["vocab.json", "merges.txt", "tokenizer.json",
                                "tokenizer_config.json", "added_tokens.json",
                                "special_tokens_map.json", "chat_template.jinja"],
            )
        except Exception:
            return None
    if not os.path.exists(os.path.join(model_dir, "vocab.json")):
        return None
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        processor = MossTTSDelayProcessor(tokenizer=hf_tok, audio_tokenizer=None,
                                          model_config=MossTTSDelayConfig())
        lite_tok = QwenBPE(os.path.join(model_dir, "vocab.json"),
                           os.path.join(model_dir, "merges.txt"),
                           os.path.join(model_dir, "tokenizer.json"))
        return processor, lite_tok
    except Exception:
        return None


def test_official_processor_parity():
    loaded = _load_official()
    if loaded is None:
        pytest.skip("official processor unavailable (needs transformers, "
                    "torchaudio, MOSS-TTS checkout, and MOSS tokenizer files)")
    processor, lite_tok = loaded

    g = torch.Generator().manual_seed(123)
    cases = []
    for i, frames in enumerate((5, 17)):
        record = {
            "audio_codes": torch.randint(0, 1024, (frames, 32), generator=g).tolist(),
            "text": "Hello world, this is parity test %d." % i,
            "language": "English",
        }
        if i == 1:
            record["ref_audio_codes"] = torch.randint(
                0, 1024, (9, 32), generator=g).tolist()
        cases.append(record)

    for record in cases:
        n_vq = 32
        # --- official path -------------------------------------------------
        user_kwargs = {k: record[k] for k in ("text", "language") if k in record}
        ref = None
        if "ref_audio_codes" in record:
            ref = [torch.tensor(record["ref_audio_codes"], dtype=torch.long)]
        user_message = processor.build_user_message(reference=ref, **user_kwargs)
        assistant = processor.build_assistant_message(
            audio_codes_list=[torch.tensor(record["audio_codes"], dtype=torch.long)])
        official_conv = processor([[user_message, assistant]],
                                  mode="computing_loss", n_vq=n_vq)
        official_prompt = processor([[user_message]], mode="generation", n_vq=n_vq)
        # --- lite path -----------------------------------------------------
        mine = build_computing_loss_ids(record, lite_tok)
        mine_prompt = build_generation_prompt_ids(record, lite_tok)
        assert torch.equal(mine, official_conv["input_ids"][0]), \
            "computing_loss sequence diverges from official processor"
        assert torch.equal(mine_prompt, official_prompt["input_ids"][0]), \
            "generation prompt diverges from official processor"
        # loss mask semantics: prompt length identical
        assert mine_prompt.shape[0] == official_prompt["input_ids"][0].shape[0]

    # pad layout parity (official _pad vs pad_batch) on a mixed batch
    seqs = [mine, build_computing_loss_ids(cases[0], lite_tok)]
    official_seqs = [official_conv["input_ids"][0],
                     _official_conv(processor, cases[0])]
    mine_padded = pad_batch(seqs)
    official_padded = processor._pad(official_seqs)
    assert torch.equal(mine_padded["input_ids"], official_padded["input_ids"])
    assert torch.equal(mine_padded["attention_mask"],
                       official_padded["attention_mask"].bool())


def _official_conv(processor, record):
    user_kwargs = {k: record[k] for k in ("text", "language") if k in record}
    ref = None
    if "ref_audio_codes" in record:
        ref = [torch.tensor(record["ref_audio_codes"], dtype=torch.long)]
    user_message = processor.build_user_message(reference=ref, **user_kwargs)
    assistant = processor.build_assistant_message(
        audio_codes_list=[torch.tensor(record["audio_codes"], dtype=torch.long)])
    return processor([[user_message, assistant]], mode="computing_loss",
                     n_vq=32)["input_ids"][0]
