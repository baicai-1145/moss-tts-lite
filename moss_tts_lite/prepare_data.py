"""Encode audio into training JSONL with the official MOSS-Audio-Tokenizer.

Produces records compatible with both the official finetuning pipeline and
`moss_tts_lite.train`: each `audio` path becomes `audio_codes` (nested int
list, [T, n_vq]) and `ref_audio` / `reference_audio` / `reference` paths
become `ref_audio_codes` / `reference_audio_codes`.

Audio loading uses soundfile; the resampler is a numpy port of
torchaudio.functional.resample's default band-limited sinc interpolation
(torchaudio is used directly when installed). transformer's AutoModel loads the
codec (trust_remote_code) -- a lazy, optional dependency.

Usage:
    python -m moss_tts_lite.prepare_data --codec-dir <MOSS-Audio-Tokenizer> \
        --input-jsonl raw.jsonl --output-jsonl train.jsonl \
        --device cuda --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

SAMPLING_RATE = 24000  # MossTTSDelayConfig.sampling_rate (official codec rate)


# ---------------------------------------------------------------------------
# waveform IO + preprocessing (mirrors MossTTSDelayProcessor.encode_audios_*)
# ---------------------------------------------------------------------------

def load_wav_mono(path: str) -> np.ndarray:
    """Read any audio file as float32 mono [T] at its native rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, dtype=np.float32)
    else:
        data = data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def resample_sinc(wav: np.ndarray, orig_freq: int, new_freq: int,
                  lowpass_filter_width: int = 6, rolloff: float = 0.99) -> np.ndarray:
    """Resample to `new_freq`; prefers torchaudio, falls back to the numpy port."""
    wav = np.asarray(wav)
    if orig_freq == new_freq:
        return wav
    gcd = math.gcd(int(orig_freq), int(new_freq))
    if orig_freq // gcd == new_freq // gcd:
        return wav.copy()
    try:  # prefer the reference implementation when available
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(wav), orig_freq, new_freq).numpy()
    except Exception:
        return _resample_sinc_numpy(wav, orig_freq, new_freq,
                                     lowpass_filter_width, rolloff)


def _resample_sinc_numpy(wav: np.ndarray, orig_freq: int, new_freq: int,
                         lowpass_filter_width: int = 6,
                         rolloff: float = 0.99) -> np.ndarray:
    """numpy port of torchaudio.functional.resample (sinc_interp_hann defaults)."""
    gcd = math.gcd(int(orig_freq), int(new_freq))
    o, n = orig_freq // gcd, new_freq // gcd

    base_freq = min(o, n) * rolloff
    width = math.ceil(lowpass_filter_width * o / base_freq)
    dt = np.float32 if wav.dtype == np.float32 else np.float64

    idx = (np.arange(-width, width + o, dtype=dt) / o)[None, None, :]
    t = (np.arange(0, -n, -1, dtype=dt)[:, None, None] / n + idx)
    t *= base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * np.pi / lowpass_filter_width / 2) ** 2
    t = t * np.pi
    scale = base_freq / o
    safe = np.where(t == 0, dt(1.0), t)
    kernels = np.where(t == 0, dt(1.0), np.sin(safe) / safe)
    kernels = (kernels * window * scale).astype(dt)     # [n, 1, K]

    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(wav.astype(dt), (width, width + o))
    windows = sliding_window_view(padded, kernels.shape[-1])[::o]   # [L, K]
    # conv1d semantics: out[c, j] = sum_i padded[j*o + i] * kernel[c, i];
    # torch then transposes to time-major [j, c] before flattening.
    out = (windows @ kernels.reshape(n, -1).T).reshape(-1)          # [L * n]
    target_length = math.ceil(n * wav.shape[-1] / o)
    return out[:target_length].astype(np.float32)


def loudness_normalize(wav: np.ndarray, target_dbfs: float = -20.0,
                       gain_range: tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    """`MossTTSDelayProcessor.loudness_normalize` in numpy (fp32)."""
    wav = wav.astype(np.float32)
    if wav.size == 0:
        return wav
    current_dbfs = 10.0 * np.log10(np.mean(wav ** 2) + 1e-9)
    gain = float(np.clip(target_dbfs - current_dbfs, *gain_range))
    return wav * np.float32(10.0 ** (gain / 20.0))


def load_wav_stereo(path: str) -> tuple[np.ndarray, int]:
    """Read audio as float32 [C, T] with C in {1, 2} at its native rate.

    Matches the v2 processor: mono is duplicated to two channels, extra
    channels are truncated to the first two.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data.T, dtype=np.float32), int(sr)


def load_audio_for_codec(path: str, target_sr: int = SAMPLING_RATE,
                         channels: int = 1) -> np.ndarray:
    """Read + (mono|stereo) + resample + loudness-normalize (official path).

    channels=1 keeps the v1 codec preprocessing (mono); channels=2 follows
    the v2 (48 kHz stereo) processor.
    """
    if channels == 2:
        wav, sr = load_wav_stereo(path)                     # [C, T]
        if sr != target_sr:
            wav = np.stack([resample_sinc(wav[c], sr, target_sr)
                            for c in range(wav.shape[0])])
    else:
        wav, sr = load_wav_mono(path)
        if sr != target_sr:
            wav = resample_sinc(wav, sr, target_sr)
    return loudness_normalize(wav)


# ---------------------------------------------------------------------------
# codec encoding (MossTTSDelayProcessor.encode_audios_from_wav)
# ---------------------------------------------------------------------------

class CodecEncoder:
    """Thin wrapper over the MOSS-Audio-Tokenizer (CAT) AutoModel."""

    def __init__(self, codec_dir: str, device: str = "cpu"):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "prepare_data requires transformers (loads the MOSS-Audio-Tokenizer "
                "with trust_remote_code): pip install transformers") from exc
        self.model = AutoModel.from_pretrained(codec_dir, trust_remote_code=True)
        self.model.to(device).eval()
        self.device = device
        # v1 (MOSS-Audio-Tokenizer): 24 kHz mono, n_vq 32.
        # v2 (MOSS-Audio-Tokenizer-v2): 48 kHz stereo, TTS uses the first 12.
        cfg_path = Path(codec_dir) / "config.json"
        codec_cfg = json.loads(cfg_path.read_text("utf-8")) if cfg_path.exists() else {}
        self.sampling_rate = int(codec_cfg.get("sample_rate", 24000))
        self.channels = int(codec_cfg.get("number_channels", 1))
        self.default_n_vq = 12 if self.channels == 2 or self.sampling_rate == 48000 else 32

    def encode(self, wav_list: list[np.ndarray],
               n_vq: Optional[int] = None) -> list[np.ndarray]:
        import torch

        if n_vq is None:
            n_vq = self.default_n_vq
        # v1 path feeds [T] (mono); v2 feeds [C, T] — both as float32.
        wavs = [torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
                .to(self.device) for w in wav_list]
        with torch.no_grad():
            if hasattr(self.model, "batch_encode"):
                enc = self.model.batch_encode(wavs, num_quantizers=n_vq)
                audio_codes = enc.audio_codes          # (NQ, B, T)
                audio_codes_lengths = enc.audio_codes_lengths
            else:
                max_len = max(int(w.shape[-1]) for w in wavs)
                input_values = torch.zeros(len(wavs), self.channels, max_len,
                                           dtype=torch.float32, device=self.device)
                padding_mask = torch.zeros(len(wavs), max_len, dtype=torch.bool,
                                           device=self.device)
                for i, w in enumerate(wavs):
                    this_len = int(w.shape[-1])
                    input_values[i, :, :this_len] = w
                    padding_mask[i, :this_len] = True
                enc = self.model.encode(input_values, padding_mask=padding_mask,
                                        num_quantizers=n_vq, return_dict=True)
                audio_codes = enc.audio_codes
                audio_codes_lengths = enc.audio_codes_lengths
        if audio_codes is None or audio_codes_lengths is None:
            raise RuntimeError("codec encode() returned empty audio_codes")
        return [audio_codes[:, i, : int(audio_codes_lengths[i].item())]
                .transpose(0, 1).to(torch.long).cpu().numpy() for i in range(len(wavs))]


# ---------------------------------------------------------------------------
# record plumbing (official prepare_data.py logic)
# ---------------------------------------------------------------------------

def normalize_audio_path_list(value: Any, field_name: str,
                              allow_none: bool = False) -> Optional[list[Optional[str]]]:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if allow_none:
            if not all(item is None or isinstance(item, str) for item in value):
                raise ValueError(f"`{field_name}` must contain strings/nulls.")
        elif not all(isinstance(item, str) for item in value):
            raise ValueError(f"`{field_name}` must be a string or list of strings.")
        return value
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


def collect_paths(records: list[dict], field_name: str) -> list[str]:
    paths: list[str] = []
    for record in records:
        values = normalize_audio_path_list(record.get(field_name), field_name,
                                           allow_none=(field_name == "reference"))
        if values is not None:
            paths.extend(v for v in values if v is not None)
    return list(dict.fromkeys(paths))


def batch_encode_paths(encoder: CodecEncoder, paths: list[str], batch_size: int,
                       n_vq: Optional[int]) -> dict[str, list[list[int]]]:
    codes: dict[str, list[list[int]]] = {}
    if n_vq is None:
        n_vq = encoder.default_n_vq
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        encoded = encoder.encode(
            [load_audio_for_codec(p, encoder.sampling_rate, encoder.channels)
             for p in batch_paths], n_vq)
        for path, c in zip(batch_paths, encoded):
            codes[path] = c.astype(int).tolist()
        print(f"[prepare_data] encoded {min(start + batch_size, len(paths))}/"
              f"{len(paths)} audios", flush=True)
    return codes


def main(argv: Optional[list[str]] = None) -> None:
    import sys
    parser = argparse.ArgumentParser(
        prog="python -m moss_tts_lite.prepare_data",
        description="Encode audio files into MOSS-TTS finetuning JSONL "
                    "(official-compatible audio_codes fields).")
    parser.add_argument("--codec-dir", type=str, required=True,
                        help="MOSS-Audio-Tokenizer dir or HF repo id.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument("--skip-reference-audio-codes", action="store_true",
                        help="Only encode the target `audio` field.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit(f"No records in {args.input_jsonl}")

    encoder = CodecEncoder(args.codec_dir, args.device)

    target_paths = []
    for index, record in enumerate(records):
        audio_path = record.get("audio")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Record {index} is missing a valid `audio` field.")
        target_paths.append(audio_path)
    target_codes = batch_encode_paths(encoder, target_paths, args.batch_size, args.n_vq)
    for record in records:
        record["audio_codes"] = target_codes[record["audio"]]

    if not args.skip_reference_audio_codes:
        unique_ref_paths: list[str] = []
        for field_name in ("ref_audio", "reference_audio", "reference"):
            for p in collect_paths(records, field_name):
                if p not in unique_ref_paths:
                    unique_ref_paths.append(p)
        if unique_ref_paths:
            ref_codes = batch_encode_paths(encoder, unique_ref_paths,
                                           args.batch_size, args.n_vq)

            def codes_of(path: Optional[str]) -> Optional[list[list[int]]]:
                return None if path is None else ref_codes[path]

            for record in records:
                ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
                if ref_audio is not None:
                    if len(ref_audio) != 1:
                        raise ValueError("`ref_audio` only supports a single path.")
                    record["ref_audio_codes"] = ref_codes[ref_audio[0]]
                reference_audio = normalize_audio_path_list(record.get("reference_audio"),
                                                            "reference_audio")
                if reference_audio is not None:
                    record["reference_audio_codes"] = [ref_codes[p]
                                                       for p in reference_audio]
                reference = normalize_audio_path_list(record.get("reference"),
                                                      "reference", allow_none=True)
                if reference is not None:
                    record["reference_audio_codes"] = [codes_of(p) for p in reference]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[prepare_data] wrote {len(records)} records to {out_path}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
