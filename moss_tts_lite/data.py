"""SFT dataset: official computing_loss sequence packing on the lite stack.

Token-for-token port of `MossTTSDelayProcessor` (mode="computing_loss" /
"generation") + `moss_tts_delay.finetuning.dataset.MossTTSSFTDataset`,
reusing the lite QwenBPE tokenizer, `prompt._render_user_inst` template and
`prompt.apply_delay_pattern`. See MOSS-TTS/moss_tts_delay/processing_moss_tts.py
for the reference implementation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .bpe import QwenBPE
from .model import (AUDIO_DELAY_SLOT_TOKEN_ID,
                    AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_PAD_CODE,
                    AUDIO_USER_SLOT_TOKEN_ID, N_VQ, PAD_TOKEN_ID)
from .normalizer import normalize_tts_text
from .prompt import _render_user_inst, apply_delay_pattern

__all__ = ["MossTTSTrainDataset", "load_jsonl", "dump_jsonl",
           "build_computing_loss_ids", "build_generation_prompt_ids"]

AUDIO_PLACEHOLDER = "<|audio|>"
USER_SLOT_TOKEN = "<|audio_user_slot|>"
GEN_SLOT_TOKEN = "<|audio_assistant_gen_slot|>"
DELAY_SLOT_TOKEN = "<|audio_assistant_delay_slot|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"

USER_MESSAGE_KEYS = ("text", "instruction", "tokens", "quality",
                     "sound_event", "ambient_sound", "language")

_TOKEN_IDS_CACHE: dict[int, tuple[int, int]] = {}


def _audio_boundary_ids(tokenizer) -> tuple[int, int]:
    """Resolve <|audio_start|>/<|audio_end|> ids from the tokenizer itself.

    v1 (ModelScope) and v2 (HF local-transformer) checkpoints assign different
    ids to these tokens (151652/151653 vs 151669/151670), so the constants in
    model.py only describe the v1 generation.
    """
    key = id(tokenizer)
    cached = _TOKEN_IDS_CACHE.get(key)
    if cached is None:
        cached = (tokenizer.encode(AUDIO_START_TOKEN)[0],
                  tokenizer.encode(AUDIO_END_TOKEN)[0])
        _TOKEN_IDS_CACHE[key] = cached
    return cached


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(records: Iterable[dict[str, Any]], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Message building (UserMessage.__post_init__ / build_user_message)
# ---------------------------------------------------------------------------

def build_user_message(
    text: Optional[str] = None,
    reference: Optional[list[Optional[torch.Tensor]]] = None,
    instruction: Optional[str] = None,
    tokens: Optional[int] = None,
    quality: Optional[str] = None,
    sound_event: Optional[str] = None,
    ambient_sound: Optional[str] = None,
    language: Optional[str] = None,
    normalizer=normalize_tts_text,
) -> dict:
    """`MossTTSDelayProcessor.build_user_message` -> {"role","content","audio_codes_list"}."""
    if text is not None:
        text = normalizer(text)
    if reference is None:
        reference_str = "None"
        codes: list[torch.Tensor] = []
    else:
        blocks = []
        codes = []
        for speaker_idx, speaker_reference in enumerate(reference):
            if speaker_reference is not None:
                blocks.append(f"[S{speaker_idx + 1}]:\n{AUDIO_PLACEHOLDER}")
                codes.append(speaker_reference)
        reference_str = "\n".join(blocks)
    content = _render_user_inst(
        text=text, reference=reference_str, instruction=instruction, tokens=tokens,
        quality=quality, sound_event=sound_event, ambient_sound=ambient_sound,
        language=language)
    return {"role": "user", "content": content, "audio_codes_list": codes}


def build_assistant_message(audio_codes_list, content: str = AUDIO_PLACEHOLDER) -> dict:
    return {"role": "assistant", "content": content,
            "audio_codes_list": list(audio_codes_list)}


# ---------------------------------------------------------------------------
# Unified codes (`_get_unified_codes`)
# ---------------------------------------------------------------------------

def _replace_audio_placeholders_local(content: str, lengths: Sequence[int],
                                      slot_token: str) -> str:
    """v2 (local-transformer): one slot token per frame, no delay expansion."""
    def build_audio_block(length: int) -> str:
        if length == 0:
            return f"{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}"
        return f"{AUDIO_START_TOKEN}{slot_token * length}{AUDIO_END_TOKEN}"

    lengths_iter = iter(lengths)
    return re.sub(re.escape(AUDIO_PLACEHOLDER),
                  lambda _m: build_audio_block(next(lengths_iter)), content)


def _replace_audio_placeholders(content: str, lengths: Sequence[int], n_vq: int,
                                gen_slot_token: str, delay_slot_token: str) -> str:
    num_placeholders = content.count(AUDIO_PLACEHOLDER)
    if num_placeholders != len(lengths):
        raise ValueError(
            f"Number of {AUDIO_PLACEHOLDER} ({num_placeholders}) does not match "
            f"lengths ({len(lengths)})")

    def build_audio_block(length: int) -> str:
        if length == 0:
            return f"{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}"
        step_tokens = gen_slot_token * length + (delay_slot_token * (n_vq - 1))
        return f"{AUDIO_START_TOKEN}{step_tokens}{AUDIO_END_TOKEN}"

    lengths_iter = iter(lengths)
    return re.sub(re.escape(AUDIO_PLACEHOLDER),
                  lambda _m: build_audio_block(next(lengths_iter)), content)


def get_unified_codes(role: str, content: str, audio_codes_list: list[torch.Tensor],
                      tokenizer, n_vq: int,
                      slot_ids: Optional[tuple[int, int]] = None) -> torch.Tensor:
    """One message -> unified codes [T, n_vq + 1] (text channel first).

    slot_ids: (user_slot_id, assistant_slot_id) from a v2 (local-transformer)
    config.json -> one slot position per frame, codes aligned directly (no
    delay-pattern expansion). None -> v1 delay layout with dedicated slot
    token strings.
    """
    if slot_ids is not None:
        user_slot_id, asst_slot_id = slot_ids
        slot_id = user_slot_id if role == "user" else asst_slot_id
        slot_token = tokenizer.id_to_added_token(slot_id)
        if slot_token is None:
            raise ValueError(
                f"slot id {slot_id} is not an added token in this tokenizer")
        content = _replace_audio_placeholders_local(
            content, [int(codes.shape[0]) for codes in audio_codes_list],
            slot_token)
    else:
        if role == "user":
            gen_token = delay_token = USER_SLOT_TOKEN
        else:
            gen_token = GEN_SLOT_TOKEN
            delay_token = DELAY_SLOT_TOKEN
        content = _replace_audio_placeholders(
            content, [int(codes.shape[0]) for codes in audio_codes_list],
            n_vq, gen_token, delay_token)
    text_codes = torch.tensor(tokenizer.encode(content), dtype=torch.long)

    audio_start_id, audio_end_id = _audio_boundary_ids(tokenizer)
    audio_start_indices = torch.where(text_codes == audio_start_id)[0]
    audio_end_indices = torch.where(text_codes == audio_end_id)[0]
    if (len(audio_start_indices) != len(audio_codes_list)
            or len(audio_end_indices) != len(audio_codes_list)):
        raise ValueError("Audio placeholders do not match the provided audio codes list.")

    if not audio_codes_list:
        delay_audio_codes = torch.full((len(text_codes), n_vq), AUDIO_PAD_CODE,
                                       dtype=torch.long)
    else:
        segments: list[torch.Tensor] = []
        prefix_idx = 0
        for audio_start_idx, audio_end_idx, audio_codes in zip(
                audio_start_indices, audio_end_indices, audio_codes_list):
            if slot_ids is not None:
                # v2 local-transformer: frame codes align 1:1 with the slot rows.
                aligned_codes = audio_codes.to(torch.long)
            else:
                aligned_codes = apply_delay_pattern(audio_codes, AUDIO_PAD_CODE)
            pad_codes = torch.full((int(audio_start_idx) - prefix_idx + 1, n_vq),
                                   AUDIO_PAD_CODE, dtype=torch.long)
            segments.extend([pad_codes, aligned_codes])
            prefix_idx = int(audio_end_idx)
        pad_codes = torch.full((len(text_codes) - prefix_idx, n_vq), AUDIO_PAD_CODE,
                               dtype=torch.long)
        segments.append(pad_codes)
        delay_audio_codes = torch.cat(segments)

    if text_codes.shape[0] != delay_audio_codes.shape[0]:
        text_codes = text_codes[: delay_audio_codes.shape[0]]

    return torch.cat([text_codes.unsqueeze(1), delay_audio_codes], dim=1)


def _message_to_unified(message: dict, tokenizer, n_vq: int,
                        add_generation_prompt: bool,
                        slot_ids: Optional[tuple[int, int]] = None) -> torch.Tensor:
    """apply_chat_template for a single message, then `_get_unified_codes`."""
    role = message["role"]
    content = (f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        content += "<|im_start|>assistant\n"
    return get_unified_codes(role, content, message["audio_codes_list"], tokenizer,
                             n_vq, slot_ids=slot_ids)


def _user_message_for_record(record: dict, target_n_vq: int) -> dict:
    user_kwargs = _user_message_from_record(record)
    user_kwargs["reference"] = resolve_reference_codes(record, target_n_vq)
    return build_user_message(**user_kwargs)


def build_generation_prompt_ids(record: dict, tokenizer,
                                n_vq: Optional[int] = None,
                                user_message: Optional[dict] = None,
                                 slot_ids: Optional[tuple[int, int]] = None) -> torch.Tensor:
    """User message (+ generation prompt) -> [T, n_vq+1]; T is the loss prompt length."""
    n_vq = n_vq or _record_n_vq(record)
    user = user_message if user_message is not None else _user_message_for_record(record, n_vq)
    return _message_to_unified(user, tokenizer, n_vq, add_generation_prompt=True,
                               slot_ids=slot_ids)


def build_computing_loss_ids(record: dict, tokenizer, n_vq: Optional[int] = None,
                              user_message: Optional[dict] = None,
                              slot_ids: Optional[tuple[int, int]] = None) -> torch.Tensor:
    """[user, assistant] conversation in computing_loss mode -> [T, n_vq+1]."""
    n_vq = n_vq or _record_n_vq(record)
    user = user_message if user_message is not None else _user_message_for_record(record, n_vq)
    assistant = build_assistant_message(
        [normalize_audio_codes(record["audio_codes"], "audio_codes")])
    unified = torch.cat([
        _message_to_unified(user, tokenizer, n_vq, add_generation_prompt=False,
                            slot_ids=slot_ids),
        _message_to_unified(assistant, tokenizer, n_vq, add_generation_prompt=False,
                            slot_ids=slot_ids),
    ])
    return unified


# ---------------------------------------------------------------------------
# Record normalization (`normalize_audio_codes` / `_resolve_reference_codes`)
# ---------------------------------------------------------------------------

def normalize_audio_codes(value: Any, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape (T, n_vq), "
                         f"got {tuple(tensor.shape)}.")
    return tensor.cpu().contiguous()


def normalize_audio_code_list(value: Any, field_name: str,
                              allow_none: bool = False
                              ) -> Optional[list[Optional[torch.Tensor]]]:
    if value in (None, "", []):
        return None
    if torch.is_tensor(value):
        return [normalize_audio_codes(value, field_name)]
    if isinstance(value, list) and value:
        if allow_none and any(item is None for item in value):
            return [None if item is None
                    else normalize_audio_codes(item, f"{field_name}[{i}]")
                    for i, item in enumerate(value)]
        first = value[0]
        if torch.is_tensor(first):
            return [normalize_audio_codes(item, f"{field_name}[{i}]")
                    for i, item in enumerate(value)]
        if isinstance(first, list):
            if first and isinstance(first[0], list):
                return [normalize_audio_codes(item, f"{field_name}[{i}]")
                        for i, item in enumerate(value)]
            return [normalize_audio_codes(value, field_name)]
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


def resolve_reference_codes(record: dict, target_n_vq: int
                            ) -> Optional[list[Optional[torch.Tensor]]]:
    for code_field in ("reference_audio_codes", "ref_audio_codes"):
        if record.get(code_field) is not None:
            codes_list = normalize_audio_code_list(
                record[code_field], code_field,
                allow_none=(code_field == "reference_audio_codes"))
            for codes in codes_list or []:
                if codes is not None and codes.shape[1] != target_n_vq:
                    raise ValueError(f"`{code_field}` n_vq={codes.shape[1]} does not "
                                     f"match target n_vq={target_n_vq}.")
            return codes_list
    for path_field in ("reference", "reference_audio", "ref_audio"):
        if record.get(path_field) not in (None, "", []):
            raise ValueError(
                f"Record has audio path field `{path_field}` but no precomputed "
                f"*_codes field. Run moss_tts_lite.prepare_data first.")
    return None


def _record_n_vq(record: dict) -> int:
    target = normalize_audio_codes(record["audio_codes"], "audio_codes")
    return int(target.shape[1])


def _user_message_from_record(record: dict) -> dict:
    user_kwargs: dict[str, Any] = {}
    for key in USER_MESSAGE_KEYS:
        if record.get(key) is not None:
            user_kwargs[key] = record[key]
    return user_kwargs


# ---------------------------------------------------------------------------
# Dataset + collate (MossTTSSFTDataset._pack_record / collate_fn)
# ---------------------------------------------------------------------------

def pad_batch(input_ids_list: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    """`MossTTSDelayProcessor._pad`: left-pad, audio pad code + text PAD_TOKEN_ID."""
    max_len = int(max(t.shape[0] for t in input_ids_list))
    n_channels = input_ids_list[0].shape[1]
    batch = len(input_ids_list)
    padded = torch.full((batch, max_len, n_channels), AUDIO_PAD_CODE, dtype=torch.long)
    lengths = torch.tensor([t.shape[0] for t in input_ids_list])
    for b, item in enumerate(input_ids_list):
        padded[b, max_len - int(item.shape[0]):] = item
    other_channel_mask = (max_len - lengths).unsqueeze(1) > torch.arange(max_len).unsqueeze(0)
    padded[..., 0][other_channel_mask] = PAD_TOKEN_ID
    attention_mask = ~other_channel_mask
    return {"input_ids": padded, "attention_mask": attention_mask}


class MossTTSTrainDataset(Dataset):
    """Records (official prepare_data JSONL format) -> teacher-forcing batches."""

    def __init__(self, records: Sequence[dict[str, Any]], tokenizer,
                 n_vq: Optional[int] = None,
                 slot_ids: Optional[tuple[int, int]] = None):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.n_vq = n_vq
        self.slot_ids = slot_ids

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.pack_record(self.records[index])

    def pack_record(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "audio_codes" not in record:
            raise ValueError("Each record must contain `audio_codes`. "
                             "Run moss_tts_lite.prepare_data first.")
        target_n_vq = _record_n_vq(record)
        if self.n_vq is not None and target_n_vq != self.n_vq:
            raise ValueError(f"Expected n_vq={self.n_vq}, but got {target_n_vq}.")

        reference_codes = resolve_reference_codes(record, target_n_vq)
        user_kwargs = _user_message_from_record(record)
        user_kwargs["reference"] = reference_codes

        # prompt length: user message rendered with the generation prompt, exactly
        # like the official dataset (processor mode="generation").
        prompt_user = build_user_message(**user_kwargs)
        prompt_ids = _message_to_unified(prompt_user, self.tokenizer, target_n_vq,
                                         add_generation_prompt=True,
                                         slot_ids=self.slot_ids)
        prompt_length = int(prompt_ids.shape[0])

        full_input_ids = build_computing_loss_ids(
            record, self.tokenizer, target_n_vq, user_message=prompt_user,
            slot_ids=self.slot_ids)
        if prompt_length >= full_input_ids.shape[0]:
            raise ValueError("Prompt length must be shorter than the packed "
                             "teacher-forcing sequence.")

        loss_mask = torch.zeros(full_input_ids.shape[0] - 1, dtype=torch.bool)
        loss_mask[prompt_length - 1:] = True
        return {"input_ids": full_input_ids, "loss_mask": loss_mask}

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        padded = pad_batch([item["input_ids"] for item in batch])
        full_input_ids = padded["input_ids"].to(torch.long)
        full_attention_mask = padded["attention_mask"].bool()

        max_mask_len = max(item["loss_mask"].shape[0] for item in batch)
        loss_masks = torch.zeros(len(batch), max_mask_len, dtype=torch.bool)
        for b, item in enumerate(batch):
            n = item["loss_mask"].shape[0]
            loss_masks[b, max_mask_len - n:] = item["loss_mask"]

        labels = full_input_ids[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_masks.unsqueeze(-1), -100)
        labels = labels.masked_fill(~full_attention_mask[:, 1:].unsqueeze(-1), -100)
        # The audio pad code is a structural placeholder from the delay pattern,
        # not a trainable target.
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == AUDIO_PAD_CODE, -100)

        return {
            "input_ids": full_input_ids[:, :-1, :].contiguous(),
            "attention_mask": full_attention_mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
        }
