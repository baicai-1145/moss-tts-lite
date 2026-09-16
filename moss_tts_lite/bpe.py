"""Pure-Python Qwen byte-level BPE (stdlib only)."""

from __future__ import annotations

import json
import unicodedata

__all__ = ["QwenBPE"]

def _is_space(ch: str) -> bool:
    r"""Unicode White_Space property (what Rust regex `\s` matches)."""
    o = ord(ch)
    return (0x09 <= o <= 0x0D or o == 0x20 or o == 0x85 or o == 0xA0
            or o == 0x1680 or 0x2000 <= o <= 0x200A or o == 0x2028
            or o == 0x2029 or o == 0x202F or o == 0x205F or o == 0x3000)

def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "L"

def _is_number(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "N"

def _is_other(ch: str) -> bool:
    r"""[^[\r\n]\p{L}\p{N}]."""
    return (ch not in "\r\n" and not _is_letter(ch) and not _is_number(ch)
            and not _is_space(ch))

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte<->unicode table (identical to HF ByteLevel)."""
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

class QwenBPE:
    """Byte-level BPE with Qwen pre-tokenization + added-token handling."""

    def __init__(self, vocab_json: str, merges_txt: str,
                 added_tokens_json: str | None = None):
        with open(vocab_json, encoding="utf-8") as f:
            self._vocab: dict[str, int] = json.load(f)

        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {c: b for b, c in self._byte_encoder.items()}

        self._ranks: dict[tuple[str, str], int] = {}
        with open(merges_txt, encoding="utf-8") as f:
            rank = 0
            for line in f:
                line = line.rstrip("\n")
                if not line or (rank == 0 and line.startswith("#")):
                    continue
                a, b = line.split(" ", 1) if " " in line else (line, "")
                if not b:
                    continue
                self._ranks[(a, b)] = rank
                rank += 1

        self._id_to_added: dict[int, str] = {}
        self._added_by_first: dict[str, list[str]] = {}
        if added_tokens_json is not None:
            with open(added_tokens_json, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "added_tokens" in data:
                added = {t["content"]: t["id"] for t in data["added_tokens"]}
            else:
                added = data
            for tok, tid in added.items():
                self._id_to_added[tid] = tok
                self._added_by_first.setdefault(tok[0], []).append(tok)
            for toks in self._added_by_first.values():
                toks.sort(key=len, reverse=True)

        self._id_to_token: dict[int, str] = {
            i: t for t, i in self._vocab.items()}
        self._cache: dict[str, list[int]] = {}

    def id_to_added_token(self, token_id: int) -> str | None:
        """Added-token string for an id (None when it is not an added token)."""
        return self._id_to_added.get(int(token_id))

    def _match_added(self, text: str, i: int) -> str | None:
        for tok in self._added_by_first.get(text[i], ()):
            if text.startswith(tok, i):
                return tok
        return None

    def _split_added(self, text: str) -> list[tuple[bool, str]]:
        segs: list[tuple[bool, str]] = []
        i, n, start = 0, len(text), 0
        while i < n:
            tok = self._match_added(text, i)
            if tok is not None:
                if i > start:
                    segs.append((False, text[start:i]))
                segs.append((True, tok))
                i += len(tok)
                start = i
            else:
                i += 1
        if start < n:
            segs.append((False, text[start:]))
        return segs

    _CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")

    def _match_contraction(self, text: str, i: int) -> int | None:
        """(?i:"""
        n = len(text)
        for suf in self._CONTRACTIONS:
            e = i + 1 + len(suf)
            if e <= n and text[i + 1:e].lower() == suf:
                return e
        return None

    def _pretokenize(self, text: str) -> list[str]:
        pieces: list[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]

            if ch == "'":
                e = self._match_contraction(text, i)
                if e is not None:
                    pieces.append(text[i:e])
                    i = e
                    continue
            is_l, is_n, is_s = _is_letter(ch), _is_number(ch), _is_space(ch)

            end = None
            if ch not in "\r\n" and not is_l and not is_n:
                j = i + 1
                if j < n and _is_letter(text[j]):
                    j += 1
                    while j < n and _is_letter(text[j]):
                        j += 1
                    end = j
            if end is None and is_l:
                j = i + 1
                while j < n and _is_letter(text[j]):
                    j += 1
                end = j
            if end is not None:
                pieces.append(text[i:end])
                i = end
                continue

            if is_n:
                pieces.append(ch)
                i += 1
                continue

            if ch == " " and i + 1 < n and _is_other(text[i + 1]):
                j = i + 1
                while j < n and _is_other(text[j]):
                    j += 1
                while j < n and text[j] in "\r\n":
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue
            if _is_other(ch):
                j = i + 1
                while j < n and _is_other(text[j]):
                    j += 1
                while j < n and text[j] in "\r\n":
                    j += 1
                pieces.append(text[i:j])
                i = j
                continue

            if is_s:
                j = i
                while j < n and _is_space(text[j]):
                    j += 1

                k = None
                for t in range(j - 1, i - 1, -1):
                    if text[t] in "\r\n":
                        k = t
                        break
                if k is not None:
                    pieces.append(text[i:k + 1])
                    i = k + 1
                    continue

                if j == n:
                    pieces.append(text[i:j])
                    i = j
                    continue
                if j - i >= 2:
                    pieces.append(text[i:j - 1])
                    i = j - 1
                    continue

                pieces.append(text[i:j])
                i = j
                continue

            pieces.append(ch)
            i += 1
        return pieces

    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        word = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
        symbols = list(word)
        ranks = self._ranks
        while len(symbols) > 1:
            best_rank = None
            best_pair = None
            prev = symbols[0]
            for cur in symbols[1:]:
                r = ranks.get((prev, cur))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = (prev, cur)
                prev = cur
            if best_pair is None:
                break
            a, b = best_pair
            merged: list[str] = []
            k = 0
            m = len(symbols)
            while k < m:
                if k < m - 1 and symbols[k] == a and symbols[k + 1] == b:
                    merged.append(a + b)
                    k += 2
                else:
                    merged.append(symbols[k])
                    k += 1
            symbols = merged
        vocab = self._vocab
        try:
            ids = [vocab[s] for s in symbols]
        except KeyError as e:
            raise KeyError(f"piece {piece!r} produced symbol {e} not in vocab") from e
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for is_added, seg in self._split_added(text):
            if is_added:
                ids.append(self._vocab_added_id(seg))
                continue
            seg = unicodedata.normalize("NFC", seg)
            for piece in self._pretokenize(seg):
                ids.extend(self._encode_piece(piece))
        return ids

    def _vocab_added_id(self, tok: str) -> int:
        for tid, t in self._id_to_added.items():
            if t == tok:
                return tid
        raise KeyError(f"added token {tok!r} has no id")

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            i = int(i)
            tok = self._id_to_added.get(i)
            if tok is not None:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(tok)
                continue
            tok = self._id_to_token.get(i)
            if tok is None:
                raise ValueError(f"unknown token id {i}")
            for ch in tok:
                b = self._byte_decoder.get(ch)
                if b is None:
                    buf.extend(ch.encode("utf-8"))
                else:
                    buf.append(b)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)
