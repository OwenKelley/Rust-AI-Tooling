"""Python reference mirroring rtokenizers (not HuggingFace)."""

from __future__ import annotations

from typing import Any, Callable


def _make_texts(n: int, seed: int) -> list[str]:
    words = ["ab", "abc", "unwanted", "hello", "world", "rust", "code"]
    state = seed | 1
    out = []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        a = words[state % len(words)]
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        b = words[state % len(words)]
        out.append(f"{a} {b}")
    return out


def _checksum_ids(ids: list[int]) -> float:
    return float(len(ids) + sum(ids))


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.unk_id = 0

    def fit(self, texts: list[str]) -> None:
        seen: set[str] = set()
        for t in texts:
            for tok in t.split():
                seen.add(tok)
        toks = sorted(seen)
        self.id_to_token = ["[UNK]"] + toks
        self.vocab = {t: i for i, t in enumerate(self.id_to_token)}
        self.unk_id = 0

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(tok, self.unk_id) for tok in text.split()]


class BpeTokenizer:
    def __init__(self, merges: list[tuple[str, str]]) -> None:
        self.merges = merges
        vocab_order = ["[UNK]"]
        seen = {"[UNK]"}
        for a, b in merges:
            for s in (a, b, a + b):
                if s not in seen:
                    seen.add(s)
                    vocab_order.append(s)
        self.vocab = {t: i for i, t in enumerate(vocab_order)}
        self.unk_id = 0

    def _apply(self, word: str) -> list[str]:
        symbols = list(word)
        for a, b in self.merges:
            i = 0
            while i + 1 < len(symbols):
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i] = a + b
                    del symbols[i + 1]
                else:
                    i += 1
        return symbols

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            for piece in self._apply(word):
                ids.append(self.vocab.get(piece, self.unk_id))
        return ids


class WordPieceTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.unk_id = self.vocab.get("[UNK]", 0)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            chars = list(word)
            start = 0
            ok = True
            word_toks: list[str] = []
            while start < len(chars):
                end = len(chars)
                cur = None
                while start < end:
                    substr = "".join(chars[start:end])
                    cand = f"##{substr}" if start > 0 else substr
                    if cand in self.vocab:
                        cur = cand
                        break
                    end -= 1
                if cur is None:
                    ok = False
                    break
                word_toks.append(cur)
                start = end
            if not ok:
                ids.append(self.unk_id)
            else:
                ids.extend(self.vocab[t] for t in word_toks)
        return ids


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    n = max(size, 8)
    texts = _make_texts(n, seed)

    def thunk():
        if op == "whitespace":
            tok = WhitespaceTokenizer()
            tok.fit(texts)
            return sum(_checksum_ids(tok.encode(t)) for t in texts)
        if op == "bpe":
            bpe = BpeTokenizer([("a", "b"), ("ab", "c")])
            return sum(_checksum_ids(bpe.encode(t)) for t in texts)
        if op == "wordpiece":
            wp = WordPieceTokenizer(
                [
                    "[UNK]",
                    "un",
                    "##want",
                    "##ed",
                    "want",
                    "hello",
                    "world",
                    "rust",
                    "code",
                    "ab",
                    "abc",
                ]
            )
            return sum(_checksum_ids(wp.encode(t)) for t in texts)
        raise ValueError(op)

    return thunk(), thunk


def checksum(value: Any) -> float:
    return float(value)
