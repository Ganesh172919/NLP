from __future__ import annotations

import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?")


def simple_tokenize(text: str) -> list[str]:
    """Regex tokenization: words (optionally with apostrophes) + numbers."""

    return _TOKEN_RE.findall(text)


def ngrams(tokens: Iterable[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    toks = list(tokens)
    return [tuple(toks[i : i + n]) for i in range(0, max(0, len(toks) - n + 1))]
