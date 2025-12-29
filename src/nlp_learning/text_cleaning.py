from __future__ import annotations

import re
from dataclasses import dataclass

import regex  # type: ignore


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CleanTextConfig:
    lowercase: bool = True
    strip_accents: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    keep_apostrophes: bool = True
    normalize_whitespace: bool = True


def clean_text(text: str, config: CleanTextConfig | None = None) -> str:
    """Basic normalization for English-like text.

    Intentionally simple and readable for learning.
    """

    cfg = config or CleanTextConfig()
    s = text

    if cfg.lowercase:
        s = s.lower()

    if cfg.remove_urls:
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)

    if cfg.remove_emails:
        s = re.sub(r"\b\S+@\S+\.\S+\b", " ", s)

    if cfg.strip_accents:
        # Unicode normalization via regex module
        s = regex.sub(r"\p{Mn}+", "", regex.normalize("NFKD", s))

    # Keep basic punctuation but remove control chars.
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)

    if not cfg.keep_apostrophes:
        s = s.replace("'", " ")

    if cfg.normalize_whitespace:
        s = _WHITESPACE_RE.sub(" ", s).strip()

    return s
