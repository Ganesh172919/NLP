from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    max_features: int | None = 20_000


def make_tfidf_vectorizer(config: TfidfConfig | None = None) -> TfidfVectorizer:
    cfg = config or TfidfConfig()
    return TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=None,
    )
