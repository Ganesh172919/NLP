from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SentimentDataset:
    text: list[str]
    label: list[str]


def load_sample_sentiment_csv(path: str | Path) -> SentimentDataset:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    return SentimentDataset(text=df["text"].astype(str).tolist(), label=df["label"].astype(str).tolist())
