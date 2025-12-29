from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from nlp_learning.text_cleaning import clean_text


def top_words(model, feature_names: list[str], n_top_words: int = 10) -> list[list[str]]:
    topics: list[list[str]] = []
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[::-1][:n_top_words]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_news.csv"
    df = pd.read_csv(data_path)

    texts = [clean_text(t) for t in df["text"].astype(str).tolist()]

    vec = CountVectorizer(max_df=0.95, min_df=1, stop_words="english")
    X = vec.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)

    vocab = vec.get_feature_names_out().tolist()
    for i, words in enumerate(top_words(lda, vocab, 10)):
        print(f"Topic {i}: {', '.join(words)}")


if __name__ == "__main__":
    main()
