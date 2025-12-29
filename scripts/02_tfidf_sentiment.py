from __future__ import annotations

from pathlib import Path

from nlp_learning.datasets import load_sample_sentiment_csv
from nlp_learning.models import train_logreg_text_classifier
from nlp_learning.text_cleaning import clean_text
from nlp_learning.vectorizers import make_tfidf_vectorizer


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_sentiment.csv"
    ds = load_sample_sentiment_csv(data_path)

    texts = [clean_text(t) for t in ds.text]
    vectorizer = make_tfidf_vectorizer()
    X = vectorizer.fit_transform(texts)

    result = train_logreg_text_classifier(X, ds.label)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(result["report"])


if __name__ == "__main__":
    main()
