from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42


def train_logreg_text_classifier(
    X,
    y,
    *,
    config: TrainConfig | None = None,
    max_iter: int = 200,
) -> dict:
    """Train a Logistic Regression classifier on already-vectorized X."""

    cfg = config or TrainConfig()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return {
        "model": clf,
        "accuracy": float(acc),
        "report": classification_report(y_test, pred, digits=4),
        "X_test": X_test,
        "y_test": np.asarray(y_test),
        "pred": pred,
    }
