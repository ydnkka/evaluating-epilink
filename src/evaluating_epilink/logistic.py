from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def build_training_indices(y: np.ndarray, train_size: int, rng_seed: int) -> np.ndarray:
    """Sample a train subset that keeps at least one example from each class when possible."""

    classes = np.unique(y)
    if len(classes) < 2:
        return np.array([], dtype=int)

    train_size = int(min(max(train_size, len(classes)), len(y)))
    rng = np.random.default_rng(rng_seed)
    selected: list[int] = []
    remaining: list[int] = []

    for class_label in classes:
        class_indices = np.flatnonzero(y == class_label)
        chosen = int(rng.choice(class_indices))
        selected.append(chosen)
        remaining.extend(int(index) for index in class_indices if index != chosen)

    extra_needed = train_size - len(selected)
    if extra_needed > 0:
        extra_pool = np.asarray(remaining, dtype=int)
        if extra_needed >= len(extra_pool):
            extra = extra_pool
        else:
            extra = rng.choice(extra_pool, size=extra_needed, replace=False)
        selected.extend(int(index) for index in np.asarray(extra, dtype=int))

    return np.asarray(sorted(selected), dtype=int)


def predict_logistic_scores(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    *,
    training_fraction: float,
    rng_seed: int,
) -> np.ndarray:
    """Fit logistic regression when feasible, otherwise return a constant prior score."""

    if len(y) == 0:
        return np.full(0, np.nan, dtype=float)

    prevalence = float(y.mean())
    if len(np.unique(y)) < 2:
        return np.full(len(y), prevalence, dtype=float)

    classifier = LogisticRegression(solver="lbfgs", max_iter=200)
    if float(training_fraction) >= 1.0:
        classifier.fit(feature_matrix, y)
        return classifier.predict_proba(feature_matrix)[:, 1]

    requested_train_size = int(np.ceil(len(y) * float(training_fraction)))
    train_indices = build_training_indices(y, requested_train_size, rng_seed)
    if len(train_indices) == 0 or len(np.unique(y[train_indices])) < 2:
        return np.full(len(y), prevalence, dtype=float)

    classifier.fit(feature_matrix[train_indices], y[train_indices])
    return classifier.predict_proba(feature_matrix)[:, 1]
