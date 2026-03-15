from __future__ import annotations

import numpy as np

from evaluating_epilink.logistic import build_training_indices, predict_logistic_scores


def test_build_training_indices_keeps_both_classes_for_rare_positive_case() -> None:
    y = np.array([0] * 103 + [1] * 2)

    indices = build_training_indices(y, train_size=11, rng_seed=12345)

    assert len(indices) == 11
    assert set(y[indices]) == {0, 1}


def test_predict_logistic_scores_handles_rare_positive_fractional_training() -> None:
    feature_matrix = np.column_stack(
        [
            np.linspace(0.0, 20.0, 105),
            np.concatenate([np.zeros(103), np.ones(2)]),
        ]
    )
    y = np.array([0] * 103 + [1] * 2)

    scores = predict_logistic_scores(
        feature_matrix,
        y,
        training_fraction=0.1,
        rng_seed=12345,
    )

    assert scores.shape == (105,)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_predict_logistic_scores_falls_back_for_single_class_data() -> None:
    feature_matrix = np.column_stack([np.arange(8, dtype=float), np.arange(8, dtype=float)])
    y = np.zeros(8, dtype=int)

    scores = predict_logistic_scores(
        feature_matrix,
        y,
        training_fraction=0.1,
        rng_seed=12345,
    )

    assert np.array_equal(scores, np.zeros(8, dtype=float))
