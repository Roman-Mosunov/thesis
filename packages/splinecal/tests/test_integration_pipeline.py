import numpy as np
import pytest
from splinecal.metrics import brier_score, expected_calibration_error


def test_expected_calibration_error_bounds() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.4])

    ece = expected_calibration_error(y_true, y_prob, n_bins=3)
    assert 0.0 <= ece <= 1.0


def test_brier_score_known_value() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.3])

    score = brier_score(y_true, y_prob)
    assert np.isclose(score, 0.075)


def test_brier_score_rejects_non_binary_targets() -> None:
    y_true = np.array([0, 2, 1])
    y_prob = np.array([0.2, 0.4, 0.8])

    with pytest.raises(ValueError, match="binary"):
        brier_score(y_true, y_prob)


def test_brier_score_rejects_out_of_range_probabilities() -> None:
    y_true = np.array([0, 1, 1])
    y_prob = np.array([0.1, 1.2, 0.9])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score(y_true, y_prob)
