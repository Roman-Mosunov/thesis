import numpy as np
import pytest
from sklearn.metrics import log_loss
from splinecal.metrics import (
    brier_calibration_refinement_loss,
    brier_score,
    expected_calibration_error,
    log_loss_calibration_refinement_loss,
)


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


def test_brier_decomposition_matches_brier_for_identical_score_groups() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.2, 0.8, 0.8])

    calibration_loss, refinement_loss = brier_calibration_refinement_loss(y_true, y_prob)
    total = brier_score(y_true, y_prob)

    assert np.isclose(calibration_loss, 0.09)
    assert np.isclose(refinement_loss, 0.25)
    assert np.isclose(calibration_loss + refinement_loss, total)


def test_brier_decomposition_rejects_invalid_bins() -> None:
    y_true = np.array([0, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.7])

    with pytest.raises(ValueError, match="at least 1"):
        brier_calibration_refinement_loss(y_true, y_prob, n_bins=0)


def test_log_loss_decomposition_known_values() -> None:
    y_true = np.array([1, 1, 1, 0, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3])

    calibration_loss, refinement_loss = log_loss_calibration_refinement_loss(y_true, y_prob)

    assert np.isclose(calibration_loss, 0.090, atol=1e-3)
    assert np.isclose(refinement_loss, 0.628, atol=1e-3)


def test_log_loss_decomposition_matches_log_loss_for_identical_score_groups() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.2])

    calibration_loss, refinement_loss = log_loss_calibration_refinement_loss(y_true, y_prob)
    total = float(log_loss(y_true, y_prob))

    assert np.isclose(calibration_loss + refinement_loss, total)
