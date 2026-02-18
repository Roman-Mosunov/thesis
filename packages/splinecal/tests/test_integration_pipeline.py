import numpy as np

from splinecal.metrics import expected_calibration_error


def test_expected_calibration_error_bounds() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.4])

    ece = expected_calibration_error(y_true, y_prob, n_bins=3)
    assert 0.0 <= ece <= 1.0
