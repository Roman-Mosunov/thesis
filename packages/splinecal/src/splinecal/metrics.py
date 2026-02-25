from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def brier_score(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute the binary Brier score."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_prob_arr.ndim != 1:
        raise ValueError("y_prob must be a 1D array of positive-class probabilities.")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")
    if np.any(~np.isfinite(y_true_arr)) or np.any(~np.isfinite(y_prob_arr)):
        raise ValueError("y_true and y_prob must be finite.")
    if np.any((y_true_arr != 0.0) & (y_true_arr != 1.0)):
        raise ValueError("y_true must contain only binary labels {0, 1}.")
    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("y_prob must contain probabilities in [0, 1].")

    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def expected_calibration_error(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_prob_arr.ndim != 1:
        raise ValueError("y_prob must be a 1D array of positive-class probabilities.")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)

    ece = 0.0
    n = y_true_arr.shape[0]
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = y_true_arr[mask].mean()
        conf = y_prob_arr[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
