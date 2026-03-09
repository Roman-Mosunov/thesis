from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_binary_prob_inputs(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_prob_arr.ndim != 1:
        raise ValueError("y_prob must be a 1D array of positive-class probabilities.")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_prob must contain at least one sample.")
    if np.any(~np.isfinite(y_true_arr)) or np.any(~np.isfinite(y_prob_arr)):
        raise ValueError("y_true and y_prob must be finite.")
    if np.any((y_true_arr != 0.0) & (y_true_arr != 1.0)):
        raise ValueError("y_true must contain only binary labels {0, 1}.")
    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("y_prob must contain probabilities in [0, 1].")

    return y_true_arr, y_prob_arr


def brier_score(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute the binary Brier score."""
    y_true_arr, y_prob_arr = _validate_binary_prob_inputs(y_true, y_prob)

    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def brier_calibration_refinement_loss(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int | None = None,
) -> tuple[float, float]:
    """Estimate Brier-score calibration and refinement losses.

    This uses the decomposition from Kull and Flach (2015):
    CL = E[d(S, C)] and RL = E[d(C, Y)] with d(a, b) = (a - b)^2.

    By default (``n_bins=None``), groups are formed by identical output scores.
    If ``n_bins`` is set, scores are grouped by uniform bins on [0, 1] as an
    approximation for continuous scores.
    """
    y_true_arr, y_prob_arr = _validate_binary_prob_inputs(y_true, y_prob)

    if n_bins is None:
        _, group_ids = np.unique(y_prob_arr, return_inverse=True)
        n_groups = int(group_ids.max()) + 1
    else:
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1 when provided.")
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        group_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
        n_groups = n_bins

    calibration_loss = 0.0
    refinement_loss = 0.0
    n = float(y_true_arr.shape[0])

    for group in range(n_groups):
        mask = group_ids == group
        if not np.any(mask):
            continue
        s_group = float(np.mean(y_prob_arr[mask]))
        c_group = float(np.mean(y_true_arr[mask]))
        weight = float(np.sum(mask)) / n
        calibration_loss += weight * (s_group - c_group) ** 2
        refinement_loss += weight * float(np.mean((y_true_arr[mask] - c_group) ** 2))

    return float(calibration_loss), float(refinement_loss)


def log_loss_calibration_refinement_loss(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int | None = None,
    eps: float = 1e-15,
) -> tuple[float, float]:
    """Estimate log-loss calibration and refinement losses.

    This uses the decomposition from Kull and Flach (2015):
    CL = E[d(S, C)] and RL = E[d(C, Y)] with
    d(p, q) = KL(q || p), i.e., the log-loss divergence.

    By default (``n_bins=None``), groups are formed by identical output scores.
    If ``n_bins`` is set, scores are grouped by uniform bins on [0, 1] as an
    approximation for continuous scores.
    """
    y_true_arr, y_prob_arr = _validate_binary_prob_inputs(y_true, y_prob)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5).")

    probs = np.clip(y_prob_arr, eps, 1.0 - eps)

    if n_bins is None:
        _, group_ids = np.unique(probs, return_inverse=True)
        n_groups = int(group_ids.max()) + 1
    else:
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1 when provided.")
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        group_ids = np.digitize(probs, bins[1:-1], right=True)
        n_groups = n_bins

    def _bernoulli_kl(q: float, p: float) -> float:
        left = 0.0 if q <= 0.0 else q * float(np.log(q / p))
        right = 0.0 if q >= 1.0 else (1.0 - q) * float(np.log((1.0 - q) / (1.0 - p)))
        return left + right

    def _bernoulli_entropy(q: float) -> float:
        left = 0.0 if q <= 0.0 else -q * float(np.log(q))
        right = 0.0 if q >= 1.0 else -(1.0 - q) * float(np.log(1.0 - q))
        return left + right

    calibration_loss = 0.0
    refinement_loss = 0.0
    n = float(y_true_arr.shape[0])

    for group in range(n_groups):
        mask = group_ids == group
        if not np.any(mask):
            continue
        s_group = float(np.mean(probs[mask]))
        c_group = float(np.mean(y_true_arr[mask]))
        weight = float(np.sum(mask)) / n
        calibration_loss += weight * _bernoulli_kl(c_group, s_group)
        refinement_loss += weight * _bernoulli_entropy(c_group)

    return float(calibration_loss), float(refinement_loss)


def expected_calibration_error(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification."""
    y_true_float, y_prob_arr = _validate_binary_prob_inputs(y_true, y_prob)
    y_true_arr = y_true_float.astype(int)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

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
