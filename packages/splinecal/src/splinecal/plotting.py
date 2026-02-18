from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reliability_points(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-bin confidence and accuracy points for reliability diagrams."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)

    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            conf[b] = y_prob_arr[mask].mean()
            acc[b] = y_true_arr[mask].mean()

    return mids, np.column_stack((conf, acc))
