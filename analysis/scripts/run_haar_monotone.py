"""Haar monotone ridge demo.

Run from repo root:
    uv run python analysis/scripts/run_haar_monotone.py
"""

from __future__ import annotations

import numpy as np
from splinecal import HaarMonotoneRidgeCalibrator, expected_calibration_error


def main() -> None:
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.0, 1.0, size=1200)

    # Synthetic nonlinear but monotone probability surface.
    true_prob = 0.05 + 0.9 * (scores**1.8)
    y = (rng.uniform(size=scores.size) < true_prob).astype(int)

    cal = HaarMonotoneRidgeCalibrator(j_max=6, lam=1e-2)
    cal.fit(scores, y)

    calibrated = cal.predict_proba(scores)[:, 1]
    ece = expected_calibration_error(y, calibrated)
    print(f"ECE: {ece:.4f}")


if __name__ == "__main__":
    main()
