"""Starter analysis script.

Run from repo root:
    uv run python analysis/scripts/run_baseline.py
"""

from __future__ import annotations

import numpy as np

from splinecal import SplineBinaryCalibrator, expected_calibration_error


def main() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.01, 0.99, size=500)
    y = (rng.uniform(size=500) < raw).astype(int)

    calibrator = SplineBinaryCalibrator(n_knots=5)
    calibrator.fit(raw, y)

    calibrated = calibrator.predict_proba(raw)[:, 1]
    ece = expected_calibration_error(y, calibrated)
    print(f"ECE: {ece:.4f}")


if __name__ == "__main__":
    main()
