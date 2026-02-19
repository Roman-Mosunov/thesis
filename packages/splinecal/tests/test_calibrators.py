import numpy as np
import pytest
from splinecal.calibrators import HaarMonotoneRidgeCalibrator, SplineBinaryCalibrator


def test_calibrator_fit_predict_proba_shape() -> None:
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.01, 0.99, size=200)
    y = (rng.uniform(size=200) < raw).astype(int)

    cal = SplineBinaryCalibrator(n_knots=4)
    cal.fit(raw, y)
    proba = cal.predict_proba(raw)

    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_calibrator_rejects_non_binary_target() -> None:
    x = np.array([0.2, 0.8, 0.6])
    y = np.array([0, 1, 2])

    cal = SplineBinaryCalibrator()

    with pytest.raises(ValueError, match="binary"):
        cal.fit(x, y)


def test_haar_monotone_calibrator_predict_proba_is_monotone() -> None:
    rng = np.random.default_rng(11)
    scores = np.sort(rng.uniform(0.0, 1.0, size=400))
    y = (rng.uniform(size=400) < scores).astype(int)

    cal = HaarMonotoneRidgeCalibrator(j_max=5, lam=1e-2)
    cal.fit(scores, y)

    grid = np.linspace(0.0, 1.0, 1001)
    proba = cal.predict_proba(grid)[:, 1]

    assert proba.shape == (1001,)
    assert np.all(np.diff(proba) >= -1e-8)


def test_haar_monotone_calibrator_supports_non_numeric_labels() -> None:
    rng = np.random.default_rng(21)
    scores = rng.uniform(0.0, 1.0, size=300)
    y = np.where(rng.uniform(size=300) < scores, "yes", "no")

    cal = HaarMonotoneRidgeCalibrator(j_max=4, lam=1e-2)
    cal.fit(scores, y)
    pred = cal.predict(scores)

    assert set(np.unique(pred)) <= {"no", "yes"}
