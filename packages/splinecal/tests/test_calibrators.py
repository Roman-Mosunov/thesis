import numpy as np

from splinecal.calibrators import SplineBinaryCalibrator


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

    try:
        cal.fit(x, y)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "binary" in str(exc).lower()
