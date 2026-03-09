import numpy as np
import pytest
from sklearn.base import clone
from splinecal.calibrators import (
    BetaBinaryCalibrator,
    HaarMonotoneRidgeCalibrator,
    IsotonicBinaryCalibrator,
    PlattBinaryCalibrator,
    SplineBinaryCalibrator,
)


@pytest.mark.parametrize(
    "est",
    [
        SplineBinaryCalibrator(n_knots=6, degree=2, c=0.5),
        PlattBinaryCalibrator(c=0.7),
        IsotonicBinaryCalibrator(out_of_bounds="clip"),
        BetaBinaryCalibrator(c=0.9),
        HaarMonotoneRidgeCalibrator(j_max=5, lam=1e-3, use_haar_norm=False, clip_probs=False),
    ],
)
def test_clone_round_trip(est) -> None:
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()


def test_fit_with_2d_inputs() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(size=(120, 2))
    y = (x[:, -1] > 0.5).astype(int)

    est = SplineBinaryCalibrator()
    est.fit(x, y)
    pred = est.predict(x)

    assert pred.shape == (120,)
    assert set(np.unique(pred)).issubset({0, 1})
