import numpy as np
import pytest
from splinecal.haar import (
    build_basis,
    check_monotone,
    design_matrix,
    fit_monotone_ridge,
    predict_monotone_ridge,
    psi_tilde,
)


def test_psi_tilde_piecewise_values_and_tail_constant() -> None:
    j = 2
    k = 1
    h = 2.0 ** (-j)
    a = k * h
    m = a + 0.5 * h
    b = a + h
    amp = 2.0 ** (0.5 * j)

    x = np.array([a - 1e-6, a, a + h / 4.0, m, a + 3.0 * h / 4.0, b, b + 1e-6])
    y = psi_tilde(x, j, k)

    expected = np.array(
        [
            0.0,
            0.0,
            amp * 0.5 * (h / 4.0) ** 2,
            amp * (h * h) / 8.0,
            amp * (7.0 * h * h) / 32.0,
            amp * (h * h) / 4.0,
            amp * (h * h) / 4.0,
        ]
    )

    assert np.allclose(y, expected)


def test_psi_tilde_is_continuous_at_b() -> None:
    j = 3
    k = 5
    h = 2.0 ** (-j)
    b = k * h + h

    left = psi_tilde(np.array([b - 1e-12]), j, k)[0]
    right = psi_tilde(np.array([b + 1e-12]), j, k)[0]

    assert np.isclose(left, right, atol=1e-9)


def test_design_matrix_shape() -> None:
    scores = np.linspace(0.0, 1.0, 11)
    basis_idx = build_basis(4)
    mat = design_matrix(scores, basis_idx)

    assert mat.shape == (11, 31)


def test_fit_monotone_ridge_is_monotone_on_grid() -> None:
    rng = np.random.default_rng(12)
    scores = np.sort(rng.uniform(0.0, 1.0, size=600))
    y = (rng.uniform(size=600) < scores).astype(float)

    fit = fit_monotone_ridge(scores, y, j_max=5, lam=1e-2)
    pred = predict_monotone_ridge(np.linspace(0.0, 1.0, 1201), fit)

    assert fit.beta >= -1e-12
    assert np.min(fit.theta) >= -1e-10
    assert np.all(np.diff(pred) >= -1e-8)
    assert check_monotone(fit)


def test_fit_monotone_ridge_rejects_out_of_range_scores() -> None:
    scores = np.array([-0.1, 0.2, 0.6])
    y = np.array([0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_monotone_ridge(scores, y, j_max=2, lam=1e-2)
