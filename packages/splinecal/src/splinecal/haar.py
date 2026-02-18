from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear


def _haar_amplitude(j: int, *, use_haar_norm: bool) -> float:
    if not use_haar_norm:
        return 1.0
    return float(2.0 ** (0.5 * j))


def psi_tilde(
    x: NDArray[np.float64] | NDArray[np.int64] | float,
    j: int,
    k: int,
    *,
    use_haar_norm: bool = True,
) -> NDArray[np.float64]:
    """Double-integrated Haar wavelet basis element \tilde{psi}_{j,k}(x)."""
    if j < 0:
        raise ValueError("j must be nonnegative.")
    if k < 0 or k >= 2**j:
        raise ValueError("k must satisfy 0 <= k < 2**j.")

    x_arr = np.asarray(x, dtype=float)
    h = 2.0 ** (-j)
    a = k * h
    m = a + 0.5 * h
    b = a + h
    amp = _haar_amplitude(j, use_haar_norm=use_haar_norm)

    y = np.zeros_like(x_arr, dtype=float)

    mask = (x_arr >= a) & (x_arr < m)
    y[mask] = amp * 0.5 * (x_arr[mask] - a) ** 2

    mask = (x_arr >= m) & (x_arr < b)
    s = x_arr[mask] - m
    y[mask] = amp * ((h * h) / 8.0 + (h / 2.0) * s - 0.5 * s * s)

    mask = x_arr >= b
    y[mask] = amp * (h * h) / 4.0

    return y


def build_basis(j_max: int) -> list[tuple[int, int]]:
    """Return basis index list (j, k) for j=0..j_max and k=0..2^j-1."""
    if j_max < 0:
        raise ValueError("j_max must be nonnegative.")

    idx: list[tuple[int, int]] = []
    for j in range(j_max + 1):
        for k in range(2**j):
            idx.append((j, k))
    return idx


def design_matrix(
    scores: NDArray[np.float64] | NDArray[np.int64],
    basis_idx: list[tuple[int, int]],
    *,
    use_haar_norm: bool = True,
) -> NDArray[np.float64]:
    """Build B[i, ell] = psi_tilde(scores[i], j, k)."""
    scores_arr = np.asarray(scores, dtype=float).ravel()
    b_mat = np.empty((scores_arr.size, len(basis_idx)), dtype=float)

    for ell, (j, k) in enumerate(basis_idx):
        b_mat[:, ell] = psi_tilde(scores_arr, j, k, use_haar_norm=use_haar_norm)

    return b_mat


@dataclass(frozen=True)
class HaarMonotoneRidgeFit:
    c: float
    beta: float
    theta: NDArray[np.float64]
    basis_idx: list[tuple[int, int]]
    j_max: int
    lam: float
    use_haar_norm: bool


def fit_monotone_ridge(
    scores: NDArray[np.float64] | NDArray[np.int64],
    y: NDArray[np.float64] | NDArray[np.int64],
    *,
    j_max: int = 6,
    lam: float = 1e-2,
    use_haar_norm: bool = True,
) -> HaarMonotoneRidgeFit:
    """Fit monotone ridge with beta >= 0 and theta >= 0.

    Objective:
    ||y - c*1 - beta*s - B*theta||_2^2 + lam*(beta^2 + ||theta||_2^2)
    """
    if lam < 0:
        raise ValueError("lam must be nonnegative.")

    scores_arr = np.asarray(scores, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    if scores_arr.shape != y_arr.shape:
        raise ValueError("scores and y must have the same shape.")

    basis_idx = build_basis(j_max)
    b_mat = design_matrix(scores_arr, basis_idx, use_haar_norm=use_haar_norm)

    x_mat = np.column_stack((scores_arr, b_mat))
    p = x_mat.shape[1]

    x_mean = x_mat.mean(axis=0)
    y_mean = float(y_arr.mean())

    x_centered = x_mat - x_mean
    y_centered = y_arr - y_mean

    if lam > 0:
        x_aug = np.vstack((x_centered, np.sqrt(lam) * np.eye(p)))
        y_aug = np.concatenate((y_centered, np.zeros(p)))
    else:
        x_aug = x_centered
        y_aug = y_centered

    result = lsq_linear(x_aug, y_aug, bounds=(0.0, np.inf), lsq_solver="exact")
    if not result.success:
        raise RuntimeError(f"Constrained solver failed: {result.message}")

    w = np.asarray(result.x, dtype=float)
    beta = float(w[0])
    theta = w[1:]
    c = float(y_mean - x_mean @ w)

    return HaarMonotoneRidgeFit(
        c=c,
        beta=beta,
        theta=theta,
        basis_idx=basis_idx,
        j_max=j_max,
        lam=lam,
        use_haar_norm=use_haar_norm,
    )


def predict_monotone_ridge(
    scores: NDArray[np.float64] | NDArray[np.int64],
    fit: HaarMonotoneRidgeFit,
) -> NDArray[np.float64]:
    """Predict from a fitted monotone-ridge model."""
    scores_arr = np.asarray(scores, dtype=float).ravel()
    b_mat = design_matrix(scores_arr, fit.basis_idx, use_haar_norm=fit.use_haar_norm)
    return fit.c + fit.beta * scores_arr + b_mat @ fit.theta


def check_monotone(fit: HaarMonotoneRidgeFit, *, grid_n: int = 2001, tol: float = 1e-10) -> bool:
    """Numerical monotonicity check on a dense grid."""
    grid = np.linspace(0.0, 1.0, grid_n)
    g = predict_monotone_ridge(grid, fit)
    return bool(np.all(np.diff(g) >= -tol))
