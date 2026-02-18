from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import SplineTransformer


def as_probability_vector(x: NDArray[np.float64] | NDArray[np.int64]) -> NDArray[np.float64]:
    """Convert model outputs to a 1D positive-class probability vector.

    Accepted shapes:
    - (n_samples,)
    - (n_samples, 1)
    - (n_samples, n_features)  -> last column is used
    """
    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim == 1:
        probs = x_arr
    elif x_arr.ndim == 2 and x_arr.shape[1] == 1:
        probs = x_arr[:, 0]
    elif x_arr.ndim == 2 and x_arr.shape[1] >= 2:
        probs = x_arr[:, -1]
    else:
        raise ValueError("Input must be 1D or 2D array-like.")

    return np.clip(probs, 1e-6, 1 - 1e-6)


def fit_spline_transformer(
    probs: NDArray[np.float64],
    *,
    n_knots: int,
    degree: int,
    include_bias: bool,
) -> tuple[NDArray[np.float64], SplineTransformer]:
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        knots="quantile",
        extrapolation="linear",
    )
    features = transformer.fit_transform(probs.reshape(-1, 1))
    return features, transformer
