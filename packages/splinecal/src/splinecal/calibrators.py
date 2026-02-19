from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted

from .haar import fit_monotone_ridge, predict_monotone_ridge
from .splines import as_probability_vector, fit_spline_transformer


class SplineBinaryCalibrator(ClassifierMixin, BaseEstimator):
    """Binary probability calibrator using spline-expanded features.

    The estimator expects model confidence inputs and calibrates the positive-class
    probability using logistic regression over spline basis features.
    """

    def __init__(
        self,
        *,
        n_knots: int = 5,
        degree: int = 3,
        include_bias: bool = False,
        c: float = 1.0,
        max_iter: int = 500,
    ) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias
        self.c = c
        self.max_iter = max_iter

    def fit(
        self,
        x: NDArray[np.float64] | NDArray[np.int64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> SplineBinaryCalibrator:
        x_checked = check_array(x, ensure_2d=False)
        y_arr = np.asarray(y)

        if type_of_target(y_arr) != "binary":
            raise ValueError("SplineBinaryCalibrator only supports binary targets.")

        probs = as_probability_vector(x_checked)
        basis, transformer = fit_spline_transformer(
            probs,
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=self.include_bias,
        )

        model = LogisticRegression(C=self.c, solver="lbfgs", max_iter=self.max_iter)
        model.fit(basis, y_arr)

        self._transformer = transformer
        self._model = model
        self.classes_ = model.classes_
        self.n_features_in_ = 1
        return self

    def predict_proba(
        self,
        x: NDArray[np.float64] | NDArray[np.int64],
    ) -> NDArray[np.float64]:
        check_is_fitted(self, ["_transformer", "_model", "classes_"])
        x_checked = check_array(x, ensure_2d=False)
        probs = as_probability_vector(x_checked)

        basis = self._transformer.transform(probs.reshape(-1, 1))
        positive = self._model.predict_proba(basis)[:, 1]
        negative = 1.0 - positive
        return np.column_stack((negative, positive))

    def predict(
        self,
        x: NDArray[np.float64] | NDArray[np.int64],
    ) -> NDArray[np.float64]:
        proba = self.predict_proba(x)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class HaarMonotoneRidgeCalibrator(ClassifierMixin, BaseEstimator):
    """Binary calibrator using double-integrated Haar basis and monotone ridge."""

    def __init__(
        self,
        *,
        j_max: int = 6,
        lam: float = 1e-2,
        use_haar_norm: bool = True,
        clip_probs: bool = True,
    ) -> None:
        self.j_max = j_max
        self.lam = lam
        self.use_haar_norm = use_haar_norm
        self.clip_probs = clip_probs

    def fit(
        self,
        x: NDArray[np.float64] | NDArray[np.int64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> HaarMonotoneRidgeCalibrator:
        x_checked = check_array(x, ensure_2d=False)
        y_arr = np.asarray(y)

        if type_of_target(y_arr) != "binary":
            raise ValueError("HaarMonotoneRidgeCalibrator only supports binary targets.")

        classes = np.unique(y_arr)
        y_binary = (y_arr == classes[1]).astype(float)
        probs = as_probability_vector(x_checked)

        fit_result = fit_monotone_ridge(
            probs,
            y_binary,
            j_max=self.j_max,
            lam=self.lam,
            use_haar_norm=self.use_haar_norm,
        )

        self._fit = fit_result
        self.classes_ = classes
        self.n_features_in_ = 1
        return self

    def predict_proba(
        self,
        x: NDArray[np.float64] | NDArray[np.int64],
    ) -> NDArray[np.float64]:
        check_is_fitted(self, ["_fit", "classes_"])
        x_checked = check_array(x, ensure_2d=False)
        probs = as_probability_vector(x_checked)

        positive = predict_monotone_ridge(probs, self._fit)
        if self.clip_probs:
            positive = np.clip(positive, 0.0, 1.0)
        negative = 1.0 - positive
        return np.column_stack((negative, positive))

    def predict(
        self,
        x: NDArray[np.float64] | NDArray[np.int64],
    ) -> NDArray[np.float64]:
        proba = self.predict_proba(x)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
