from ._version import __version__
from .calibrators import HaarMonotoneRidgeCalibrator, SplineBinaryCalibrator
from .haar import (
    build_basis,
    check_monotone,
    design_matrix,
    fit_monotone_ridge,
    predict_monotone_ridge,
    psi_tilde,
)
from .metrics import brier_score, expected_calibration_error
from .plotting import plot_reliability_diagram, reliability_points, save_reliability_diagram

__all__ = [
    "HaarMonotoneRidgeCalibrator",
    "SplineBinaryCalibrator",
    "build_basis",
    "brier_score",
    "check_monotone",
    "design_matrix",
    "expected_calibration_error",
    "fit_monotone_ridge",
    "plot_reliability_diagram",
    "predict_monotone_ridge",
    "psi_tilde",
    "reliability_points",
    "save_reliability_diagram",
    "__version__",
]
