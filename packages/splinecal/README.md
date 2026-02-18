# splinecal

Spline-based probability calibration with scikit-learn compatible estimators.

## Estimators

- `SplineBinaryCalibrator`: spline features + logistic regression.
- `HaarMonotoneRidgeCalibrator`: double-integrated Haar basis + ridge with nonnegative slope/weights for guaranteed monotonicity.
