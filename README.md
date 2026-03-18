# Flexible Calibration for Decision Support on Small, Imbalanced Datasets

Research code and thesis materials for the bachelor’s thesis **“Calibrating Predictive Models for Decision Support With Flexible Methods on Small, Imbalanced Datasets”** by **Romans Mosunovs**.

This project studies **post-hoc probability calibration** in the regime where calibration is hardest: **small samples, class imbalance, and decision-support settings where probability reliability matters as much as classification accuracy**. The repository combines:

- a reusable Python package for calibration methods (`packages/splinecal`)
- experiment workflows and benchmark scripts (`analysis`)
- a Quarto manuscript project for thesis writing (`thesis`)

## Motivation

A model can rank cases well and still produce **misleading probabilities**. In many decision-support contexts, that is a serious problem: a score of 0.80 should behave like an 80% event rate, not just “a high score.” This thesis focuses on methods that improve calibration while respecting the structural constraints that matter in practice, especially **smoothness**, **monotonicity**, and **stability under limited data**.

The core research question is whether **flexible but constrained regression-based calibrators** can outperform standard post-hoc approaches when datasets are small and imbalanced.

## What is implemented

The main package, `splinecal`, provides scikit-learn-compatible calibration components and evaluation utilities.

### Calibrators

- `SplineBinaryCalibrator` — spline-expanded features with logistic regression
- `PlattBinaryCalibrator` — logistic calibration baseline
- `IsotonicBinaryCalibrator` — non-parametric monotone baseline
- `BetaBinaryCalibrator` — beta calibration using log-score features
- `HaarMonotoneRidgeCalibrator` — a monotone ridge calibrator built on a **double-integrated Haar basis** with nonnegative coefficients

### Evaluation utilities

- Brier score
- Expected Calibration Error (ECE)
- Brier calibration / refinement decomposition
- Log-loss calibration / refinement decomposition
- Reliability-diagram utilities and plotting helpers

## Repository structure

```text
.
├── packages/
│   └── splinecal/        # installable calibration package
├── analysis/            # notebooks, scripts, benchmark workflows
│   ├── notebooks/
│   └── scripts/
├── thesis/              # Quarto manuscript project
├── data/
│   └── raw/
├── pyproject.toml       # workspace definition
└── README.md
```
## Quick start

1. Install `uv`.
2. Sync workspace dependencies:

```bash
uv sync --all-packages --group dev
```

3. Run tests:

```bash
uv run pytest
```

4. Lint:

```bash
uv run ruff check .
```

## Package development

From repo root, editable install is managed via workspace membership. You can also work directly from package dir:

```bash
cd packages/splinecal
uv run pytest
```
