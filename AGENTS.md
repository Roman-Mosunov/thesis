# AGENTS.md

## Project Specs
- Repository type: monorepo.
- Core language: Python.
- ML ecosystem: scikit-learn.
- Package path: `packages/splinecal`.
- Analysis path: `analysis/`.
- Thesis path: `thesis/` (Quarto).
- Data paths: `data/raw/` and `data/processed/`.
- Primary package goal: spline-based probability calibration.
- Implemented estimators: `SplineBinaryCalibrator`, `HaarMonotoneRidgeCalibrator`.
- Haar calibration requirement: monotone ridge with nonnegative slope/weights.
- Dev tooling: `uv`, `pytest`, `ruff`, `mypy`, pre-commit.

## Thesis Writing
- Use APA style for all thesis writing in `thesis/`.
- Follow APA7 rules for citations, references, headings, tables, and figures.
- Keep wording, punctuation, and capitalization consistent with APA style.
