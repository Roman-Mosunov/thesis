# Calibration Monorepo

This repository is organized as a monorepo for three parallel tracks:

- Python package development (`packages/splinecal`)
- Analysis workflows (`analysis`)
- Thesis writing with Quarto (`thesis`)

## Layout

- `packages/splinecal`: installable package for spline-based calibration with a scikit-learn style API.
- `analysis`: scripts and notebooks that use the package to run experiments.
- `thesis`: Quarto project for manuscript/thesis writing.
- `data`: local data folders (`raw` and `processed`).

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
