# TabArena Calibration Workflow

This runbook provides a reproducible workflow for OpenML/TabArena experiments with:
- base model (uncalibrated),
- spline calibrator,
- Haar calibrator (fixed),
- Haar calibrator (grid-search best).

It keeps the previous reliability outputs and adds explicit comparison plots for:
- estimator mappings (`raw score -> calibrated probability`),
- reliability curves across all estimators.

## Default dataset binding

- TabArena dataset: `blood-transfusion-service-center`
- OpenML dataset id: `46913`
- TabArena task id: `363621`
- Local snapshot:
  - `data/raw/tabarena/blood-transfusion-service-center/openml_46913.csv`
  - `data/raw/tabarena/blood-transfusion-service-center/openml_46913_metadata.json`

## Reproducible run steps

From repo root:

```bash
uv sync --all-packages --group dev
uv run python analysis/scripts/run_tabarena_haar_experiment.py
```

Optional: force a dataset refresh from OpenML.

```bash
uv run python analysis/scripts/run_tabarena_haar_experiment.py --refresh-dataset
```

## What each run executes

1. Train base classifier and create calibration/test splits.
2. Run subset-stage comparisons (base, spline, Haar fixed).
3. Run Haar interval scan (`j_max`, `lam`) on subset.
4. Run Haar full-data `GridSearchCV`.
5. Evaluate all estimators on test split.
6. Save per-estimator reliability diagrams and global comparison plots.

## Main adjustable parameters

- Dataset:
  - `--dataset-name`
  - `--dataset-id`
  - `--task-id`
  - `--positive-label`
  - `--refresh-dataset`
- Spline calibrator:
  - `--spline-n-knots`
  - `--spline-degree`
  - `--spline-c`
  - `--spline-max-iter`
  - `--spline-include-bias`
- Haar fixed:
  - `--fixed-j-max`
  - `--fixed-lam`
- Haar interval/grid:
  - `--interval-subset-frac`
  - `--interval-j-values`
  - `--interval-lam-values`
  - `--grid-top-quantile`
  - `--cv-folds`
- Plotting:
  - `--plot-bins`
  - `--ece-bins`
  - `--estimator-grid-points`

## Persisted outputs (per run)

Run directory pattern:

`analysis/outputs/tabarena/<dataset-slug>/<UTC-run-id>/`

Saved tables:
- `subset_fixed_results.csv`
- `interval_scan_results.csv`
- `gridsearch_cv_results.csv`
- `final_test_metrics.csv`
- `predictions_test.csv`
- `estimator_curves.csv`
- `recommended_ranges.json`
- `run_metadata.json`

Saved models:
- `models/base_model.joblib`
- `models/spline_fixed_calibrator.joblib`
- `models/haar_fixed_calibrator.joblib`
- `models/haar_gridsearch_best_calibrator.joblib`

Saved plots (old ones retained + new comparisons):
- `plots/reliability_base.png`
- `plots/reliability_spline_fixed.png`
- `plots/reliability_haar_fixed.png`
- `plots/reliability_haar_gridsearch_best.png`
- `plots/reliability_all_estimators_comparison.png`
- `plots/reliability_all_estimators_panel.png`
- `plots/estimator_all_calibrators_comparison.png`

## Thesis usage notes

- Use `final_test_metrics.csv` for the summary comparison table in text.
- Use `plots/estimator_all_calibrators_comparison.png` for estimator-shape comparison.
- Use `plots/reliability_all_estimators_comparison.png` for calibration-curve comparison.
- Keep per-estimator reliability images for appendix or method-specific discussion.

## Comparability with TabArena benchmark

Stored outputs include:
- dataset identifiers (`dataset_id`, `task_id`),
- core metrics (`brier_score`, `ece`, `log_loss`),
- estimator hyperparameters (spline + Haar),
- reproducibility metadata (python, package versions, git commit/dirty).

References:
- https://github.com/valeman/classifier_calibration/tree/release-v1.0
- https://github.com/TabArena/tabarena_dataset_curation
- https://huggingface.co/spaces/TabArena/leaderboard
