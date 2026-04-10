# TabArena Calibration Workflow

This runbook provides a reproducible workflow for OpenML/TabArena experiments with:
- uncalibrated logistic model,
- spline calibrator,
- Platt calibrator,
- isotonic calibrator,
- beta calibrator,
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

## Additional imbalanced binary dataset options

Strong small/medium options from TabArena curation metadata + OpenML class counts:
- `taiwanese-bankruptcy-prediction` (`dataset_id=46962`, `task_id=363706`, minority rate `3.23%`, `n=6819`)
- `coil2000-insurance-policies` (`dataset_id=46916`, `task_id=363624`, minority rate `5.97%`, `n=9822`)
- `polish-companies-bankruptcy` (`dataset_id=46950`, `task_id=363694`, minority rate `6.94%`, `n=5910`)


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

List built-in dataset presets:

```bash
uv run python analysis/scripts/run_tabarena_haar_experiment.py --list-dataset-presets
```

Run one preset dataset:

```bash
uv run python analysis/scripts/run_tabarena_haar_experiment.py \
  --dataset-preset taiwanese-bankruptcy-prediction
```

Run a batch of datasets (same outputs per dataset):

```bash
uv run python analysis/scripts/run_tabarena_haar_batch.py \
  --use-recommended-small-imbalanced
```

Run all TabArena binary datasets from metadata:

```bash
uv run python analysis/scripts/run_tabarena_haar_batch.py \
  --all-binary
```

Inspect which datasets will run (without executing):

```bash
uv run python analysis/scripts/run_tabarena_haar_batch.py \
  --all-binary --dry-run --max-datasets 10
```

Or pass your own dataset list:

```bash
uv run python analysis/scripts/run_tabarena_haar_batch.py \
  --dataset-presets kddcup09-appetency,bank-marketing,credit-card-clients-default
```

Recommended convergence-friendly run settings (many datasets):

```bash
uv run python analysis/scripts/run_tabarena_haar_batch.py \
  --all-binary \
  --numeric-scaler robust \
  --logreg-solver saga \
  --logreg-max-iter 8000 \
  --suppress-fitfailed-warnings
```

Quick development run (faster, less exhaustive than thesis defaults):

```bash
uv run python analysis/scripts/run_tabarena_haar_experiment.py \
  --cv-folds 2 \
  --lambda-stage1-points 50 \
  --lambda-stage2-points 50
```

## What each run executes

1. Train base classifier and create calibration/test splits.
2. Run subset-stage comparisons (uncalibrated logistic, spline, Platt, isotonic, beta).
3. Run Haar two-stage `GridSearchCV` on calibration data:
   - Stage 1: broad logspace over `10^-6` to `10^0` to identify the best decade.
   - Stage 2: dense logspace within the selected decade.
4. Evaluate all estimators on test split.
5. Run cross-validated train/test evaluation of calibration fitting (`--cv-folds`, default 5).
6. Save per-estimator reliability diagrams and global comparison plots.

## Main adjustable parameters

- Dataset:
  - `--dataset-preset`
  - `--list-dataset-presets`
  - `--dataset-name`
  - `--dataset-id`
  - `--task-id`
  - `--positive-label`
  - `--refresh-dataset`
  - `--numeric-scaler` (`standard|robust|minmax|none`)
  - `--onehot-min-frequency`
  - `--logreg-solver` (`auto|lbfgs|liblinear|newton-cg|newton-cholesky|sag|saga`)
  - `--logreg-max-iter`
  - `--suppress-fitfailed-warnings`
  - `--suppress-convergence-warnings`
- Spline calibrator:
  - `--spline-n-knots`
  - `--spline-degree`
  - `--spline-c`
  - `--spline-max-iter`
  - `--spline-include-bias`
- Platt calibrator:
  - `--platt-c`
  - `--platt-max-iter`
- Beta calibrator:
  - `--beta-c`
  - `--beta-max-iter`
- Haar two-stage grid search:
  - `--grid-j-min` (default `1`)
  - `--grid-j-max` (default `6`)
  - `--lambda-min-exp` (default `-6`)
  - `--lambda-max-exp` (default `0`)
  - `--lambda-stage1-points` (default `50`)
  - `--lambda-stage2-points` (default `50`)
  - `--cv-folds` (`5`)
- Plotting:
  - `--plot-bins`
  - `--ece-bins`
  - `--estimator-grid-points`

## Persisted outputs (per run)

Run directory pattern:

`analysis/outputs/tabarena/<dataset-slug>/<UTC-run-id>/`

Saved tables:
- `subset_results.csv`
- `lambda_stage1_gridsearch_cv_results.csv`
- `lambda_stage2_gridsearch_cv_results.csv`
- `interval_scan_results.csv` (backward-compatible alias of stage-1 grid results)
- `gridsearch_cv_results.csv` (backward-compatible alias of stage-2 grid results)
- `cross_validated_train_test_metrics.csv`
- `final_test_metrics.csv`
- `predictions_test.csv`
- `estimator_curves.csv`
- `graph_summary_reliability.csv`
- `graph_summary_estimator_mapping.csv`
- `graph_summaries.md`
- `recommended_ranges.json`
- `run_metadata.json`

Saved models:
- `models/base_model.joblib`
- `models/spline_fixed_calibrator.joblib`
- `models/platt_calibrator.joblib`
- `models/isotonic_calibrator.joblib`
- `models/beta_calibrator.joblib`
- `models/haar_gridsearch_best_calibrator.joblib`

Saved plots:
- `plots/reliability_uncalibrated_logistic.png`
- `plots/reliability_spline_fixed.png`
- `plots/reliability_platt.png`
- `plots/reliability_isotonic.png`
- `plots/reliability_beta.png`
- `plots/reliability_haar_gridsearch_best.png`
- `plots/reliability_all_estimators_comparison.png`
- `plots/reliability_all_estimators_panel.png`
- `plots/smoothed_calibration_uncalibrated_spline_beta_comparison.png`
- `plots/smoothed_calibration_platt_isotonic_haar_comparison.png`
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
- estimator hyperparameters (spline, Platt, beta + Haar),
- reproducibility metadata (python, package versions, git commit/dirty).

References:
- https://github.com/valeman/classifier_calibration/tree/release-v1.0
- https://github.com/TabArena/tabarena_dataset_curation
- https://huggingface.co/spaces/TabArena/leaderboard
