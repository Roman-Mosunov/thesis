# TabArena Calibration Workflow

This runbook links `splinecal` experiments to TabArena dataset IDs and stores reproducible artifacts.

## Dataset used by default

- TabArena dataset: `blood-transfusion-service-center`
- OpenML dataset id: `46913`
- TabArena task id: `363621`
- Local persistent snapshot:
  - `data/raw/tabarena/blood-transfusion-service-center/openml_46913.csv`
  - `data/raw/tabarena/blood-transfusion-service-center/openml_46913_metadata.json`

## End-to-end command

From repo root:

```bash
uv run python analysis/scripts/run_tabarena_haar_experiment.py
```

This runs, in order:
1. Subset-first experiments (`--subset-fracs`, fixed `j_max/lambda`)
2. Fixed-parameter full-test evaluation
3. Interval scan for `j_max/lambda` on subset
4. Full `GridSearchCV` on calibration data from the full dataset

## Adjustable parameters

- Fixed configuration:
  - `--fixed-j-max`
  - `--fixed-lam`
- Subset stage:
  - `--subset-fracs`
- Interval discovery:
  - `--interval-subset-frac`
  - `--interval-j-values`
  - `--interval-lam-values`
  - `--grid-top-quantile`
- Grid search:
  - `--cv-folds`
- Dataset binding:
  - `--dataset-name`
  - `--dataset-id`
  - `--task-id`
  - `--positive-label`

## Persisted outputs (per run)

Run directory pattern:

`analysis/outputs/tabarena/<dataset-slug>/<UTC-run-id>/`

Saved tables:
- `subset_fixed_results.csv`
- `interval_scan_results.csv`
- `gridsearch_cv_results.csv`
- `final_test_metrics.csv`
- `predictions_test.csv`
- `recommended_ranges.json`
- `run_metadata.json`

Saved models:
- `models/base_model.joblib`
- `models/haar_fixed_calibrator.joblib`
- `models/haar_gridsearch_best_calibrator.joblib`

Saved plots:
- `plots/reliability_base.png`
- `plots/reliability_haar_fixed.png`
- `plots/reliability_haar_gridsearch_best.png`

## Comparability with TabArena classifier-calibration benchmark

The stored outputs align with the benchmark style by including:
- Dataset identifiers (`dataset_id`, `task_id`)
- Core calibration metrics (`brier_score`, `ece`, `log_loss`)
- Hyperparameters (`j_max`, `lam`)
- Reproducibility metadata (`python`, package versions, git commit/dirty status)

TabArena references:
- https://github.com/valeman/classifier_calibration/tree/release-v1.0
- https://github.com/TabArena/tabarena_dataset_curation
- https://huggingface.co/spaces/TabArena/leaderboard
