"""Run TabArena-based calibration experiments with persistent artifacts.

This script implements:
1) subset experiments first,
2) fixed j_max/lambda experiments,
3) interval discovery for j_max/lambda,
4) full-data GridSearchCV.

Outputs are stored under `analysis/outputs/tabarena/...` and include:
- result tables for each stage,
- saved models,
- prediction snapshots,
- reliability plots,
- reproducibility metadata (versions, params, ids, git hash).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from splinecal import (
    HaarMonotoneRidgeCalibrator,
    brier_score,
    expected_calibration_error,
    save_reliability_diagram,
)


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    y_train: NDArrayInt
    x_cal: pd.DataFrame
    y_cal: NDArrayInt
    x_test: pd.DataFrame
    y_test: NDArrayInt


NDArrayFloat = np.ndarray
NDArrayInt = np.ndarray


def _slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-").replace("/", "-")


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = "unknown"
    dirty = None
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        )
        dirty = (
            subprocess.run(["git", "diff", "--quiet"], cwd=repo_root, check=False).returncode != 0
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {"commit": commit, "dirty": dirty}


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "splinecal"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _fetch_dataset(
    *,
    dataset_id: int,
    dataset_name: str,
    data_root: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], Path]:
    dataset_slug = _slugify(dataset_name)
    dataset_dir = data_root / dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / f"openml_{dataset_id}.csv"
    meta_path = dataset_dir / f"openml_{dataset_id}_metadata.json"

    if csv_path.exists() and meta_path.exists() and not refresh:
        df = pd.read_csv(csv_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        target_col = metadata["target_column"]
        x = df.drop(columns=[target_col])
        y = df[target_col]
        return x, y, metadata, csv_path

    bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    x = bunch.data.copy()
    y = bunch.target.copy()
    target_col = y.name or "target"

    full_df = x.copy()
    full_df[target_col] = y
    full_df.to_csv(csv_path, index=False)

    class_counts = y.astype(str).value_counts(dropna=False).to_dict()
    metadata = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "target_column": target_col,
        "n_rows": int(full_df.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(y.astype(str).nunique()),
        "class_counts_raw": class_counts,
        "persisted_at_utc": _utc_now_iso(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return x, y, metadata, csv_path


def _binary_encode_target(
    y_raw: pd.Series,
    *,
    positive_label: str | None,
) -> tuple[NDArrayInt, dict[str, Any]]:
    y_str = y_raw.astype(str)
    value_counts = y_str.value_counts(dropna=False)
    if value_counts.shape[0] != 2:
        raise ValueError(f"Expected binary target, got {value_counts.shape[0]} classes.")

    if positive_label is None:
        positive = str(value_counts.sort_values(ascending=True).index[0])
    else:
        positive = positive_label
        if positive not in value_counts.index:
            raise ValueError(
                f"positive_label '{positive}' is not in target classes "
                f"{list(value_counts.index)}."
            )

    negative = [label for label in value_counts.index.astype(str).tolist() if label != positive][0]
    y_bin = (y_str == positive).astype(int).to_numpy()

    mapping = {
        "positive_label": positive,
        "negative_label": negative,
        "class_counts_raw": {str(k): int(v) for k, v in value_counts.to_dict().items()},
        "positive_rate": float(y_bin.mean()),
    }
    return y_bin, mapping


def _build_base_model(x: pd.DataFrame, random_state: int) -> Pipeline:
    num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in x.columns if col not in num_cols]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("prep", preprocessor), ("model", classifier)])


def _split_data(
    x: pd.DataFrame,
    y: NDArrayInt,
    *,
    random_state: int,
) -> SplitData:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=random_state,
    )
    return SplitData(
        x_train=x_train.reset_index(drop=True),
        y_train=np.asarray(y_train, dtype=int),
        x_cal=x_cal.reset_index(drop=True),
        y_cal=np.asarray(y_cal, dtype=int),
        x_test=x_test.reset_index(drop=True),
        y_test=np.asarray(y_test, dtype=int),
    )


def _metrics_row(
    *,
    method: str,
    phase: str,
    y_true: NDArrayInt,
    y_prob: NDArrayFloat,
    ece_bins: int,
    j_max: int | None = None,
    lam: float | None = None,
    subset_fraction: float | None = None,
) -> dict[str, Any]:
    probs = np.asarray(y_prob, dtype=float).ravel()
    return {
        "method": method,
        "phase": phase,
        "subset_fraction": subset_fraction,
        "j_max": j_max,
        "lam": lam,
        "n_samples": int(y_true.shape[0]),
        "positive_rate": float(np.mean(y_true)),
        "brier_score": brier_score(y_true, probs),
        "ece": expected_calibration_error(y_true, probs, n_bins=ece_bins),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-15, 1.0 - 1e-15))),
    }


def _fit_haar(
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_eval: NDArrayFloat,
    *,
    j_max: int,
    lam: float,
) -> NDArrayFloat:
    calibrator = HaarMonotoneRidgeCalibrator(j_max=j_max, lam=lam)
    calibrator.fit(scores_cal, y_cal)
    return calibrator.predict_proba(scores_eval)[:, 1]


def _subset_indices(y: NDArrayInt, frac: float, random_state: int) -> NDArrayInt:
    if frac >= 1.0:
        return np.arange(y.shape[0], dtype=int)
    if frac <= 0.0:
        raise ValueError("subset fraction must be in (0, 1].")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=random_state)
    try:
        train_idx, _ = next(splitter.split(np.zeros_like(y), y))
        return np.asarray(train_idx, dtype=int)
    except ValueError:
        rng = np.random.default_rng(random_state)
        cls_indices = [np.where(y == cls)[0] for cls in np.unique(y)]
        sampled: list[int] = []
        for ids in cls_indices:
            take = max(1, int(round(ids.size * frac)))
            sampled.extend(rng.choice(ids, size=min(take, ids.size), replace=False).tolist())
        sampled_arr = np.asarray(sorted(set(sampled)), dtype=int)
        return sampled_arr


def _run_subset_fixed_stage(
    *,
    subset_fracs: list[float],
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_test: NDArrayFloat,
    y_test: NDArrayInt,
    fixed_j_max: int,
    fixed_lam: float,
    ece_bins: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, frac in enumerate(subset_fracs):
        seed = random_state + idx
        cal_idx = _subset_indices(y_cal, frac, seed)
        test_idx = _subset_indices(y_test, frac, seed + 1000)

        y_cal_sub = y_cal[cal_idx]
        p_cal_sub = scores_cal[cal_idx]
        y_test_sub = y_test[test_idx]
        p_test_sub = scores_test[test_idx]

        rows.append(
            _metrics_row(
                method="base_model",
                phase="subset_fixed",
                y_true=y_test_sub,
                y_prob=p_test_sub,
                ece_bins=ece_bins,
                subset_fraction=frac,
            )
        )

        p_fixed = _fit_haar(
            p_cal_sub,
            y_cal_sub,
            p_test_sub,
            j_max=fixed_j_max,
            lam=fixed_lam,
        )
        rows.append(
            _metrics_row(
                method="haar_fixed",
                phase="subset_fixed",
                y_true=y_test_sub,
                y_prob=p_fixed,
                ece_bins=ece_bins,
                j_max=fixed_j_max,
                lam=fixed_lam,
                subset_fraction=frac,
            )
        )
    return pd.DataFrame(rows)


def _run_interval_scan(
    *,
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_test: NDArrayFloat,
    y_test: NDArrayInt,
    subset_fraction: float,
    j_values: list[int],
    lam_values: list[float],
    ece_bins: int,
    random_state: int,
) -> pd.DataFrame:
    cal_idx = _subset_indices(y_cal, subset_fraction, random_state + 222)
    test_idx = _subset_indices(y_test, subset_fraction, random_state + 333)

    y_cal_sub = y_cal[cal_idx]
    p_cal_sub = scores_cal[cal_idx]
    y_test_sub = y_test[test_idx]
    p_test_sub = scores_test[test_idx]

    rows: list[dict[str, Any]] = []
    for j_max in j_values:
        for lam in lam_values:
            p_pred = _fit_haar(p_cal_sub, y_cal_sub, p_test_sub, j_max=j_max, lam=lam)
            row = _metrics_row(
                method="haar_scan",
                phase="interval_scan",
                y_true=y_test_sub,
                y_prob=p_pred,
                ece_bins=ece_bins,
                j_max=j_max,
                lam=lam,
                subset_fraction=subset_fraction,
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values("brier_score", ascending=True).reset_index(drop=True)


def _recommend_ranges(
    scan_df: pd.DataFrame,
    *,
    top_quantile: float,
) -> dict[str, Any]:
    if scan_df.empty:
        raise ValueError("scan_df is empty.")
    if not 0.0 < top_quantile <= 1.0:
        raise ValueError("top_quantile must be in (0, 1].")

    n_top = max(1, int(np.ceil(scan_df.shape[0] * top_quantile)))
    top = scan_df.nsmallest(n_top, "brier_score")

    j_candidates = sorted(top["j_max"].astype(int).unique().tolist())
    lam_candidates = sorted(top["lam"].astype(float).unique().tolist())
    return {
        "top_quantile": top_quantile,
        "n_top_rows": n_top,
        "j_candidates": j_candidates,
        "lam_candidates": lam_candidates,
        "recommended_j_interval": [int(min(j_candidates)), int(max(j_candidates))],
        "recommended_lam_interval": [float(min(lam_candidates)), float(max(lam_candidates))],
    }


def _run_gridsearch(
    *,
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    j_candidates: list[int],
    lam_candidates: list[float],
    cv_folds: int,
    random_state: int,
) -> GridSearchCV:
    estimator = HaarMonotoneRidgeCalibrator(use_haar_norm=True, clip_probs=True)
    param_grid = {"j_max": j_candidates, "lam": lam_candidates}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_brier_score",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(scores_cal.reshape(-1, 1), y_cal)
    return grid


def _save_predictions_table(
    *,
    output_path: Path,
    y_true: NDArrayInt,
    prob_base: NDArrayFloat,
    prob_fixed: NDArrayFloat,
    prob_gridsearch_best: NDArrayFloat,
) -> None:
    table = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "prob_base": prob_base,
            "prob_fixed": prob_fixed,
            "prob_gridsearch_best": prob_gridsearch_best,
        }
    )
    table.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persistent TabArena calibration experiments.")
    parser.add_argument("--dataset-name", type=str, default="blood-transfusion-service-center")
    parser.add_argument("--dataset-id", type=int, default=46913)
    parser.add_argument("--task-id", type=int, default=363621)
    parser.add_argument("--positive-label", type=str, default=None)
    parser.add_argument("--refresh-dataset", action="store_true")

    parser.add_argument("--subset-fracs", type=str, default="0.1,0.25,0.5")
    parser.add_argument("--fixed-j-max", type=int, default=6)
    parser.add_argument("--fixed-lam", type=float, default=1e-2)

    parser.add_argument("--interval-subset-frac", type=float, default=0.5)
    parser.add_argument("--interval-j-values", type=str, default="2,3,4,5,6,7,8")
    parser.add_argument(
        "--interval-lam-values",
        type=str,
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0",
    )
    parser.add_argument("--grid-top-quantile", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)

    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--plot-bins", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--data-root", type=Path, default=Path("data/raw/tabarena"))
    parser.add_argument("--output-root", type=Path, default=Path("analysis/outputs/tabarena"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = _utc_now_iso()

    repo_root = Path(__file__).resolve().parents[2]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dataset_slug = _slugify(args.dataset_name)
    run_dir = args.output_root / dataset_slug / run_id
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    x, y_raw, dataset_meta, dataset_csv_path = _fetch_dataset(
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        refresh=args.refresh_dataset,
    )
    y, label_map = _binary_encode_target(y_raw, positive_label=args.positive_label)

    split = _split_data(x, y, random_state=args.random_state)
    base_model = _build_base_model(x, random_state=args.random_state)
    base_model.fit(split.x_train, split.y_train)

    p_cal = base_model.predict_proba(split.x_cal)[:, 1]
    p_test = base_model.predict_proba(split.x_test)[:, 1]

    subset_fracs = _parse_float_list(args.subset_fracs)
    subset_df = _run_subset_fixed_stage(
        subset_fracs=subset_fracs,
        scores_cal=p_cal,
        y_cal=split.y_cal,
        scores_test=p_test,
        y_test=split.y_test,
        fixed_j_max=args.fixed_j_max,
        fixed_lam=args.fixed_lam,
        ece_bins=args.ece_bins,
        random_state=args.random_state,
    )
    subset_df.to_csv(run_dir / "subset_fixed_results.csv", index=False)

    p_fixed_full = _fit_haar(
        p_cal,
        split.y_cal,
        p_test,
        j_max=args.fixed_j_max,
        lam=args.fixed_lam,
    )

    interval_scan = _run_interval_scan(
        scores_cal=p_cal,
        y_cal=split.y_cal,
        scores_test=p_test,
        y_test=split.y_test,
        subset_fraction=args.interval_subset_frac,
        j_values=_parse_int_list(args.interval_j_values),
        lam_values=_parse_float_list(args.interval_lam_values),
        ece_bins=args.ece_bins,
        random_state=args.random_state,
    )
    interval_scan.to_csv(run_dir / "interval_scan_results.csv", index=False)

    recommendation = _recommend_ranges(interval_scan, top_quantile=args.grid_top_quantile)
    (run_dir / "recommended_ranges.json").write_text(
        json.dumps(recommendation, indent=2),
        encoding="utf-8",
    )

    grid = _run_gridsearch(
        scores_cal=p_cal,
        y_cal=split.y_cal,
        j_candidates=recommendation["j_candidates"],
        lam_candidates=recommendation["lam_candidates"],
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    grid_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    grid_results.to_csv(run_dir / "gridsearch_cv_results.csv", index=False)

    best_calibrator = grid.best_estimator_
    p_grid_best = best_calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]

    final_rows = [
        _metrics_row(
            method="base_model",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_test,
            ece_bins=args.ece_bins,
        ),
        _metrics_row(
            method="haar_fixed",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_fixed_full,
            ece_bins=args.ece_bins,
            j_max=args.fixed_j_max,
            lam=args.fixed_lam,
        ),
        _metrics_row(
            method="haar_gridsearch_best",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_grid_best,
            ece_bins=args.ece_bins,
            j_max=int(grid.best_params_["j_max"]),
            lam=float(grid.best_params_["lam"]),
        ),
    ]
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(run_dir / "final_test_metrics.csv", index=False)

    _save_predictions_table(
        output_path=run_dir / "predictions_test.csv",
        y_true=split.y_test,
        prob_base=p_test,
        prob_fixed=p_fixed_full,
        prob_gridsearch_best=p_grid_best,
    )

    save_reliability_diagram(
        split.y_test,
        p_test,
        output_path=plots_dir / "reliability_base.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title="Base model reliability",
    )
    save_reliability_diagram(
        split.y_test,
        p_fixed_full,
        output_path=plots_dir / "reliability_haar_fixed.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title=f"Haar fixed reliability (j_max={args.fixed_j_max}, lam={args.fixed_lam:g})",
    )
    save_reliability_diagram(
        split.y_test,
        p_grid_best,
        output_path=plots_dir / "reliability_haar_gridsearch_best.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title=(
            f"Haar best reliability (j_max={grid.best_params_['j_max']}, "
            f"lam={grid.best_params_['lam']:g})"
        ),
    )

    dump(base_model, models_dir / "base_model.joblib")
    fixed_calibrator = HaarMonotoneRidgeCalibrator(j_max=args.fixed_j_max, lam=args.fixed_lam)
    fixed_calibrator.fit(p_cal, split.y_cal)
    dump(
        fixed_calibrator,
        models_dir / "haar_fixed_calibrator.joblib",
    )
    dump(best_calibrator, models_dir / "haar_gridsearch_best_calibrator.joblib")

    finished_at = _utc_now_iso()
    run_metadata = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "dataset": {
            "dataset_name": args.dataset_name,
            "dataset_id": args.dataset_id,
            "task_id": args.task_id,
            "dataset_csv_path": str(dataset_csv_path),
            "tabarena_dataset_metadata": dataset_meta,
            "label_mapping": label_map,
        },
        "split": {
            "train_rows": int(split.x_train.shape[0]),
            "calibration_rows": int(split.x_cal.shape[0]),
            "test_rows": int(split.x_test.shape[0]),
            "random_state": args.random_state,
        },
        "parameters": {
            "fixed_j_max": args.fixed_j_max,
            "fixed_lam": args.fixed_lam,
            "subset_fracs": subset_fracs,
            "interval_subset_frac": args.interval_subset_frac,
            "interval_j_values": _parse_int_list(args.interval_j_values),
            "interval_lam_values": _parse_float_list(args.interval_lam_values),
            "grid_top_quantile": args.grid_top_quantile,
            "grid_cv_folds": args.cv_folds,
            "ece_bins": args.ece_bins,
            "plot_bins": args.plot_bins,
        },
        "gridsearch_best_params": {
            "j_max": int(grid.best_params_["j_max"]),
            "lam": float(grid.best_params_["lam"]),
            "best_cv_neg_brier": float(grid.best_score_),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "git": _git_metadata(repo_root),
        },
        "tabarena_reference": {
            "dataset_curation_repo": "https://github.com/TabArena/tabarena_dataset_curation",
            "classifier_calibration_repo": "https://github.com/valeman/classifier_calibration/tree/release-v1.0",
            "leaderboard": "https://huggingface.co/spaces/TabArena/leaderboard",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print("Saved files:")
    print(f"- {run_dir / 'subset_fixed_results.csv'}")
    print(f"- {run_dir / 'interval_scan_results.csv'}")
    print(f"- {run_dir / 'gridsearch_cv_results.csv'}")
    print(f"- {run_dir / 'final_test_metrics.csv'}")
    print(f"- {run_dir / 'predictions_test.csv'}")
    print(f"- {run_dir / 'recommended_ranges.json'}")
    print(f"- {run_dir / 'run_metadata.json'}")
    print(f"- {plots_dir / 'reliability_base.png'}")
    print(f"- {plots_dir / 'reliability_haar_fixed.png'}")
    print(f"- {plots_dir / 'reliability_haar_gridsearch_best.png'}")
    print(f"- {models_dir / 'base_model.joblib'}")
    print(f"- {models_dir / 'haar_fixed_calibrator.joblib'}")
    print(f"- {models_dir / 'haar_gridsearch_best_calibrator.joblib'}")


if __name__ == "__main__":
    main()
