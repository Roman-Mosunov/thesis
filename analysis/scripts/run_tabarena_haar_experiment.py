"""Run TabArena-based calibration experiments with persistent artifacts.

This script implements:
1) subset experiments first,
2) interval discovery for Haar j_max/lambda,
3) full-data GridSearchCV for Haar best model.

Outputs are stored under `analysis/outputs/tabarena/...` and include:
- result tables for each stage,
- saved models,
- prediction snapshots,
- reliability plots (per-estimator and comparison),
- estimator mapping comparison plots,
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

import matplotlib.pyplot as plt
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
    BetaBinaryCalibrator,
    HaarMonotoneRidgeCalibrator,
    IsotonicBinaryCalibrator,
    PlattBinaryCalibrator,
    SplineBinaryCalibrator,
    brier_calibration_refinement_loss,
    brier_score,
    expected_calibration_error,
    log_loss_calibration_refinement_loss,
    reliability_points,
    save_reliability_diagram,
)
from tabarena_dataset_presets import (
    format_dataset_presets_table,
    resolve_dataset_preset,
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


METRIC_COLUMNS_ORDER = [
    "method",
    "phase",
    "cv_fold",
    "subset_fraction",
    "j_max",
    "lam",
    "spline_n_knots",
    "train_samples",
    "calibration_samples",
    "test_samples",
    "n_samples",
    "positive_rate",
    "brier_score",
    "brier_calibration_loss",
    "brier_refinement_loss",
    "ece",
    "log_loss",
    "calibration_loss",
    "refinement_loss",
]


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
    spline_n_knots: int | None = None,
    subset_fraction: float | None = None,
) -> dict[str, Any]:
    probs = np.asarray(y_prob, dtype=float).ravel()
    brier_calibration_loss, brier_refinement_loss = brier_calibration_refinement_loss(
        y_true,
        probs,
        n_bins=ece_bins,
    )
    calibration_loss, refinement_loss = log_loss_calibration_refinement_loss(
        y_true,
        probs,
        n_bins=ece_bins,
    )
    return {
        "method": method,
        "phase": phase,
        "subset_fraction": subset_fraction,
        "j_max": j_max,
        "lam": lam,
        "spline_n_knots": spline_n_knots,
        "n_samples": int(y_true.shape[0]),
        "positive_rate": float(np.mean(y_true)),
        "brier_score": brier_score(y_true, probs),
        "brier_calibration_loss": brier_calibration_loss,
        "brier_refinement_loss": brier_refinement_loss,
        "calibration_loss": calibration_loss,
        "refinement_loss": refinement_loss,
        "ece": expected_calibration_error(y_true, probs, n_bins=ece_bins),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-15, 1.0 - 1e-15))),
    }


def _reorder_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered_existing = [col for col in METRIC_COLUMNS_ORDER if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered_existing]
    return df.loc[:, ordered_existing + remaining]


def _fit_spline(
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_eval: NDArrayFloat,
    *,
    n_knots: int,
    degree: int,
    include_bias: bool,
    c: float,
    max_iter: int,
) -> NDArrayFloat:
    calibrator = SplineBinaryCalibrator(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        c=c,
        max_iter=max_iter,
    )
    calibrator.fit(scores_cal, y_cal)
    return calibrator.predict_proba(scores_eval)[:, 1]


def _fit_platt(
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_eval: NDArrayFloat,
    *,
    c: float,
    max_iter: int,
) -> NDArrayFloat:
    calibrator = PlattBinaryCalibrator(c=c, max_iter=max_iter)
    calibrator.fit(scores_cal, y_cal)
    return calibrator.predict_proba(scores_eval)[:, 1]


def _fit_isotonic(
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_eval: NDArrayFloat,
) -> NDArrayFloat:
    calibrator = IsotonicBinaryCalibrator(out_of_bounds="clip")
    calibrator.fit(scores_cal, y_cal)
    return calibrator.predict_proba(scores_eval)[:, 1]


def _fit_beta(
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_eval: NDArrayFloat,
    *,
    c: float,
    max_iter: int,
) -> NDArrayFloat:
    calibrator = BetaBinaryCalibrator(c=c, max_iter=max_iter)
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


def _run_subset_stage(
    *,
    subset_fracs: list[float],
    scores_cal: NDArrayFloat,
    y_cal: NDArrayInt,
    scores_test: NDArrayFloat,
    y_test: NDArrayInt,
    spline_n_knots: int,
    spline_degree: int,
    spline_include_bias: bool,
    spline_c: float,
    spline_max_iter: int,
    platt_c: float,
    platt_max_iter: int,
    beta_c: float,
    beta_max_iter: int,
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
                method="uncalibrated_logistic",
                phase="subset_comparison",
                y_true=y_test_sub,
                y_prob=p_test_sub,
                ece_bins=ece_bins,
                subset_fraction=frac,
            )
        )

        p_spline = _fit_spline(
            p_cal_sub,
            y_cal_sub,
            p_test_sub,
            n_knots=spline_n_knots,
            degree=spline_degree,
            include_bias=spline_include_bias,
            c=spline_c,
            max_iter=spline_max_iter,
        )
        rows.append(
            _metrics_row(
                method="spline_fixed",
                phase="subset_comparison",
                y_true=y_test_sub,
                y_prob=p_spline,
                ece_bins=ece_bins,
                spline_n_knots=spline_n_knots,
                subset_fraction=frac,
            )
        )

        p_platt = _fit_platt(
            p_cal_sub,
            y_cal_sub,
            p_test_sub,
            c=platt_c,
            max_iter=platt_max_iter,
        )
        rows.append(
            _metrics_row(
                method="platt",
                phase="subset_comparison",
                y_true=y_test_sub,
                y_prob=p_platt,
                ece_bins=ece_bins,
                subset_fraction=frac,
            )
        )

        p_isotonic = _fit_isotonic(
            p_cal_sub,
            y_cal_sub,
            p_test_sub,
        )
        rows.append(
            _metrics_row(
                method="isotonic",
                phase="subset_comparison",
                y_true=y_test_sub,
                y_prob=p_isotonic,
                ece_bins=ece_bins,
                subset_fraction=frac,
            )
        )

        p_beta = _fit_beta(
            p_cal_sub,
            y_cal_sub,
            p_test_sub,
            c=beta_c,
            max_iter=beta_max_iter,
        )
        rows.append(
            _metrics_row(
                method="beta",
                phase="subset_comparison",
                y_true=y_test_sub,
                y_prob=p_beta,
                ece_bins=ece_bins,
                subset_fraction=frac,
            )
        )
    return pd.DataFrame(rows)


def _run_cross_validated_train_test_stage(
    *,
    x: pd.DataFrame,
    y: NDArrayInt,
    cv_folds: int,
    random_state: int,
    spline_n_knots: int,
    spline_degree: int,
    spline_include_bias: bool,
    spline_c: float,
    spline_max_iter: int,
    platt_c: float,
    platt_max_iter: int,
    beta_c: float,
    beta_max_iter: int,
    haar_j_max: int,
    haar_lam: float,
    ece_bins: int,
) -> pd.DataFrame:
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2 for cross-validated train/test metrics.")

    outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
        x_train_full = x.iloc[train_idx].reset_index(drop=True)
        y_train_full = np.asarray(y[train_idx], dtype=int)
        x_test = x.iloc[test_idx].reset_index(drop=True)
        y_test = np.asarray(y[test_idx], dtype=int)

        x_train, x_cal, y_train, y_cal = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.25,
            stratify=y_train_full,
            random_state=random_state + fold_idx,
        )

        x_train = x_train.reset_index(drop=True)
        x_cal = x_cal.reset_index(drop=True)
        y_train = np.asarray(y_train, dtype=int)
        y_cal = np.asarray(y_cal, dtype=int)

        base_model = _build_base_model(x, random_state=random_state + fold_idx)
        base_model.fit(x_train, y_train)

        p_cal = base_model.predict_proba(x_cal)[:, 1]
        p_test = base_model.predict_proba(x_test)[:, 1]

        p_spline = _fit_spline(
            p_cal,
            y_cal,
            p_test,
            n_knots=spline_n_knots,
            degree=spline_degree,
            include_bias=spline_include_bias,
            c=spline_c,
            max_iter=spline_max_iter,
        )
        p_platt = _fit_platt(
            p_cal,
            y_cal,
            p_test,
            c=platt_c,
            max_iter=platt_max_iter,
        )
        p_isotonic = _fit_isotonic(p_cal, y_cal, p_test)
        p_beta = _fit_beta(
            p_cal,
            y_cal,
            p_test,
            c=beta_c,
            max_iter=beta_max_iter,
        )

        haar_calibrator = HaarMonotoneRidgeCalibrator(
            j_max=haar_j_max,
            lam=haar_lam,
            use_haar_norm=True,
            clip_probs=True,
        )
        haar_calibrator.fit(p_cal.reshape(-1, 1), y_cal)
        p_haar = haar_calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]

        fold_rows = [
            _metrics_row(
                method="uncalibrated_logistic",
                phase="cross_validated_test",
                y_true=y_test,
                y_prob=p_test,
                ece_bins=ece_bins,
            ),
            _metrics_row(
                method="spline_fixed",
                phase="cross_validated_test",
                y_true=y_test,
                y_prob=p_spline,
                ece_bins=ece_bins,
                spline_n_knots=spline_n_knots,
            ),
            _metrics_row(
                method="platt",
                phase="cross_validated_test",
                y_true=y_test,
                y_prob=p_platt,
                ece_bins=ece_bins,
            ),
            _metrics_row(
                method="isotonic",
                phase="cross_validated_test",
                y_true=y_test,
                y_prob=p_isotonic,
                ece_bins=ece_bins,
            ),
            _metrics_row(
                method="beta",
                phase="cross_validated_test",
                y_true=y_test,
                y_prob=p_beta,
                ece_bins=ece_bins,
            ),
            _metrics_row(
                method="haar_gridsearch_best",
                phase="cross_validated_test",
                y_true=y_test,
                y_prob=p_haar,
                ece_bins=ece_bins,
                j_max=haar_j_max,
                lam=haar_lam,
            ),
        ]
        for row in fold_rows:
            row["cv_fold"] = fold_idx
            row["train_samples"] = int(y_train.shape[0])
            row["calibration_samples"] = int(y_cal.shape[0])
            row["test_samples"] = int(y_test.shape[0])
        rows.extend(fold_rows)

    return _reorder_metric_columns(pd.DataFrame(rows))


def _select_lambda_decade(
    best_lambda: float,
    *,
    min_exp: int,
    max_exp: int,
) -> tuple[int, int]:
    """Select decade [10^k, 10^(k+1)] within configured exponent bounds."""
    if min_exp >= max_exp:
        raise ValueError("min_exp must be smaller than max_exp.")
    if best_lambda <= 0.0:
        raise ValueError("best_lambda must be positive.")

    decade_start = int(np.floor(np.log10(best_lambda)))
    decade_start = max(min_exp, min(decade_start, max_exp - 1))
    return decade_start, decade_start + 1


def _sorted_cv_results(grid: GridSearchCV) -> pd.DataFrame:
    return pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").reset_index(drop=True)


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
    prob_uncalibrated_logistic: NDArrayFloat,
    prob_spline: NDArrayFloat,
    prob_platt: NDArrayFloat,
    prob_isotonic: NDArrayFloat,
    prob_beta: NDArrayFloat,
    prob_gridsearch_best: NDArrayFloat,
) -> None:
    table = pd.DataFrame(
        {
            "y_true": y_true.astype(int),
            "prob_uncalibrated_logistic": prob_uncalibrated_logistic,
            "prob_spline": prob_spline,
            "prob_platt": prob_platt,
            "prob_isotonic": prob_isotonic,
            "prob_beta": prob_beta,
            "prob_gridsearch_best": prob_gridsearch_best,
        }
    )
    table.to_csv(output_path, index=False)


def _save_reliability_comparison_plot(
    *,
    y_true: NDArrayInt,
    output_path: Path,
    probs_by_method: dict[str, NDArrayFloat],
    n_bins: int,
    ece_bins: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.2, label="Perfect")

    style_map = [
        ("uncalibrated_logistic", "Uncalibrated logistic", "#4C72B0"),
        ("spline_fixed", "Spline", "#55A868"),
        ("platt", "Platt", "#C44E52"),
        ("isotonic", "Isotonic", "#CCB974"),
        ("beta", "Beta", "#64B5CD"),
        ("haar_gridsearch_best", "Haar best", "#8172B2"),
    ]
    for method, label, color in style_map:
        if method not in probs_by_method:
            continue
        probs = np.asarray(probs_by_method[method], dtype=float)
        _, points = reliability_points(y_true, probs, n_bins=n_bins)
        conf = points[:, 0]
        acc = points[:, 1]
        valid = (~np.isnan(conf)) & (~np.isnan(acc))
        ece = expected_calibration_error(y_true, probs, n_bins=ece_bins)
        brier = brier_score(y_true, probs)
        ax.plot(
            conf[valid],
            acc[valid],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{label} (Brier={brier:.3f}, ECE={ece:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability Comparison: All Estimators")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_reliability_panel_plot(
    *,
    y_true: NDArrayInt,
    output_path: Path,
    probs_by_method: dict[str, NDArrayFloat],
    n_bins: int,
    ece_bins: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    method_panels = [
        ("uncalibrated_logistic", "Uncalibrated logistic", "#4C72B0"),
        ("spline_fixed", "Spline", "#55A868"),
        ("platt", "Platt", "#C44E52"),
        ("isotonic", "Isotonic", "#CCB974"),
        ("beta", "Beta", "#64B5CD"),
        ("haar_gridsearch_best", "Haar best", "#8172B2"),
    ]

    for ax, (method, title, color) in zip(axes.ravel(), method_panels, strict=True):
        if method not in probs_by_method:
            ax.set_visible(False)
            continue
        probs = np.asarray(probs_by_method[method], dtype=float)
        _, points = reliability_points(y_true, probs, n_bins=n_bins)
        conf = points[:, 0]
        acc = points[:, 1]
        valid = (~np.isnan(conf)) & (~np.isnan(acc))
        ece = expected_calibration_error(y_true, probs, n_bins=ece_bins)
        brier = brier_score(y_true, probs)

        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.0)
        ax.plot(conf[valid], acc[valid], marker="o", linewidth=1.8, color=color)
        ax.set_title(f"{title} (Brier={brier:.3f}, ECE={ece:.3f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)

    fig.supxlabel("Mean predicted probability")
    fig.supylabel("Empirical positive rate")
    fig.suptitle("Reliability Comparison: Panel View", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_estimator_comparison_plot(
    *,
    output_plot_path: Path,
    output_table_path: Path,
    spline_calibrator: SplineBinaryCalibrator,
    platt_calibrator: PlattBinaryCalibrator,
    isotonic_calibrator: IsotonicBinaryCalibrator,
    beta_calibrator: BetaBinaryCalibrator,
    haar_best_calibrator: HaarMonotoneRidgeCalibrator,
    grid_points: int,
) -> tuple[Path, Path]:
    if grid_points < 50:
        raise ValueError("grid_points must be at least 50.")

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(0.0, 1.0, grid_points)
    curves = {
        "identity": x_grid,
        "spline_fixed": spline_calibrator.predict_proba(x_grid)[:, 1],
        "platt": platt_calibrator.predict_proba(x_grid)[:, 1],
        "isotonic": isotonic_calibrator.predict_proba(x_grid)[:, 1],
        "beta": beta_calibrator.predict_proba(x_grid)[:, 1],
        "haar_gridsearch_best": haar_best_calibrator.predict_proba(x_grid)[:, 1],
    }

    pd.DataFrame(
        {
            "raw_score": x_grid,
            "identity": curves["identity"],
            "spline_fixed": curves["spline_fixed"],
            "platt": curves["platt"],
            "isotonic": curves["isotonic"],
            "beta": curves["beta"],
            "haar_gridsearch_best": curves["haar_gridsearch_best"],
        }
    ).to_csv(output_table_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    spline_label = f"Spline fixed (n_knots={spline_calibrator.n_knots})"
    platt_label = f"Platt (C={platt_calibrator.c:g})"
    isotonic_label = "Isotonic"
    beta_label = f"Beta (C={beta_calibrator.c:g})"
    haar_best_label = (
        f"Haar best (j_max={haar_best_calibrator.j_max}, lam={haar_best_calibrator.lam:g})"
    )
    ax.plot(
        x_grid,
        curves["identity"],
        linestyle="--",
        color="#666666",
        linewidth=1.2,
        label="Identity",
    )
    ax.plot(x_grid, curves["spline_fixed"], color="#55A868", linewidth=2.0, label=spline_label)
    ax.plot(x_grid, curves["platt"], color="#C44E52", linewidth=2.0, label=platt_label)
    ax.plot(x_grid, curves["isotonic"], color="#CCB974", linewidth=2.0, label=isotonic_label)
    ax.plot(x_grid, curves["beta"], color="#64B5CD", linewidth=2.0, label=beta_label)
    ax.plot(
        x_grid,
        curves["haar_gridsearch_best"],
        color="#8172B2",
        linewidth=2.0,
        label=haar_best_label,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Raw model score")
    ax.set_ylabel("Calibrated probability")
    ax.set_title("Estimator Mapping Comparison")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_plot_path, dpi=180)
    plt.close(fig)
    return output_plot_path, output_table_path


def _reliability_summary_row(
    *,
    method: str,
    y_true: NDArrayInt,
    probs: NDArrayFloat,
    n_bins: int,
    ece_bins: int,
) -> dict[str, Any]:
    probs_arr = np.asarray(probs, dtype=float).ravel()
    _, points = reliability_points(y_true, probs_arr, n_bins=n_bins)
    conf = points[:, 0]
    acc = points[:, 1]
    valid = (~np.isnan(conf)) & (~np.isnan(acc))
    abs_gap = np.abs(acc[valid] - conf[valid]) if np.any(valid) else np.asarray([0.0], dtype=float)

    brier_cl, brier_rl = brier_calibration_refinement_loss(y_true, probs_arr, n_bins=ece_bins)
    log_cl, log_rl = log_loss_calibration_refinement_loss(y_true, probs_arr, n_bins=ece_bins)

    return {
        "method": method,
        "n_samples": int(y_true.shape[0]),
        "plot_bins": n_bins,
        "non_empty_bins": int(np.sum(valid)),
        "brier_score": brier_score(y_true, probs_arr),
        "brier_calibration_loss": brier_cl,
        "brier_refinement_loss": brier_rl,
        "ece": expected_calibration_error(y_true, probs_arr, n_bins=ece_bins),
        "log_loss": float(log_loss(y_true, np.clip(probs_arr, 1e-15, 1.0 - 1e-15))),
        "calibration_loss": log_cl,
        "refinement_loss": log_rl,
        "mean_abs_bin_gap": float(np.mean(abs_gap)),
        "max_abs_bin_gap": float(np.max(abs_gap)),
    }


def _save_graph_summaries(
    *,
    run_dir: Path,
    plots_dir: Path,
    y_true: NDArrayInt,
    probs_by_method: dict[str, NDArrayFloat],
    estimator_curves_path: Path,
    plot_bins: int,
    ece_bins: int,
) -> tuple[Path, Path, Path]:
    reliability_rows = [
        _reliability_summary_row(
            method=method,
            y_true=y_true,
            probs=probs,
            n_bins=plot_bins,
            ece_bins=ece_bins,
        )
        for method, probs in probs_by_method.items()
    ]
    reliability_df = pd.DataFrame(reliability_rows).sort_values(
        ["brier_score", "ece"],
        ascending=[True, True],
    )
    reliability_csv_path = run_dir / "graph_summary_reliability.csv"
    reliability_df.to_csv(reliability_csv_path, index=False)

    estimator_df = pd.read_csv(estimator_curves_path)
    identity = estimator_df["identity"].to_numpy(dtype=float)
    mapping_rows: list[dict[str, Any]] = []
    for column in estimator_df.columns:
        if column in {"raw_score", "identity"}:
            continue
        curve = estimator_df[column].to_numpy(dtype=float)
        delta = curve - identity
        mapping_rows.append(
            {
                "method": column,
                "mean_abs_shift_from_identity": float(np.mean(np.abs(delta))),
                "max_abs_shift_from_identity": float(np.max(np.abs(delta))),
                "mean_signed_shift_from_identity": float(np.mean(delta)),
                "is_monotone_non_decreasing": bool(np.all(np.diff(curve) >= -1e-8)),
                "curve_min": float(np.min(curve)),
                "curve_max": float(np.max(curve)),
            }
        )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(
        "mean_abs_shift_from_identity",
        ascending=False,
    )
    mapping_csv_path = run_dir / "graph_summary_estimator_mapping.csv"
    mapping_df.to_csv(mapping_csv_path, index=False)

    best_brier = reliability_df.loc[reliability_df["brier_score"].idxmin()]
    best_ece = reliability_df.loc[reliability_df["ece"].idxmin()]
    best_log_loss = reliability_df.loc[reliability_df["log_loss"].idxmin()]
    strongest_mapping = mapping_df.iloc[0]

    md_lines = [
        "# Graph Summaries",
        "",
        "## Reliability Graphs",
        "Metrics aggregated from per-estimator reliability curves (including all combined panels).",
        "",
        "```csv",
        reliability_df.to_csv(index=False, float_format="%.6f").strip(),
        "```",
        "",
        "Highlights:",
        f"- Lowest Brier: `{best_brier['method']}` ({best_brier['brier_score']:.6f})",
        f"- Lowest ECE: `{best_ece['method']}` ({best_ece['ece']:.6f})",
        f"- Lowest log-loss: `{best_log_loss['method']}` ({best_log_loss['log_loss']:.6f})",
        "",
        "## Estimator Mapping Graph",
        "Summary statistics from the `raw score -> calibrated probability` mapping plot.",
        "",
        "```csv",
        mapping_df.to_csv(index=False, float_format="%.6f").strip(),
        "```",
        "",
        "Highlights:",
        (
            f"- Largest mean absolute shift from identity: `{strongest_mapping['method']}` "
            f"({strongest_mapping['mean_abs_shift_from_identity']:.6f})"
        ),
        "",
        "## Generated Plot Files",
        f"- `{plots_dir / 'reliability_uncalibrated_logistic.png'}`",
        f"- `{plots_dir / 'reliability_spline_fixed.png'}`",
        f"- `{plots_dir / 'reliability_platt.png'}`",
        f"- `{plots_dir / 'reliability_isotonic.png'}`",
        f"- `{plots_dir / 'reliability_beta.png'}`",
        f"- `{plots_dir / 'reliability_haar_gridsearch_best.png'}`",
        f"- `{plots_dir / 'reliability_all_estimators_comparison.png'}`",
        f"- `{plots_dir / 'reliability_all_estimators_panel.png'}`",
        f"- `{plots_dir / 'estimator_all_calibrators_comparison.png'}`",
    ]
    summary_md_path = run_dir / "graph_summaries.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return reliability_csv_path, mapping_csv_path, summary_md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persistent TabArena calibration experiments.")
    parser.add_argument(
        "--dataset-preset",
        type=str,
        default=None,
        help="Dataset preset key (e.g. taiwanese-bankruptcy-prediction).",
    )
    parser.add_argument(
        "--list-dataset-presets",
        action="store_true",
        help="Print available dataset presets and exit.",
    )
    parser.add_argument("--dataset-name", type=str, default="blood-transfusion-service-center")
    parser.add_argument("--dataset-id", type=int, default=46913)
    parser.add_argument("--task-id", type=int, default=363621)
    parser.add_argument("--positive-label", type=str, default=None)
    parser.add_argument("--refresh-dataset", action="store_true")

    parser.add_argument("--subset-fracs", type=str, default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--spline-n-knots", type=int, default=5)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument("--spline-c", type=float, default=1.0)
    parser.add_argument("--spline-max-iter", type=int, default=500)
    parser.add_argument("--spline-include-bias", action="store_true")
    parser.add_argument("--platt-c", type=float, default=1.0)
    parser.add_argument("--platt-max-iter", type=int, default=500)
    parser.add_argument("--beta-c", type=float, default=1.0)
    parser.add_argument("--beta-max-iter", type=int, default=500)

    parser.add_argument("--grid-j-min", type=int, default=1)
    parser.add_argument("--grid-j-max", type=int, default=6)
    parser.add_argument("--lambda-min-exp", type=int, default=-6)
    parser.add_argument("--lambda-max-exp", type=int, default=0)
    parser.add_argument("--lambda-stage1-points", type=int, default=50)
    parser.add_argument("--lambda-stage2-points", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)

    parser.add_argument("--ece-bins", type=int, default=20)
    parser.add_argument("--plot-bins", type=int, default=20)
    parser.add_argument("--estimator-grid-points", type=int, default=1001)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--data-root", type=Path, default=Path("data/raw/tabarena"))
    parser.add_argument("--output-root", type=Path, default=Path("analysis/outputs/tabarena"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_dataset_presets:
        print(format_dataset_presets_table())
        return

    selected_dataset_preset: str | None = None
    if args.dataset_preset:
        preset = resolve_dataset_preset(args.dataset_preset)
        selected_dataset_preset = preset.key
        args.dataset_name = preset.dataset_name
        args.dataset_id = preset.dataset_id
        args.task_id = preset.task_id
        if args.positive_label is None:
            args.positive_label = preset.positive_label

    started_at = _utc_now_iso()

    if args.grid_j_min < 1:
        raise ValueError("--grid-j-min must be >= 1.")
    if args.grid_j_max < args.grid_j_min:
        raise ValueError("--grid-j-max must be >= --grid-j-min.")
    if args.lambda_max_exp <= args.lambda_min_exp:
        raise ValueError("--lambda-max-exp must be > --lambda-min-exp.")
    if args.lambda_stage1_points < 5:
        raise ValueError("--lambda-stage1-points must be >= 5.")
    if args.lambda_stage2_points < 5:
        raise ValueError("--lambda-stage2-points must be >= 5.")

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
    subset_df = _run_subset_stage(
        subset_fracs=subset_fracs,
        scores_cal=p_cal,
        y_cal=split.y_cal,
        scores_test=p_test,
        y_test=split.y_test,
        spline_n_knots=args.spline_n_knots,
        spline_degree=args.spline_degree,
        spline_include_bias=args.spline_include_bias,
        spline_c=args.spline_c,
        spline_max_iter=args.spline_max_iter,
        platt_c=args.platt_c,
        platt_max_iter=args.platt_max_iter,
        beta_c=args.beta_c,
        beta_max_iter=args.beta_max_iter,
        ece_bins=args.ece_bins,
        random_state=args.random_state,
    )
    subset_df = _reorder_metric_columns(subset_df)
    subset_df.to_csv(run_dir / "subset_results.csv", index=False)

    spline_calibrator = SplineBinaryCalibrator(
        n_knots=args.spline_n_knots,
        degree=args.spline_degree,
        include_bias=args.spline_include_bias,
        c=args.spline_c,
        max_iter=args.spline_max_iter,
    )
    spline_calibrator.fit(p_cal, split.y_cal)
    p_spline_full = spline_calibrator.predict_proba(p_test)[:, 1]

    platt_calibrator = PlattBinaryCalibrator(c=args.platt_c, max_iter=args.platt_max_iter)
    platt_calibrator.fit(p_cal, split.y_cal)
    p_platt_full = platt_calibrator.predict_proba(p_test)[:, 1]

    isotonic_calibrator = IsotonicBinaryCalibrator(out_of_bounds="clip")
    isotonic_calibrator.fit(p_cal, split.y_cal)
    p_isotonic_full = isotonic_calibrator.predict_proba(p_test)[:, 1]

    beta_calibrator = BetaBinaryCalibrator(c=args.beta_c, max_iter=args.beta_max_iter)
    beta_calibrator.fit(p_cal, split.y_cal)
    p_beta_full = beta_calibrator.predict_proba(p_test)[:, 1]

    j_candidates = list(range(args.grid_j_min, args.grid_j_max + 1))
    lambda_stage1_candidates = np.logspace(
        args.lambda_min_exp,
        args.lambda_max_exp,
        args.lambda_stage1_points,
    ).tolist()
    grid_stage1 = _run_gridsearch(
        scores_cal=p_cal,
        y_cal=split.y_cal,
        j_candidates=j_candidates,
        lam_candidates=lambda_stage1_candidates,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    stage1_results = _sorted_cv_results(grid_stage1)
    stage1_results.to_csv(run_dir / "lambda_stage1_gridsearch_cv_results.csv", index=False)
    # Backward-compatible alias for existing analysis references.
    stage1_results.to_csv(run_dir / "interval_scan_results.csv", index=False)

    best_stage1_lam = float(grid_stage1.best_params_["lam"])
    decade_start_exp, decade_end_exp = _select_lambda_decade(
        best_stage1_lam,
        min_exp=args.lambda_min_exp,
        max_exp=args.lambda_max_exp,
    )
    lambda_stage2_candidates = np.logspace(
        decade_start_exp,
        decade_end_exp,
        args.lambda_stage2_points,
    ).tolist()
    grid = _run_gridsearch(
        scores_cal=p_cal,
        y_cal=split.y_cal,
        j_candidates=j_candidates,
        lam_candidates=lambda_stage2_candidates,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    grid_results = _sorted_cv_results(grid)
    grid_results.to_csv(run_dir / "lambda_stage2_gridsearch_cv_results.csv", index=False)
    # Backward-compatible alias for existing analysis references.
    grid_results.to_csv(run_dir / "gridsearch_cv_results.csv", index=False)

    recommendation = {
        "lambda_search_strategy": "two_stage_logspace",
        "search_exponent_bounds": [args.lambda_min_exp, args.lambda_max_exp],
        "j_candidates": j_candidates,
        "stage1": {
            "lambda_points": args.lambda_stage1_points,
            "best_j_max": int(grid_stage1.best_params_["j_max"]),
            "best_lam": best_stage1_lam,
            "best_cv_neg_brier": float(grid_stage1.best_score_),
        },
        "selected_decade": {
            "start_exp": decade_start_exp,
            "end_exp": decade_end_exp,
            "lower_bound": float(10.0**decade_start_exp),
            "upper_bound": float(10.0**decade_end_exp),
        },
        "stage2": {
            "lambda_points": args.lambda_stage2_points,
            "best_j_max": int(grid.best_params_["j_max"]),
            "best_lam": float(grid.best_params_["lam"]),
            "best_cv_neg_brier": float(grid.best_score_),
        },
    }
    (run_dir / "recommended_ranges.json").write_text(
        json.dumps(recommendation, indent=2),
        encoding="utf-8",
    )

    best_calibrator = grid.best_estimator_
    p_grid_best = best_calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]
    cv_train_test_df = _run_cross_validated_train_test_stage(
        x=x,
        y=y,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        spline_n_knots=args.spline_n_knots,
        spline_degree=args.spline_degree,
        spline_include_bias=args.spline_include_bias,
        spline_c=args.spline_c,
        spline_max_iter=args.spline_max_iter,
        platt_c=args.platt_c,
        platt_max_iter=args.platt_max_iter,
        beta_c=args.beta_c,
        beta_max_iter=args.beta_max_iter,
        haar_j_max=int(grid.best_params_["j_max"]),
        haar_lam=float(grid.best_params_["lam"]),
        ece_bins=args.ece_bins,
    )
    cv_train_test_df.to_csv(run_dir / "cross_validated_train_test_metrics.csv", index=False)

    final_rows = [
        _metrics_row(
            method="uncalibrated_logistic",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_test,
            ece_bins=args.ece_bins,
        ),
        _metrics_row(
            method="spline_fixed",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_spline_full,
            ece_bins=args.ece_bins,
            spline_n_knots=args.spline_n_knots,
        ),
        _metrics_row(
            method="platt",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_platt_full,
            ece_bins=args.ece_bins,
        ),
        _metrics_row(
            method="isotonic",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_isotonic_full,
            ece_bins=args.ece_bins,
        ),
        _metrics_row(
            method="beta",
            phase="full_test",
            y_true=split.y_test,
            y_prob=p_beta_full,
            ece_bins=args.ece_bins,
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
    final_df = _reorder_metric_columns(pd.DataFrame(final_rows))
    final_df.to_csv(run_dir / "final_test_metrics.csv", index=False)

    _save_predictions_table(
        output_path=run_dir / "predictions_test.csv",
        y_true=split.y_test,
        prob_uncalibrated_logistic=p_test,
        prob_spline=p_spline_full,
        prob_platt=p_platt_full,
        prob_isotonic=p_isotonic_full,
        prob_beta=p_beta_full,
        prob_gridsearch_best=p_grid_best,
    )

    save_reliability_diagram(
        split.y_test,
        p_test,
        output_path=plots_dir / "reliability_uncalibrated_logistic.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title="Uncalibrated logistic reliability",
    )
    save_reliability_diagram(
        split.y_test,
        p_spline_full,
        output_path=plots_dir / "reliability_spline_fixed.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title=f"Spline reliability (n_knots={args.spline_n_knots})",
    )
    save_reliability_diagram(
        split.y_test,
        p_platt_full,
        output_path=plots_dir / "reliability_platt.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title=f"Platt reliability (C={args.platt_c:g})",
    )
    save_reliability_diagram(
        split.y_test,
        p_isotonic_full,
        output_path=plots_dir / "reliability_isotonic.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title="Isotonic reliability",
    )
    save_reliability_diagram(
        split.y_test,
        p_beta_full,
        output_path=plots_dir / "reliability_beta.png",
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
        title=f"Beta reliability (C={args.beta_c:g})",
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
    probs_for_comparison = {
        "uncalibrated_logistic": p_test,
        "spline_fixed": p_spline_full,
        "platt": p_platt_full,
        "isotonic": p_isotonic_full,
        "beta": p_beta_full,
        "haar_gridsearch_best": p_grid_best,
    }
    _save_reliability_comparison_plot(
        y_true=split.y_test,
        output_path=plots_dir / "reliability_all_estimators_comparison.png",
        probs_by_method=probs_for_comparison,
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
    )
    _save_reliability_panel_plot(
        y_true=split.y_test,
        output_path=plots_dir / "reliability_all_estimators_panel.png",
        probs_by_method=probs_for_comparison,
        n_bins=args.plot_bins,
        ece_bins=args.ece_bins,
    )
    estimator_plot_path, estimator_table_path = _save_estimator_comparison_plot(
        output_plot_path=plots_dir / "estimator_all_calibrators_comparison.png",
        output_table_path=run_dir / "estimator_curves.csv",
        spline_calibrator=spline_calibrator,
        platt_calibrator=platt_calibrator,
        isotonic_calibrator=isotonic_calibrator,
        beta_calibrator=beta_calibrator,
        haar_best_calibrator=best_calibrator,
        grid_points=args.estimator_grid_points,
    )
    graph_summary_reliability_path, graph_summary_mapping_path, graph_summary_md_path = (
        _save_graph_summaries(
            run_dir=run_dir,
            plots_dir=plots_dir,
            y_true=split.y_test,
            probs_by_method=probs_for_comparison,
            estimator_curves_path=estimator_table_path,
            plot_bins=args.plot_bins,
            ece_bins=args.ece_bins,
        )
    )

    dump(base_model, models_dir / "base_model.joblib")
    dump(spline_calibrator, models_dir / "spline_fixed_calibrator.joblib")
    dump(platt_calibrator, models_dir / "platt_calibrator.joblib")
    dump(isotonic_calibrator, models_dir / "isotonic_calibrator.joblib")
    dump(beta_calibrator, models_dir / "beta_calibrator.joblib")
    dump(best_calibrator, models_dir / "haar_gridsearch_best_calibrator.joblib")

    finished_at = _utc_now_iso()
    run_metadata = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "dataset": {
            "dataset_preset": selected_dataset_preset,
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
            "spline_n_knots": args.spline_n_knots,
            "spline_degree": args.spline_degree,
            "spline_include_bias": args.spline_include_bias,
            "spline_c": args.spline_c,
            "spline_max_iter": args.spline_max_iter,
            "platt_c": args.platt_c,
            "platt_max_iter": args.platt_max_iter,
            "beta_c": args.beta_c,
            "beta_max_iter": args.beta_max_iter,
            "subset_fracs": subset_fracs,
            "grid_j_min": args.grid_j_min,
            "grid_j_max": args.grid_j_max,
            "lambda_min_exp": args.lambda_min_exp,
            "lambda_max_exp": args.lambda_max_exp,
            "lambda_stage1_points": args.lambda_stage1_points,
            "lambda_stage2_points": args.lambda_stage2_points,
            "grid_cv_folds": args.cv_folds,
            "cross_validated_train_test_folds": args.cv_folds,
            "ece_bins": args.ece_bins,
            "plot_bins": args.plot_bins,
            "estimator_grid_points": args.estimator_grid_points,
        },
        "two_stage_lambda_search": recommendation,
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
    print(f"- {run_dir / 'subset_results.csv'}")
    print(f"- {run_dir / 'lambda_stage1_gridsearch_cv_results.csv'}")
    print(f"- {run_dir / 'lambda_stage2_gridsearch_cv_results.csv'}")
    print(f"- {run_dir / 'interval_scan_results.csv'}")
    print(f"- {run_dir / 'gridsearch_cv_results.csv'}")
    print(f"- {run_dir / 'cross_validated_train_test_metrics.csv'}")
    print(f"- {run_dir / 'final_test_metrics.csv'}")
    print(f"- {run_dir / 'predictions_test.csv'}")
    print(f"- {run_dir / 'estimator_curves.csv'}")
    print(f"- {graph_summary_reliability_path}")
    print(f"- {graph_summary_mapping_path}")
    print(f"- {graph_summary_md_path}")
    print(f"- {run_dir / 'recommended_ranges.json'}")
    print(f"- {run_dir / 'run_metadata.json'}")
    print(f"- {plots_dir / 'reliability_uncalibrated_logistic.png'}")
    print(f"- {plots_dir / 'reliability_spline_fixed.png'}")
    print(f"- {plots_dir / 'reliability_platt.png'}")
    print(f"- {plots_dir / 'reliability_isotonic.png'}")
    print(f"- {plots_dir / 'reliability_beta.png'}")
    print(f"- {plots_dir / 'reliability_haar_gridsearch_best.png'}")
    print(f"- {plots_dir / 'reliability_all_estimators_comparison.png'}")
    print(f"- {plots_dir / 'reliability_all_estimators_panel.png'}")
    print(f"- {estimator_plot_path}")
    print(f"- {models_dir / 'base_model.joblib'}")
    print(f"- {models_dir / 'spline_fixed_calibrator.joblib'}")
    print(f"- {models_dir / 'platt_calibrator.joblib'}")
    print(f"- {models_dir / 'isotonic_calibrator.joblib'}")
    print(f"- {models_dir / 'beta_calibrator.joblib'}")
    print(f"- {models_dir / 'haar_gridsearch_best_calibrator.joblib'}")


if __name__ == "__main__":
    main()
