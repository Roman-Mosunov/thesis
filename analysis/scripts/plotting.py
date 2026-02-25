"""Generate a reliability diagram from predictions or a synthetic demo run.

Examples:
    uv run python analysis/scripts/plotting.py --model spline --n-bins 20
    uv run python analysis/scripts/plotting.py \
        --predictions-csv analysis/outputs/tabarena/.../predictions_test.csv \
        --y-prob-col prob_gridsearch_best
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from splinecal import (
    HaarMonotoneRidgeCalibrator,
    SplineBinaryCalibrator,
    brier_score,
    expected_calibration_error,
    save_reliability_diagram,
)


def _read_predictions_csv(
    path: Path,
    y_true_col: str,
    y_prob_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[float] = []
    y_prob: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header row.")
        if y_true_col not in reader.fieldnames:
            raise ValueError(f"Column '{y_true_col}' not found in {path}.")
        if y_prob_col not in reader.fieldnames:
            raise ValueError(f"Column '{y_prob_col}' not found in {path}.")
        for row in reader:
            y_true.append(float(row[y_true_col]))
            y_prob.append(float(row[y_prob_col]))
    return np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)


def _demo_predictions(
    model: str,
    *,
    n_samples: int,
    seed: int,
    n_knots: int,
    j_max: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.01, 0.99, size=n_samples)

    if model == "spline":
        y_true = (rng.uniform(size=n_samples) < scores).astype(int)
        calibrator = SplineBinaryCalibrator(n_knots=n_knots)
    else:
        true_prob = 0.05 + 0.9 * (scores**1.8)
        y_true = (rng.uniform(size=n_samples) < true_prob).astype(int)
        calibrator = HaarMonotoneRidgeCalibrator(j_max=j_max, lam=lam)

    calibrator.fit(scores, y_true)
    y_prob = calibrator.predict_proba(scores)[:, 1]
    return y_true, y_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reliability diagrams for calibration experiments."
    )
    parser.add_argument("--predictions-csv", type=Path, default=None, help="CSV with predictions.")
    parser.add_argument(
        "--y-true-col",
        type=str,
        default="y_true",
        help="Ground-truth column name.",
    )
    parser.add_argument("--y-prob-col", type=str, default="y_prob", help="Probability column name.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/outputs/reliability_diagram.png"),
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=15, help="Bins for reliability diagram.")
    parser.add_argument("--ece-bins", type=int, default=15, help="Bins for ECE metric text.")
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Disable probability histogram overlay.",
    )

    parser.add_argument("--model", choices=["spline", "haar"], default="spline")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-knots", type=int, default=5)
    parser.add_argument("--j-max", type=int, default=6)
    parser.add_argument("--lam", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.predictions_csv is not None:
        y_true, y_prob = _read_predictions_csv(
            args.predictions_csv,
            args.y_true_col,
            args.y_prob_col,
        )
    else:
        y_true, y_prob = _demo_predictions(
            args.model,
            n_samples=args.n_samples,
            seed=args.seed,
            n_knots=args.n_knots,
            j_max=args.j_max,
            lam=args.lam,
        )

    output = save_reliability_diagram(
        y_true,
        y_prob,
        output_path=args.output,
        n_bins=args.n_bins,
        ece_bins=args.ece_bins,
        title=args.title,
        show_histogram=not args.no_histogram,
    )

    ece = expected_calibration_error(y_true.astype(int), y_prob, n_bins=args.ece_bins)
    brier = brier_score(y_true.astype(int), y_prob)
    print(f"Saved reliability diagram: {output}")
    print(f"ECE ({args.ece_bins} bins): {ece:.4f}")
    print(f"Brier: {brier:.4f}")


if __name__ == "__main__":
    main()
