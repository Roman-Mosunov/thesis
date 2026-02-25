from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import brier_score, expected_calibration_error


def _validate_binary_inputs(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_prob_arr.ndim != 1:
        raise ValueError("y_prob must be a 1D array.")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")
    if np.any(~np.isfinite(y_true_arr)) or np.any(~np.isfinite(y_prob_arr)):
        raise ValueError("y_true and y_prob must be finite.")
    if np.any((y_true_arr != 0.0) & (y_true_arr != 1.0)):
        raise ValueError("y_true must contain only binary labels {0, 1}.")
    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("y_prob must contain probabilities in [0, 1].")

    return y_true_arr, y_prob_arr


def reliability_points(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-bin confidence and accuracy points for reliability diagrams."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)

    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            conf[b] = y_prob_arr[mask].mean()
            acc[b] = y_true_arr[mask].mean()

    return mids, np.column_stack((conf, acc))


def reliability_bin_frequencies(
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return bin midpoints and normalized frequency per bin."""
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if y_prob_arr.ndim != 1:
        raise ValueError("y_prob must be a 1D array.")
    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("y_prob must contain probabilities in [0, 1].")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(y_prob_arr, bins=bins)
    freqs = counts / y_prob_arr.size
    return mids, freqs


def plot_reliability_diagram(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 10,
    ece_bins: int = 10,
    title: str | None = None,
    show_histogram: bool = True,
    show_metrics: bool = True,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a reliability diagram for binary predictions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("matplotlib is required for plotting reliability diagrams.") from exc

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    mids, points = reliability_points(y_true_arr, y_prob_arr, n_bins=n_bins)
    conf = points[:, 0]
    acc = points[:, 1]
    valid = ~np.isnan(conf) & ~np.isnan(acc)

    if ax is None:
        fig, ax_main = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
        ax_main = ax

    ax_main.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="Perfect",
    )
    ax_main.plot(
        conf[valid],
        acc[valid],
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="Model",
    )
    ax_main.set_xlabel("Mean predicted probability")
    ax_main.set_ylabel("Empirical positive rate")
    ax_main.set_xlim(0.0, 1.0)
    ax_main.set_ylim(0.0, 1.0)
    ax_main.grid(alpha=0.3)

    if show_histogram:
        mids_hist, freqs = reliability_bin_frequencies(y_prob_arr, n_bins=n_bins)
        ax_hist = ax_main.twinx()
        ax_hist.bar(
            mids_hist,
            freqs,
            width=1.0 / n_bins,
            alpha=0.2,
            color="tab:gray",
            edgecolor="none",
            label="Bin frequency",
        )
        ax_hist.set_ylabel("Bin frequency")
        ax_hist.set_ylim(0.0, max(0.05, float(freqs.max()) * 1.25))

    if show_metrics:
        ece = expected_calibration_error(y_true_arr.astype(int), y_prob_arr, n_bins=ece_bins)
        brier = brier_score(y_true_arr.astype(int), y_prob_arr)
        text = f"ECE ({ece_bins} bins): {ece:.4f}\nBrier: {brier:.4f}"
        ax_main.text(
            0.03,
            0.97,
            text,
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )

    ax_main.legend(loc="lower right")
    ax_main.set_title(title or f"Reliability Diagram ({n_bins} bins)")
    return fig, ax_main


def save_reliability_diagram(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    output_path: str | Path,
    n_bins: int = 10,
    ece_bins: int = 10,
    title: str | None = None,
    show_histogram: bool = True,
    show_metrics: bool = True,
    dpi: int = 160,
) -> Path:
    """Generate and save a reliability diagram to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_reliability_diagram(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        ece_bins=ece_bins,
        title=title,
        show_histogram=show_histogram,
        show_metrics=show_metrics,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.clf()
    return output
