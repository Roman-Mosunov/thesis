from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import brier_score, expected_calibration_error

PHAT_THEME_FIGURE_FACE = "#2C323C"
PHAT_THEME_AXES_FACE = "#313842"
PHAT_THEME_TEXT = "#F2F4F8"
PHAT_THEME_MUTED = "#C0C7D1"
PHAT_THEME_GRID = "#D6DDE6"
PHAT_THEME_METRICS_FACE = "#232931"


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


def binned_calibration_curve(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 20,
    min_bin_count: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Return a Metaculus-style binned calibration curve on a fixed x-grid."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be at least 1.")

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    observed = np.full(n_bins, np.nan)
    mean_pred = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=True)
    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = int(np.sum(mask))
        if counts[b] >= min_bin_count:
            observed[b] = y_true_arr[mask].mean()
            mean_pred[b] = y_prob_arr[mask].mean()

    return mids, observed, counts, mean_pred


def _apply_phat_dark_theme(ax: Any, *, grid_axis: str = "both") -> None:
    ax.set_facecolor(PHAT_THEME_AXES_FACE)
    for spine in ax.spines.values():
        spine.set_color(PHAT_THEME_MUTED)
    ax.tick_params(colors=PHAT_THEME_TEXT)
    ax.xaxis.label.set_color(PHAT_THEME_TEXT)
    ax.yaxis.label.set_color(PHAT_THEME_TEXT)
    ax.title.set_color(PHAT_THEME_TEXT)
    ax.grid(color=PHAT_THEME_GRID, alpha=0.28, linewidth=0.8, axis=grid_axis)


def _apply_phat_dark_legend(legend: Any | None) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_edgecolor("none")
    for text in legend.get_texts():
        text.set_color(PHAT_THEME_TEXT)


def _default_smooth_bandwidth(n_samples: int) -> float:
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to smooth a calibration curve.")
    bandwidth = 1.06 * 0.25 * (float(n_samples) ** (-1.0 / 5.0))
    return float(np.clip(bandwidth, 0.02, 0.25))


def smoothed_calibration_curve(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    grid_points: int = 200,
    bandwidth: float | None = None,
    min_effective_n: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a smooth estimate of p_hat(p) on a common probability grid."""
    if grid_points < 10:
        raise ValueError("grid_points must be at least 10.")
    if min_effective_n <= 0.0:
        raise ValueError("min_effective_n must be positive.")

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    bandwidth_value = _default_smooth_bandwidth(y_prob_arr.size) if bandwidth is None else float(
        bandwidth
    )
    if not np.isfinite(bandwidth_value) or bandwidth_value <= 0.0:
        raise ValueError("bandwidth must be a positive finite number.")

    grid = np.linspace(0.0, 1.0, grid_points)
    scaled_distance = (grid[:, None] - y_prob_arr[None, :]) / bandwidth_value
    weights = np.exp(-0.5 * np.square(scaled_distance))
    total_weight = weights.sum(axis=1)
    weighted_positive = weights @ y_true_arr
    p_hat = np.divide(
        weighted_positive,
        total_weight,
        out=np.full(grid_points, np.nan),
        where=total_weight > 0.0,
    )

    squared_weight = np.square(weights).sum(axis=1)
    effective_n = np.divide(
        np.square(total_weight),
        squared_weight,
        out=np.zeros(grid_points, dtype=float),
        where=squared_weight > 0.0,
    )
    p_hat[effective_n < min_effective_n] = np.nan
    return grid, p_hat, effective_n


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

    created_figure = ax is None
    if created_figure:
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

    ax_main.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
    )
    ax_main.set_title(title or f"Reliability Diagram ({n_bins} bins)")
    if created_figure:
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return fig, ax_main


def plot_smoothed_calibration_diagram(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    grid_points: int = 200,
    hist_bins: int = 20,
    ece_bins: int = 10,
    bandwidth: float | None = None,
    min_effective_n: float = 5.0,
    title: str | None = None,
    show_histogram: bool = True,
    show_metrics: bool = True,
    curve_color: str = "#CCB974",
) -> tuple[Any, Any, Any | None]:
    """Plot a smooth p_hat(p) curve with an optional p_hat histogram panel."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("matplotlib is required for plotting reliability diagrams.") from exc

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    grid, p_hat, effective_n = smoothed_calibration_curve(
        y_true_arr,
        y_prob_arr,
        grid_points=grid_points,
        bandwidth=bandwidth,
        min_effective_n=min_effective_n,
    )
    valid = ~np.isnan(p_hat)

    if show_histogram:
        fig, (ax_curve, ax_hist) = plt.subplots(
            2,
            1,
            figsize=(7, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0]},
        )
    else:
        fig, ax_curve = plt.subplots(figsize=(7, 6))
        ax_hist = None

    ax_curve.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="Perfect",
    )
    ax_curve.plot(
        grid[valid],
        p_hat[valid],
        color=curve_color,
        linewidth=2.2,
        label="Smoothed p_hat",
    )
    ax_curve.set_ylabel("Smoothed empirical positive rate")
    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.grid(alpha=0.3)

    if show_histogram and ax_hist is not None:
        mids_hist, freqs = reliability_bin_frequencies(y_prob_arr, n_bins=hist_bins)
        ax_hist.bar(
            mids_hist,
            freqs,
            width=1.0 / hist_bins,
            alpha=0.35,
            color=curve_color,
            edgecolor="white",
            linewidth=0.4,
        )
        ax_hist.set_ylabel("Share")
        ax_hist.set_ylim(0.0, max(0.05, float(freqs.max()) * 1.25))
        ax_hist.grid(alpha=0.2, axis="y")
        ax_hist.set_xlabel("Predicted probability p_hat")
    else:
        ax_curve.set_xlabel("Predicted probability p_hat")

    if show_metrics:
        ece = expected_calibration_error(y_true_arr.astype(int), y_prob_arr, n_bins=ece_bins)
        brier = brier_score(y_true_arr.astype(int), y_prob_arr)
        peak_effective_n = int(np.max(effective_n)) if effective_n.size else 0
        text = (
            f"ECE ({ece_bins} bins): {ece:.4f}\n"
            f"Brier: {brier:.4f}\n"
            f"Peak local n_eff: {peak_effective_n}"
        )
        ax_curve.text(
            0.03,
            0.97,
            text,
            transform=ax_curve.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
    )
    ax_curve.set_title(title or "Smoothed Calibration Curve")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return fig, ax_curve, ax_hist


def plot_phat_calibration_diagram(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    n_bins: int = 20,
    hist_bins: int = 20,
    ece_bins: int = 10,
    title: str | None = None,
    curve_label: str = "Calibration",
    curve_color: str = "#FFB000",
    min_bin_count: int = 1,
    show_histogram: bool = True,
    show_metrics: bool = True,
) -> tuple[Any, Any, Any | None]:
    """Plot a Metaculus-style p_hat calibration curve with a histogram below."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("matplotlib is required for plotting reliability diagrams.") from exc

    y_true_arr, y_prob_arr = _validate_binary_inputs(y_true, y_prob)
    mids, observed, counts, _ = binned_calibration_curve(
        y_true_arr,
        y_prob_arr,
        n_bins=n_bins,
        min_bin_count=min_bin_count,
    )
    valid = (~np.isnan(observed)) & (counts > 0)

    if show_histogram:
        fig, (ax_curve, ax_hist) = plt.subplots(
            2,
            1,
            figsize=(7.5, 7.0),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0]},
        )
    else:
        fig, ax_curve = plt.subplots(figsize=(7.5, 5.8))
        ax_hist = None

    fig.patch.set_facecolor(PHAT_THEME_FIGURE_FACE)
    _apply_phat_dark_theme(ax_curve)
    if ax_hist is not None:
        _apply_phat_dark_theme(ax_hist, grid_axis="y")

    ax_curve.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#AAB3BE",
        linewidth=1.4,
        label="Perfect calibration",
    )
    ax_curve.plot(
        mids[valid],
        observed[valid],
        color=curve_color,
        linewidth=1.5,
        marker="D",
        markersize=5.4,
        markerfacecolor=curve_color,
        markeredgewidth=0.0,
        label=curve_label,
    )
    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.set_ylabel("Fraction resolved yes")
    ax_curve.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_curve.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    if show_histogram and ax_hist is not None:
        mids_hist, freqs = reliability_bin_frequencies(y_prob_arr, n_bins=hist_bins)
        ax_hist.bar(
            mids_hist,
            freqs,
            width=1.0 / hist_bins,
            color=curve_color,
            alpha=0.32,
            edgecolor=PHAT_THEME_MUTED,
            linewidth=0.4,
        )
        ax_hist.set_ylabel("Share")
        ax_hist.set_xlabel("Predictions (p_hat)")
        ax_hist.set_ylim(0.0, max(0.05, float(freqs.max()) * 1.25))
        ax_hist.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    else:
        ax_curve.set_xlabel("Predictions (p_hat)")

    if show_metrics:
        ece = expected_calibration_error(y_true_arr.astype(int), y_prob_arr, n_bins=ece_bins)
        brier = brier_score(y_true_arr.astype(int), y_prob_arr)
        text = f"ECE ({ece_bins} bins): {ece:.4f}\nBrier: {brier:.4f}"
        ax_curve.text(
            0.03,
            0.97,
            text,
            transform=ax_curve.transAxes,
            ha="left",
            va="top",
            color=PHAT_THEME_TEXT,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": PHAT_THEME_METRICS_FACE,
                "alpha": 0.92,
                "edgecolor": PHAT_THEME_MUTED,
            },
        )

    legend = fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
    )
    _apply_phat_dark_legend(legend)
    ax_curve.set_title(
        title or "Calibration Curve",
        loc="left",
        pad=10,
        fontweight="bold",
        color=PHAT_THEME_TEXT,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return fig, ax_curve, ax_hist


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


def save_smoothed_calibration_diagram(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    output_path: str | Path,
    grid_points: int = 200,
    hist_bins: int = 20,
    ece_bins: int = 10,
    bandwidth: float | None = None,
    min_effective_n: float = 5.0,
    title: str | None = None,
    show_histogram: bool = True,
    show_metrics: bool = True,
    curve_color: str = "#CCB974",
    dpi: int = 160,
) -> Path:
    """Generate and save a smooth calibration curve with a p_hat histogram."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _, _ = plot_smoothed_calibration_diagram(
        y_true=y_true,
        y_prob=y_prob,
        grid_points=grid_points,
        hist_bins=hist_bins,
        ece_bins=ece_bins,
        bandwidth=bandwidth,
        min_effective_n=min_effective_n,
        title=title,
        show_histogram=show_histogram,
        show_metrics=show_metrics,
        curve_color=curve_color,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.clf()
    return output


def save_phat_calibration_diagram(
    y_true: NDArray[np.float64] | NDArray[np.int64],
    y_prob: NDArray[np.float64],
    *,
    output_path: str | Path,
    n_bins: int = 20,
    hist_bins: int = 20,
    ece_bins: int = 10,
    title: str | None = None,
    curve_label: str = "Calibration",
    curve_color: str = "#FFB000",
    min_bin_count: int = 1,
    show_histogram: bool = True,
    show_metrics: bool = True,
    dpi: int = 160,
) -> Path:
    """Generate and save a Metaculus-style p_hat calibration plot."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _, _ = plot_phat_calibration_diagram(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        hist_bins=hist_bins,
        ece_bins=ece_bins,
        title=title,
        curve_label=curve_label,
        curve_color=curve_color,
        min_bin_count=min_bin_count,
        show_histogram=show_histogram,
        show_metrics=show_metrics,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.clf()
    return output
