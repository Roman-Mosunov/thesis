from pathlib import Path

import numpy as np
import pytest
from splinecal.plotting import (
    binned_calibration_curve,
    plot_phat_calibration_diagram,
    plot_reliability_diagram,
    plot_smoothed_calibration_diagram,
    reliability_points,
    save_phat_calibration_diagram,
    save_reliability_diagram,
    save_smoothed_calibration_diagram,
    smoothed_calibration_curve,
)


def test_reliability_points_shape() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.75, 0.2])

    mids, points = reliability_points(y_true, y_prob, n_bins=5)

    assert mids.shape == (5,)
    assert points.shape == (5, 2)


def test_plot_reliability_diagram_builds_figure() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.7, 0.85, 0.8, 0.4, 0.9, 0.2])

    fig, ax = plot_reliability_diagram(y_true, y_prob, n_bins=4, ece_bins=4)
    assert fig is not None
    assert ax is not None


def test_smoothed_calibration_curve_shape() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.7, 0.85, 0.8, 0.4, 0.9, 0.2])

    grid, p_hat, effective_n = smoothed_calibration_curve(
        y_true,
        y_prob,
        grid_points=40,
        min_effective_n=1.0,
    )

    assert grid.shape == (40,)
    assert p_hat.shape == (40,)
    assert effective_n.shape == (40,)
    assert np.all((grid >= 0.0) & (grid <= 1.0))
    assert np.nanmin(p_hat) >= 0.0
    assert np.nanmax(p_hat) <= 1.0


def test_binned_calibration_curve_shape() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.7, 0.85, 0.8, 0.4, 0.9, 0.2])

    mids, observed, counts, mean_pred = binned_calibration_curve(y_true, y_prob, n_bins=5)

    assert mids.shape == (5,)
    assert observed.shape == (5,)
    assert counts.shape == (5,)
    assert mean_pred.shape == (5,)
    assert counts.sum() == y_true.shape[0]


def test_plot_smoothed_calibration_diagram_builds_figure() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.7, 0.85, 0.8, 0.4, 0.9, 0.2])

    fig, ax_curve, ax_hist = plot_smoothed_calibration_diagram(
        y_true,
        y_prob,
        grid_points=40,
        hist_bins=6,
        min_effective_n=1.0,
    )
    assert fig is not None
    assert ax_curve is not None
    assert ax_hist is not None


def test_plot_phat_calibration_diagram_builds_figure() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.15, 0.7, 0.85, 0.8, 0.4, 0.9, 0.2])

    fig, ax_curve, ax_hist = plot_phat_calibration_diagram(
        y_true,
        y_prob,
        n_bins=5,
        hist_bins=5,
    )
    assert fig is not None
    assert ax_curve is not None
    assert ax_hist is not None


def test_save_reliability_diagram_writes_file(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.7])
    output = tmp_path / "reliability.png"

    saved = save_reliability_diagram(y_true, y_prob, output_path=output, n_bins=4)
    assert saved.exists()


def test_save_smoothed_calibration_diagram_writes_file(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.7])
    output = tmp_path / "smoothed_calibration.png"

    saved = save_smoothed_calibration_diagram(
        y_true,
        y_prob,
        output_path=output,
        grid_points=30,
        hist_bins=4,
        min_effective_n=1.0,
    )
    assert saved.exists()


def test_save_phat_calibration_diagram_writes_file(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.7])
    output = tmp_path / "phat_calibration.png"

    saved = save_phat_calibration_diagram(
        y_true,
        y_prob,
        output_path=output,
        n_bins=4,
        hist_bins=4,
    )
    assert saved.exists()
