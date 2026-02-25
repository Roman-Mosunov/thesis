from pathlib import Path

import numpy as np
import pytest
from splinecal.plotting import (
    plot_reliability_diagram,
    reliability_points,
    save_reliability_diagram,
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


def test_save_reliability_diagram_writes_file(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.7])
    output = tmp_path / "reliability.png"

    saved = save_reliability_diagram(y_true, y_prob, output_path=output, n_bins=4)
    assert saved.exists()
