import numpy as np

from splinecal.splines import as_probability_vector


def test_as_probability_vector_clips_values() -> None:
    x = np.array([-0.2, 0.4, 1.4])
    out = as_probability_vector(x)
    assert np.all(out >= 1e-6)
    assert np.all(out <= 1 - 1e-6)


def test_as_probability_vector_uses_last_column() -> None:
    x = np.array([[0.1, 0.9], [0.3, 0.7]])
    out = as_probability_vector(x)
    assert np.allclose(out, np.array([0.9, 0.7]))
