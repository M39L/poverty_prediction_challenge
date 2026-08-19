import numpy as np
import pytest

from src.metrics import (
    blended_score,
    mean_absolute_percentage_error,
    poverty_rate_weights,
    weighted_poverty_mape,
)


def test_consumption_mape():
    result = mean_absolute_percentage_error([10.0, 20.0], [9.0, 22.0])
    assert result == pytest.approx(0.1)


def test_poverty_weights_peak_at_40th_percentile():
    weights = poverty_rate_weights(19)
    assert weights.shape == (19,)
    assert weights[7] == pytest.approx(1.0)
    assert np.all(weights > 0)


def test_weighted_poverty_mape_is_zero_for_exact_prediction():
    actual = np.linspace(0.05, 0.95, 19)
    assert weighted_poverty_mape(actual, actual) == pytest.approx(0.0)


def test_blended_score_uses_competition_weights():
    assert blended_score(0.2, 0.1) == pytest.approx(11.0)
