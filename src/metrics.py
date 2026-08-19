from __future__ import annotations

import numpy as np


def mean_absolute_percentage_error(y_true, y_pred, *, epsilon=1e-9):
    """Return an unweighted MAPE as a fraction, where lower is better."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(actual), epsilon)
    return float(np.mean(np.abs(actual - predicted) / denominator))


def poverty_rate_weights(n_thresholds):
    """Competition weights for poverty thresholds at percentiles 5% to 95%."""
    percentiles = np.arange(1, n_thresholds + 1, dtype=float) * 0.05
    return 1.0 - np.abs(0.4 - percentiles)


def weighted_poverty_mape(y_true, y_pred, *, epsilon=1e-9):
    """Return the weighted poverty-distribution MAPE used by the challenge."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Poverty-rate arrays must have the same shape")
    if actual.ndim != 1:
        raise ValueError("Poverty-rate arrays must be one-dimensional")

    weights = poverty_rate_weights(len(actual))
    percentage_errors = np.abs(actual - predicted) / np.maximum(
        np.abs(actual), epsilon
    )
    return float(np.average(percentage_errors, weights=weights))


def blended_score(consumption_mape, poverty_mape):
    """Return the official 10/90 blended error score; lower is better."""
    return 10.0 * float(consumption_mape) + 90.0 * float(poverty_mape)
