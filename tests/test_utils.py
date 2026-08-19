import numpy as np
import pandas as pd

from src.utils import get_gt_cdf, softmax


def test_softmax_is_stable_and_normalized():
    values = softmax(np.array([1000.0, 1001.0, 1002.0]))

    assert np.isfinite(values).all()
    assert np.isclose(values.sum(), 1.0)
    assert values.argmax() == 2


def test_get_gt_cdf_preserves_threshold_order():
    frame = pd.DataFrame(
        {"survey_id": [100000], "low": [0.2], "high": [0.8]}
    )

    result = get_gt_cdf(
        frame,
        survey_id=100000,
        poverty_map={1.0: "low", 2.0: "high"},
        poverty_thresholds=np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(result, np.array([0.2, 0.8]))
