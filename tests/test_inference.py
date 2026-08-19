import numpy as np
import pandas as pd

from src.inference import run_inference


class IdentityPreprocessor:
    def transform(self, frame):
        return frame[["feature"]]


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def test_run_inference_bounds_and_monotonizes_poverty_rates():
    test_frame = pd.DataFrame(
        {
            "survey_id": [400000, 400000],
            "hhid": [1, 2],
            "weight": [1.0, 1.0],
            "feature": [0.0, 1.0],
        }
    )
    thresholds = np.array([3.17, 3.94])

    consumption, poverty = run_inference(
        test_frame,
        IdentityPreprocessor(),
        models={100000: ConstantModel(0.0)},
        poverty_models={
            3.17: ConstantModel(-0.2),
            3.94: ConstantModel(1.2),
        },
        train_cdfs={100000: np.array([0.1, 0.9])},
        survey_log_means={100000: 0.0},
        poverty_thresholds=thresholds,
    )

    values = poverty.iloc[0, 1:].to_numpy(dtype=float)
    assert len(consumption) == len(test_frame)
    assert (values >= 0.0).all()
    assert (values <= 1.0).all()
    assert (np.diff(values) >= 0.0).all()
