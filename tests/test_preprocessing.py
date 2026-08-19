import pandas as pd

from src.preprocessing import FeaturePreprocessor


def _frame(dwelling_type="House"):
    return pd.DataFrame(
        {
            "survey_id": [100000],
            "hhid": [1],
            "weight": [1.0],
            "hsize": [4],
            "educ_max": ["Complete Secondary Education"],
            "dweltyp": [dwelling_type],
            "water_source": ["Piped"],
            "sanitation_source": ["Flush"],
            "sector1d": ["Services"],
            "any_nonagric": ["Yes"],
            "male": ["Yes"],
            "owner": ["Owner"],
            "water": ["Access"],
            "toilet": ["Access"],
            "sewer": ["Access"],
            "elect": ["Access"],
            "employed": ["Employed"],
            "urban": ["Urban"],
            "consumed_rice": ["Yes"],
            "consumed_meat": ["No"],
        }
    )


def test_transform_adds_engineered_features():
    preprocessor = FeaturePreprocessor().fit(_frame())
    transformed = preprocessor.transform(_frame())

    assert transformed.loc[0, "consumed_yes_count"] == 1
    assert transformed.loc[0, "log_hsize"] > 0


def test_transform_maps_unseen_categories_to_fallback():
    preprocessor = FeaturePreprocessor().fit(_frame())
    transformed = preprocessor.transform(_frame(dwelling_type="Unseen dwelling"))

    expected = list(preprocessor.label_encoders["dweltyp"].classes_).index("missing")
    assert transformed.loc[0, "dweltyp"] == expected
