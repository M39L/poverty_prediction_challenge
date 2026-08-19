from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.inference import run_inference
from src.metrics import (
    blended_score,
    mean_absolute_percentage_error,
    weighted_poverty_mape,
)
from src.preprocessing import FeaturePreprocessor
from src.trainer import train_consumption_models, train_poverty_models
from src.utils import get_gt_cdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leakage-safe leave-one-survey-out validation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Trees per LightGBM model. Use 50 for a quick smoke test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/validation_results.csv"),
    )
    return parser.parse_args()


def load_training_data(data_dir):
    features = pd.read_csv(data_dir / "train_hh_features.csv")
    targets = pd.read_csv(data_dir / "train_hh_gt.csv")
    rates = pd.read_csv(data_dir / "train_rates_gt.csv")
    frame = features.merge(targets, on=["survey_id", "hhid"], how="inner")
    return frame, rates


def threshold_config(rates):
    columns = [c for c in rates.columns if c.startswith("pct_hh_below_")]
    mapping = {float(c.replace("pct_hh_below_", "")): c for c in columns}
    thresholds = np.array(sorted(mapping))
    return mapping, thresholds


def rates_from_consumption(predictions, weights, thresholds):
    return np.array(
        [np.average(predictions < threshold, weights=weights) for threshold in thresholds]
    )


def score_fold(actual_consumption, predicted_consumption, actual_rates, predicted_rates):
    consumption_error = mean_absolute_percentage_error(
        actual_consumption, predicted_consumption
    )
    poverty_error = weighted_poverty_mape(actual_rates, predicted_rates)
    return {
        "consumption_mape": consumption_error,
        "poverty_wmape": poverty_error,
        "blended_score": blended_score(consumption_error, poverty_error),
    }


def evaluate_pooled_baseline(
    train_frame,
    heldout_frame,
    actual_rates,
    thresholds,
    *,
    n_estimators,
):
    preprocessor = FeaturePreprocessor().fit(train_frame)
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        preprocessor.transform(train_frame),
        np.log(train_frame["cons_ppp17"].to_numpy() + 1e-6),
    )
    predicted_consumption = np.exp(model.predict(preprocessor.transform(heldout_frame)))
    predicted_rates = rates_from_consumption(
        predicted_consumption,
        heldout_frame["weight"].to_numpy(),
        thresholds,
    )
    return score_fold(
        heldout_frame["cons_ppp17"].to_numpy(),
        predicted_consumption,
        actual_rates,
        predicted_rates,
    )


def evaluate_adaptive_pipeline(
    train_frame,
    heldout_frame,
    donor_rates,
    actual_rates,
    poverty_map,
    thresholds,
    *,
    n_estimators,
):
    preprocessor = FeaturePreprocessor().fit(train_frame)
    consumption_models, survey_log_means = train_consumption_models(
        train_frame, preprocessor, n_estimators=n_estimators
    )
    poverty_models = train_poverty_models(
        train_frame,
        preprocessor,
        thresholds,
        n_estimators=n_estimators,
    )
    train_cdfs = {
        survey_id: get_gt_cdf(donor_rates, survey_id, poverty_map, thresholds)
        for survey_id in train_frame["survey_id"].unique()
    }
    predicted_consumption, predicted_poverty = run_inference(
        heldout_frame.drop(columns=["cons_ppp17"]),
        preprocessor,
        consumption_models,
        poverty_models,
        train_cdfs,
        survey_log_means,
        thresholds,
    )
    predicted_rates = predicted_poverty.iloc[0, 1:].to_numpy(dtype=float)
    return score_fold(
        heldout_frame["cons_ppp17"].to_numpy(),
        predicted_consumption["cons_ppp17"].to_numpy(),
        actual_rates,
        predicted_rates,
    )


def main():
    args = parse_args()
    if args.n_estimators <= 0:
        raise SystemExit("--n-estimators must be positive")

    started_at = perf_counter()
    frame, rates = load_training_data(args.data_dir)
    poverty_map, thresholds = threshold_config(rates)
    rows = []

    for heldout_survey in sorted(frame["survey_id"].unique()):
        is_heldout = frame["survey_id"] == heldout_survey
        train_frame = frame.loc[~is_heldout].reset_index(drop=True)
        heldout_frame = frame.loc[is_heldout].reset_index(drop=True)
        donor_rates = rates[rates["survey_id"] != heldout_survey]
        actual_rates = get_gt_cdf(
            rates, heldout_survey, poverty_map, thresholds
        )

        for model_name, evaluator in (
            ("pooled_lightgbm", evaluate_pooled_baseline),
            ("survey_adaptive_cdf_ensemble", evaluate_adaptive_pipeline),
        ):
            if model_name == "pooled_lightgbm":
                scores = evaluator(
                    train_frame,
                    heldout_frame,
                    actual_rates,
                    thresholds,
                    n_estimators=args.n_estimators,
                )
            else:
                scores = evaluator(
                    train_frame,
                    heldout_frame,
                    donor_rates,
                    actual_rates,
                    poverty_map,
                    thresholds,
                    n_estimators=args.n_estimators,
                )
            rows.append(
                {
                    "model": model_name,
                    "heldout_survey": heldout_survey,
                    **scores,
                }
            )
            print(
                f"{heldout_survey} | {model_name} | "
                f"blended={scores['blended_score']:.4f}"
            )

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = (
        results.groupby("model")
        .agg(
            consumption_mape=("consumption_mape", "mean"),
            poverty_wmape=("poverty_wmape", "mean"),
            blended_score=("blended_score", "mean"),
        )
        .sort_values("blended_score")
    )
    print("\nMean leave-one-survey-out metrics (lower is better):")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved fold-level results to {args.output}")
    print(f"Validation completed in {perf_counter() - started_at:.1f}s")


if __name__ == "__main__":
    main()
