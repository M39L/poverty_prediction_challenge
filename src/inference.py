import pandas as pd
import numpy as np
from src.utils import softmax

def run_inference(test_df, prep, models, poverty_models, train_cdfs, survey_log_means, poverty_thresholds):
    
    # Main execution logic for generating predictions using a survey-aware ensemble

    cons_rows = []
    poverty_rows = []

    for survey in test_df["survey_id"].unique():
        mask = test_df["survey_id"] == survey
        X = prep.transform(test_df.loc[mask])
        w = test_df.loc[mask, "weight"].values
        hhids = test_df.loc[mask, "hhid"].values

        # 1. Predict Poverty CDF for the current test survey 
        pred_cdf = np.array([
            np.average(poverty_models[z].predict(X), weights=w)
            for z in poverty_thresholds
        ])
        # Ensure the CDF is monotonically increasing
        pred_cdf = np.maximum.accumulate(pred_cdf)
        pred_cdf = np.clip(pred_cdf, 0, 1)

        # 2. Calculate ensemble weights based on similarity to training surveys 
        # Squared distance between predicted CDF and ground truth training CDFs
        distances = [
            np.mean((train_cdfs[s] - pred_cdf) ** 2)
            for s in train_cdfs
        ]
        # Temperature-scaled softmax for weights assignment
        survey_weights = softmax(-np.array(distances) / 0.5)

        # 3. Weighted ensemble of consumption predictions 
        y_log_hat = np.zeros(len(X))
        for wgt, s in zip(survey_weights, train_cdfs):
            y_log_hat += wgt * models[s].predict(X)

        # 4. Mean-shift calibration to align predicted scale with likely truth 
        weighted_log_mean = np.sum([wgt * survey_log_means[s] for wgt, s in zip(survey_weights, train_cdfs)])
        y_log_hat += (weighted_log_mean - y_log_hat.mean())

        # 5. Final transformation and outlier clipping 
        y_hat = np.exp(y_log_hat)
        y_hat = np.clip(y_hat, np.percentile(y_hat, 0.5), np.percentile(y_hat, 99.5))

        # Store household level results
        cons_rows.extend({
            "survey_id": survey,
            "hhid": hhid,
            "cons_ppp17": float(cons)
        } for hhid, cons in zip(hhids, y_hat))

        # 6. Compute poverty distribution per threshold 
        p_row = {"survey_id": survey}
        for z in poverty_thresholds:
            # Calculate weighted average of binary predictions
            p_row[f"pct_hh_below_{z:.2f}"] = float(np.average(poverty_models[z].predict(X), weights=w))
        poverty_rows.append(p_row)

    # Convert results to DataFrames
    pred_cons = pd.DataFrame(cons_rows)
    pred_poverty = pd.DataFrame(poverty_rows)
    
    # Final post-processing to guarantee logical consistency (CDF monotonicity)
    cols = [c for c in pred_poverty.columns if "pct_hh_below_" in c]
    pred_poverty[cols] = np.maximum.accumulate(pred_poverty[cols].values, axis=1)

    return pred_cons, pred_poverty