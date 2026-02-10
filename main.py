import pandas as pd
import numpy as np
import os

# Import custom modules from the src directory
from src.preprocessing import FeaturePreprocessor
from src.utils import get_gt_cdf
from src.trainer import train_consumption_models, train_poverty_models
from src.inference import run_inference

def main():
    # 1. Path Configuration 
    DATA_DIR = "data"
    train_x_path = os.path.join(DATA_DIR, "train_hh_features.csv")
    train_y_path = os.path.join(DATA_DIR, "train_hh_gt.csv")
    cdf_gt_path  = os.path.join(DATA_DIR, "train_rates_gt.csv")
    test_x_path  = os.path.join(DATA_DIR, "test_hh_features.csv")

    try:
        train_X = pd.read_csv(train_x_path)
        train_y = pd.read_csv(train_y_path)
        cdf_df  = pd.read_csv(cdf_gt_path)
        test_df = pd.read_csv(test_x_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in {DATA_DIR}. {e}")
        return

    df = train_X.merge(train_y, on=["survey_id", "hhid"], how="inner")

    # Feature engineering: log-scale household size and activity counts
    for d in [df, test_df]:
        d["log_hsize"] = np.log1p(d["hsize"])

    #  2. Preprocessing 
    # Initializing and fitting the preprocessor to learn categorical encodings
    prep = FeaturePreprocessor().fit(df)

    #  3. Poverty Configuration 
    # Dynamically extract poverty thresholds from the column names
    poverty_cols = [c for c in cdf_df.columns if c.startswith("pct_hh_below_")]
    poverty_map = {float(c.replace("pct_hh_below_", "")): c for c in poverty_cols}
    poverty_thresholds = np.array(sorted(poverty_map))

    # Store Ground Truth CDFs for each training survey to use in similarity matching
    train_cdfs = {
        s: get_gt_cdf(cdf_df, s, poverty_map, poverty_thresholds) 
        for s in df["survey_id"].unique()
    }

    #  4. Model Training 
    # Train survey-specific LightGBM models for consumption
    models, survey_log_means = train_consumption_models(df, prep)
    
    # Train global LightGBM models for each poverty threshold
    poverty_models = train_poverty_models(df, prep, poverty_thresholds)

    #  5. Inference & Ensembling 
    # Perform prediction with similarity-based weighting and mean-shift calibration
    pred_cons, pred_poverty = run_inference(
        test_df, prep, models, poverty_models, 
        train_cdfs, survey_log_means, poverty_thresholds
    )

    #  6. Exporting Results
    pred_cons.to_csv("result/predicted_household_consumption.csv", index=False)
    pred_poverty.to_csv("result/predicted_poverty_distribution.csv", index=False)
    
if __name__ == "__main__":
    main()