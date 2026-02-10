import lightgbm as lgb
import numpy as np

def train_consumption_models(df, prep):

    #Trains a separate LightGBM regressor for each unique survey
    models = {}
    survey_log_means = {}
    for survey in df["survey_id"].unique():
        mask = df["survey_id"] == survey
        X = prep.transform(df.loc[mask])
        y_log = np.log(df.loc[mask, "cons_ppp17"] + 1e-6)
        
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X, y_log)
        models[survey] = model
        survey_log_means[survey] = y_log.mean()
    return models, survey_log_means


def train_poverty_models(df, prep, poverty_thresholds):

    #Trains a global LightGBM model for each binary poverty threshold
    poverty_models = {}
    X_all = prep.transform(df)
    weights_all = df["weight"].values
    for z in poverty_thresholds:
        # Convert continuous target to binary indicator for each poverty line
        y_bin = (df["cons_ppp17"].values < z).astype(int)
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5, 
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_all, y_bin, sample_weight=weights_all)
        poverty_models[z] = model
    return poverty_models