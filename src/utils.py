import numpy as np

def softmax(x):

    # Compute softmax values for each sets of scores in x
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def get_gt_cdf(cdf_df, survey_id, poverty_map, poverty_thresholds):

    # Extracts the ground truth poverty distribution for a specific survey
    row = cdf_df[cdf_df["survey_id"] == survey_id].iloc[0]
    return np.array([row[poverty_map[z]] for z in poverty_thresholds])