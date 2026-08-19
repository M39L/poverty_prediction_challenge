import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeaturePreprocessor:
    def __init__(self):
        # Mapping qualitative survey responses to numeric values
        self.binary_map = {
            "Yes": 1, "No": 0, "Male": 1, "Female": 0,
            "Owner": 1, "Not owner": 0, "Employed": 1, "Not employed": 0,
            "Urban": 1, "Rural": 0, "Access": 1, "No access": 0
        }
        self.binary_cols = ['male','owner','water','toilet','sewer','elect','employed','urban']
        
        # Ordinal mapping for education
        self.educ_map = {
            "Never attended": 0, "Incomplete Primary Education": 1,
            "Complete Primary Education": 2, "Incomplete Secondary Education": 3,
            "Complete Secondary Education": 4, "Incomplete Tertiary Education": 5,
            "Complete Tertiary Education": 6
        }
        self.nominal_cols = ['dweltyp','water_source','sanitation_source','sector1d','any_nonagric']
        self.label_encoders = {}
        self.feature_names = None

    def _add_custom_features(self, df):
        """
        Internal helper to create engineered features:
        1. log_hsize: logarithmic scale of household size
        2. consumed_yes_count: number of 'Yes' responses in consumption columns
        """
        # 1. Household size log transformation
        if 'hsize' in df.columns:
            df["log_hsize"] = np.log1p(df["hsize"])
        
        # 2. Count 'Yes' for consumption-related columns
        consumed_cols = [c for c in df.columns if c.startswith("consumed")]
        if consumed_cols:
            # We check for the string 'Yes' BEFORE mapping to integers
            df['consumed_yes_count'] = (df[consumed_cols] == 'Yes').sum(axis=1)
            
        return df

    def _binary_encode(self, df):
        """Internal helper to convert text binary features to integers."""
        consumed_cols = [
            c
            for c in df.columns
            if c.startswith("consumed") and c != "consumed_yes_count"
        ]
        for c in self.binary_cols + consumed_cols:
            if c in df.columns:
                df[c] = df[c].map(self.binary_map).fillna(0).astype(int)

    def fit(self, df):
        """Learn encoding maps from the training data."""
        df = df.copy()
        
        # Apply feature engineering first
        df = self._add_custom_features(df)
        
        # Apply encodings
        self._binary_encode(df)
        df['educ_max'] = df['educ_max'].map(self.educ_map).fillna(0).astype(int)
        
        for c in self.nominal_cols:
            if c in df.columns:
                le = LabelEncoder()
                values = df[c].fillna("missing").astype(str)
                # Reserve a fallback class for categories first seen at inference.
                le.fit(np.append(values.to_numpy(), "missing"))
                self.label_encoders[c] = le
        
        # Define the final feature set
        self.feature_names = [
            c for c in df.columns 
            if c not in ['survey_id', 'hhid', 'com', 'cons_ppp17', 'weight', 'hsize']
        ]
        return self

    def transform(self, df):
        """Apply engineered features and learned encodings to new data."""
        df = df.copy()
        
        # Must follow the same order as in fit()
        df = self._add_custom_features(df)
        self._binary_encode(df)
        df['educ_max'] = df['educ_max'].map(self.educ_map).fillna(0).astype(int)
        
        for c, le in self.label_encoders.items():
            df[c] = df[c].fillna("missing").astype(str)
            df[c] = df[c].where(df[c].isin(le.classes_), "missing")
            df[c] = le.transform(df[c])
            
        return df[self.feature_names]
