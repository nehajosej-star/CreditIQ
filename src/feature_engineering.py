"""
Feature Engineering and Preprocessing Pipeline for Credit Risk Scoring.
Calculates domain-specific credit ratios, handles imputation, encoding, and prevents data leakage.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DomainRatioExtractor(BaseEstimator, TransformerMixin):
    """
    Computes key financial ratios and transformed domain features:
    - CREDIT_TO_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
    - ANNUITY_TO_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL
    - CREDIT_TERM = AMT_ANNUITY / AMT_CREDIT
    - EXT_SOURCES_MEAN = mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
    - AGE_YEARS = abs(DAYS_BIRTH) / 365.25
    - EMPLOYMENT_YEARS = abs(DAYS_EMPLOYED) / 365.25 (with 365243 handled as 0)
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Handle employment anomaly (Home Credit sentinel 365243 means unemployed/pensioner)
        days_employed_clean = df["DAYS_EMPLOYED"].copy()
        days_employed_clean = np.where(days_employed_clean == 365243, 0, days_employed_clean)

        # Domain calculations
        df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365.25
        df["EMPLOYMENT_YEARS"] = np.abs(days_employed_clean) / 365.25

        # Avoid division by zero with small epsilons
        income_safe = np.where(df["AMT_INCOME_TOTAL"] <= 0, 1.0, df["AMT_INCOME_TOTAL"])
        credit_safe = np.where(df["AMT_CREDIT"] <= 0, 1.0, df["AMT_CREDIT"])

        df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / income_safe
        df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / income_safe
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / credit_safe

        # Goods price to credit ratio
        if "AMT_GOODS_PRICE" in df.columns:
            df["GOODS_TO_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit_safe

        # External sources mean (ignoring NaNs row-wise)
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols:
            df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1, skipna=True)
            # Impute overall mean if all 3 are missing in a row
            df["EXT_SOURCES_MEAN"] = df["EXT_SOURCES_MEAN"].fillna(0.5)

        return df


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive, leak-free preprocessor for credit risk modeling.
    Fits imputers and encoders strictly on the training partition.
    Maintains clean feature names for SHAP, LIME, and feature attribution.
    """

    def __init__(self, scale_numeric: bool = False):
        self.scale_numeric = scale_numeric
        self.ratio_extractor = DomainRatioExtractor()
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None

        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _identify_columns(self, df: pd.DataFrame):
        exclude_cols = ["SK_ID_CURR", "TARGET"]
        candidate_cols = [c for c in df.columns if c not in exclude_cols]

        self.num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
        self.cat_cols = [c for c in candidate_cols if c not in self.num_cols]

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting CreditRiskPreprocessor on training data...")
        df_feat = self.ratio_extractor.transform(X)

        self._identify_columns(df_feat)
        logger.info(f"Identified {len(self.num_cols)} numeric features and {len(self.cat_cols)} categorical features.")

        # Numeric Imputer (Median)
        if self.num_cols:
            self.numeric_imputer = SimpleImputer(strategy="median")
            self.numeric_imputer.fit(df_feat[self.num_cols])

            if self.scale_numeric:
                self.scaler = StandardScaler()
                num_imp = self.numeric_imputer.transform(df_feat[self.num_cols])
                self.scaler.fit(num_imp)

        # Categorical Imputer (Most Frequent) & One-Hot Encoder
        if self.cat_cols:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            cat_imp = self.categorical_imputer.fit_transform(df_feat[self.cat_cols])

            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.one_hot_encoder.fit(cat_imp)

        # Construct final feature names
        feature_names = list(self.num_cols)
        if self.cat_cols and self.one_hot_encoder is not None:
            encoded_cat_names = self.one_hot_encoder.get_feature_names_out(self.cat_cols).tolist()
            feature_names.extend(encoded_cat_names)

        self.feature_names_ = feature_names
        self.is_fitted = True
        logger.info(f"Preprocessing pipeline successfully fitted. Total transformed features: {len(self.feature_names_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("CreditRiskPreprocessor must be fitted before transform can be called.")

        df_feat = self.ratio_extractor.transform(X)

        # Process numeric features
        if self.num_cols:
            num_data = self.numeric_imputer.transform(df_feat[self.num_cols])
            if self.scale_numeric and self.scaler is not None:
                num_data = self.scaler.transform(num_data)
        else:
            num_data = np.empty((len(df_feat), 0))

        # Process categorical features
        if self.cat_cols and self.one_hot_encoder is not None:
            cat_imp = self.categorical_imputer.transform(df_feat[self.cat_cols])
            cat_data = self.one_hot_encoder.transform(cat_imp)
        else:
            cat_data = np.empty((len(df_feat), 0))

        # Combine
        combined_data = np.hstack([num_data, cat_data])
        df_transformed = pd.DataFrame(combined_data, columns=self.feature_names_, index=X.index)
        return df_transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    from src.data_loader import generate_synthetic_home_credit_data
    df = generate_synthetic_home_credit_data(100)
    prep = CreditRiskPreprocessor()
    df_proc = prep.fit_transform(df)
    print(f"Transformed shape: {df_proc.shape}")
    print(f"Feature Names: {prep.feature_names_}")
    print(df_proc.head(2))
