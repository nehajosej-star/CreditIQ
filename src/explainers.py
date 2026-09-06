"""
Explainability Module: SHAP TreeExplainer, LIME Tabular Explainer,
and Automated Adverse Action Code Generator compliant with ECOA / FCRA regulations.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import cloudpickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Regulatory Adverse Action Dictionary mapping features to CFPB / ECOA compliant reasons
ADVERSE_ACTION_MAPPING = {
    "ANNUITY_TO_INCOME_RATIO": {
        "code": "AA-DTI-01",
        "reason": "Excessive Payment-to-Income (Debt-to-Income) Ratio",
        "detail": "The required loan annuity payment exceeds the sustainable threshold relative to verified annual income.",
    },
    "CREDIT_TO_INCOME_RATIO": {
        "code": "AA-DTI-02",
        "reason": "Total Debt Obligation Exceeds Income Capacity",
        "detail": "The requested total credit facility is disproportionate to annual declared earnings.",
    },
    "CREDIT_TERM": {
        "code": "AA-TERM-03",
        "reason": "Aggressive Loan Repayment Schedule / Term",
        "detail": "The ratio of annuity payment to overall principal implies an excessively compressed repayment structure.",
    },
    "EXT_SOURCES_MEAN": {
        "code": "AA-BUR-04",
        "reason": "Sub-Optimal External Credit Bureau Composite Rating",
        "detail": "Aggregated credit bureau scores indicate elevated historical delinquency probability or insufficient positive credit file depth.",
    },
    "EXT_SOURCE_1": {
        "code": "AA-BUR-01",
        "reason": "External Credit Score 1 Below Underwriting Standards",
        "detail": "Primary external credit agency report reflects elevated historical credit risk.",
    },
    "EXT_SOURCE_2": {
        "code": "AA-BUR-02",
        "reason": "External Credit Score 2 Below Underwriting Standards",
        "detail": "Secondary credit agency profile reflects unfavorable bureau scoring metrics.",
    },
    "EXT_SOURCE_3": {
        "code": "AA-BUR-03",
        "reason": "External Credit Score 3 Below Underwriting Standards",
        "detail": "Tertiary external credit agency record indicates non-qualifying risk indicators.",
    },
    "EMPLOYMENT_YEARS": {
        "code": "AA-EMP-05",
        "reason": "Insufficient Length of Steady Employment",
        "detail": "Continuous verified employment tenure does not meet minimum program stability requirements.",
    },
    "DAYS_EMPLOYED": {
        "code": "AA-EMP-05",
        "reason": "Insufficient Employment History Duration",
        "detail": "Tenure at current employer or historical work stability is below underwriting guidelines.",
    },
    "AGE_YEARS": {
        "code": "AA-AGE-06",
        "reason": "Limited Historical Credit File Maturity",
        "detail": "Length of verifiable financial obligations is insufficient to qualify for the requested credit tier.",
    },
    "AMT_INCOME_TOTAL": {
        "code": "AA-INC-07",
        "reason": "Declared Gross Income Insufficient",
        "detail": "Reported income does not provide sufficient safety margin for credit facility debt servicing.",
    },
    "AMT_CREDIT": {
        "code": "AA-CRE-08",
        "reason": "Total Credit Requested Exceeds Policy Limits",
        "detail": "The principal balance sought exceeds standard automated lending capacity.",
    },
    "AMT_ANNUITY": {
        "code": "AA-ANN-09",
        "reason": "High Periodic Installment Amount",
        "detail": "Scheduled periodic installment is high compared to liquid cash-flow buffers.",
    },
    "GOODS_TO_CREDIT_RATIO": {
        "code": "AA-COL-10",
        "reason": "Insufficient Asset Collateral Coverage",
        "detail": "The financed goods asset valuation does not adequately secure the principal balance.",
    },
}

DEFAULT_ADVERSE_REASON = {
    "code": "AA-GEN-99",
    "reason": "Credit Scoring Model Attribute Contribution",
    "detail": "Applicant profile factor contributed adversely to empirical creditworthiness scoring.",
}


def generate_adverse_action_codes(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Translates top risk-increasing SHAP attributions into clear, ECOA/FCRA-compliant explanations.

    Args:
        shap_values: Array of SHAP values for the default (risk) class for a single applicant.
        feature_names: List of preprocessed feature names matching shap_values.
        feature_values: Optional original or transformed values of the features.
        top_k: Number of adverse action reasons to produce (typically 3 to 4 per CFPB standards).

    Returns:
        List of structured adverse action entries formatted for formal Notice of Action Taken.
    """
    shap_arr = np.asarray(shap_values).flatten()

    # Identify positive contributions (features that pushed predicted default risk UP)
    # Higher SHAP value = higher contribution to default risk
    sorted_indices = np.argsort(-shap_arr)

    adverse_reasons = []
    rank = 1

    for idx in sorted_indices:
        attribution = float(shap_arr[idx])
        # Only consider features that actively increased default risk
        if attribution <= 0 and len(adverse_reasons) >= 1:
            break

        feat_name = feature_names[idx]
        val = None
        if feature_values is not None:
            val_arr = np.asarray(feature_values).flatten()
            if idx < len(val_arr):
                val = float(val_arr[idx])

        # Find matching regulatory mapping
        mapped_info = None
        # Exact match
        if feat_name in ADVERSE_ACTION_MAPPING:
            mapped_info = ADVERSE_ACTION_MAPPING[feat_name]
        else:
            # Check prefix/substring (for one-hot encoded features like NAME_EDUCATION_TYPE_...)
            for k, v in ADVERSE_ACTION_MAPPING.items():
                if k in feat_name:
                    mapped_info = v
                    break

        if mapped_info is None:
            mapped_info = {
                "code": f"AA-FEAT-{rank:02d}",
                "reason": f"Elevated Risk from {feat_name.replace('_', ' ').title()}",
                "detail": f"Feature '{feat_name}' weighted unfavorably against loan repayment probability.",
            }

        entry = {
            "rank": rank,
            "feature": feat_name,
            "feature_value": val,
            "shap_attribution": round(attribution, 4),
            "adverse_action_code": mapped_info["code"],
            "regulatory_reason": mapped_info["reason"],
            "detailed_explanation": mapped_info["detail"],
        }
        adverse_reasons.append(entry)
        rank += 1

        if len(adverse_reasons) >= top_k:
            break

    # If all SHAP values were negative or none increased risk (extremely low risk applicant)
    if not adverse_reasons:
        adverse_reasons.append({
            "rank": 1,
            "feature": "OVERALL_PROFILE",
            "feature_value": None,
            "shap_attribution": 0.0,
            "adverse_action_code": "AA-NONE-00",
            "regulatory_reason": "No Material Adverse Factors",
            "detailed_explanation": "Applicant meets all baseline credit criteria; no primary adverse flags identified.",
        })

    return adverse_reasons


def setup_explainers(
    base_tree_model: Any,
    X_train_transformed: pd.DataFrame,
    feature_names: List[str],
    class_names: List[str] = ["Repaid", "Default"],
    output_dir: str = "models",
) -> Tuple[Any, Any]:
    """
    Initializes and saves:
    1. shap.TreeExplainer on the uncalibrated base tree model
    2. lime.lime_tabular.LimeTabularExplainer on the preprocessed feature space
    """
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Initializing SHAP TreeExplainer on base tree model...")
    # TreeExplainer natively understands LightGBM, XGBoost, and CatBoost
    shap_explainer = shap.TreeExplainer(base_tree_model)

    logger.info("Initializing LIME Tabular Explainer on preprocessed feature distributions...")
    X_sample = X_train_transformed.values
    if len(X_sample) > 2000:
        # Subsample for background distribution speed if dataset is large
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_sample), size=2000, replace=False)
        X_sample = X_sample[idx]

    lime_explainer = LimeTabularExplainer(
        training_data=X_sample,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42,
    )

    # Save artifacts
    shap_path = os.path.join(output_dir, "shap_explainer.pkl")
    lime_path = os.path.join(output_dir, "lime_explainer.pkl")

    with open(shap_path, "wb") as f:
        cloudpickle.dump(shap_explainer, f)
    with open(lime_path, "wb") as f:
        cloudpickle.dump(lime_explainer, f)
    logger.info(f"Saved SHAP explainer to {shap_path} and LIME explainer to {lime_path}")

    return shap_explainer, lime_explainer


def load_shap_explainer(path: str = "models/shap_explainer.pkl") -> Any:
    """Loads saved SHAP TreeExplainer artifact."""
    import cloudpickle
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def load_lime_explainer(path: str = "models/lime_explainer.pkl") -> Any:
    """Loads saved LIME TabularExplainer artifact."""
    import cloudpickle
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def compute_lime_local_explanation(
    lime_explainer: Any,
    predict_fn: Any,
    applicant_vector: np.ndarray,
    num_features: int = 8,
) -> List[Tuple[str, float]]:
    """
    Generates LIME feature contribution pairs for a single applicant.
    """
    exp = lime_explainer.explain_instance(
        data_row=applicant_vector,
        predict_fn=predict_fn,
        num_features=num_features,
        labels=(1,),  # Default class
    )
    # List of (feature_rule, weight)
    return exp.as_list(label=1)
