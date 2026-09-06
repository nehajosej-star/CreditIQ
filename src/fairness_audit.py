"""
Algorithmic Fairness and Regulatory Compliance Auditing.
Evaluates Disparate Impact, Demographic Parity, and Equalized Odds across
protected demographic attributes (Gender, Age) according to ECOA and CFPB standards.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes classification performance metrics for a specific demographic subgroup.
    Note: In loan underwriting, y=1 indicates DEFAULT (unfavorable), and y=0 indicates REPAID (favorable).
    Approval decision: y_pred_approval = (y_pred == 0).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total = len(y_true)
    if total == 0:
        return {"count": 0, "approval_rate": 0.0, "tpr_approval": 0.0, "fpr_approval": 0.0, "default_rate": 0.0}

    # Approvals: model predicts no default (0)
    approval_pred = (y_pred == 0).astype(int)
    approval_true = (y_true == 0).astype(int)

    approval_rate = float(np.mean(approval_pred))
    actual_creditworthy_rate = float(np.mean(approval_true))

    # True Positive Rate for Approval (Sensitivity to truly creditworthy applicants)
    positives = np.sum(approval_true == 1)
    if positives > 0:
        tpr_approval = float(np.sum((approval_pred == 1) & (approval_true == 1)) / positives)
    else:
        tpr_approval = 1.0

    # False Positive Rate for Approval (Approving an applicant who actually defaults)
    negatives = np.sum(approval_true == 0)
    if negatives > 0:
        fpr_approval = float(np.sum((approval_pred == 1) & (approval_true == 0)) / negatives)
    else:
        fpr_approval = 0.0

    return {
        "count": int(total),
        "approval_rate": round(approval_rate, 4),
        "tpr_approval": round(tpr_approval, 4),
        "fpr_approval": round(fpr_approval, 4),
        "actual_creditworthy_rate": round(actual_creditworthy_rate, 4),
        "observed_default_rate": round(float(np.mean(y_true)), 4),
    }


def audit_protected_attribute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: pd.Series,
    privileged_group: Any,
    unprivileged_group: Any,
    attribute_name: str = "Attribute",
) -> Dict[str, Any]:
    """
    Computes regulatory fairness metrics for a single protected attribute:
    - Disparate Impact Ratio (Four-Fifths Rule threshold: 0.80)
    - Demographic Parity Difference
    - Equal Opportunity Difference (TPR Parity)
    """
    mask_priv = protected_attribute == privileged_group
    mask_unpriv = protected_attribute == unprivileged_group

    priv_metrics = compute_binary_metrics(y_true[mask_priv], y_pred[mask_priv])
    unpriv_metrics = compute_binary_metrics(y_true[mask_unpriv], y_pred[mask_unpriv])

    # Disparate Impact: Ratio of approval rates (unprivileged / privileged)
    priv_rate = priv_metrics["approval_rate"]
    unpriv_rate = unpriv_metrics["approval_rate"]

    if priv_rate > 0:
        disparate_impact_ratio = float(unpriv_rate / priv_rate)
    else:
        disparate_impact_ratio = 1.0 if unpriv_rate == 0 else float("inf")

    # Demographic Parity Difference: |P(Approve|Priv) - P(Approve|Unpriv)|
    demographic_parity_diff = abs(priv_rate - unpriv_rate)

    # Equal Opportunity Difference: |TPR(Approve|Priv) - TPR(Approve|Unpriv)|
    equal_opportunity_diff = abs(priv_metrics["tpr_approval"] - unpriv_metrics["tpr_approval"])

    # Four-Fifths Rule compliance: 0.80 <= Disparate Impact <= 1.25
    four_fifths_pass = 0.80 <= disparate_impact_ratio <= 1.25

    status = "COMPLIANT (PASS)" if four_fifths_pass and demographic_parity_diff < 0.10 else "REVIEW REQUIRED (FLAG)"

    return {
        "attribute_name": attribute_name,
        "privileged_group": str(privileged_group),
        "unprivileged_group": str(unprivileged_group),
        "privileged_metrics": priv_metrics,
        "unprivileged_metrics": unpriv_metrics,
        "disparate_impact_ratio": round(disparate_impact_ratio, 4),
        "demographic_parity_difference": round(demographic_parity_diff, 4),
        "equal_opportunity_difference": round(equal_opportunity_diff, 4),
        "four_fifths_threshold": 0.80,
        "four_fifths_compliant": bool(four_fifths_pass),
        "status": status,
    }


def perform_fairness_audit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    df_raw: pd.DataFrame,
    threshold: float = 0.35,  # Decision threshold for default rejection
    output_path: str = "models/fairness_report.json",
) -> Dict[str, Any]:
    """
    Conducts comprehensive fairness audit across CODE_GENDER and AGE_GROUP.
    Saves and prints executive audit summary report.
    """
    logger.info("Executing Algorithmic Fairness and Equal Credit Opportunity Audit...")

    # Binary decision: If pred proba >= threshold, model predicts DEFAULT (deny loan)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Derive age group: < 30 vs >= 30
    if "AGE_YEARS" in df_raw.columns:
        age_years = df_raw["AGE_YEARS"].values
    elif "DAYS_BIRTH" in df_raw.columns:
        age_years = np.abs(df_raw["DAYS_BIRTH"].values) / 365.25
    else:
        age_years = np.full(len(df_raw), 35.0)

    age_group_series = pd.Series(np.where(age_years < 30, "< 30", ">= 30"), index=df_raw.index)

    # Gender Audit: Privileged='M', Unprivileged='F' (or vice versa for parity checks)
    gender_series = df_raw["CODE_GENDER"] if "CODE_GENDER" in df_raw.columns else pd.Series(["M"] * len(df_raw))
    gender_audit = audit_protected_attribute(
        y_true=y_true,
        y_pred=y_pred,
        protected_attribute=gender_series,
        privileged_group="M",
        unprivileged_group="F",
        attribute_name="Gender (CODE_GENDER)",
    )

    # Age Audit: Privileged='>= 30', Unprivileged='< 30'
    age_audit = audit_protected_attribute(
        y_true=y_true,
        y_pred=y_pred,
        protected_attribute=age_group_series,
        privileged_group=">= 30",
        unprivileged_group="< 30",
        attribute_name="Age Cohort (AGE_GROUP)",
    )

    overall_pass = gender_audit["four_fifths_compliant"] and age_audit["four_fifths_compliant"]

    report = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "underwriting_threshold": threshold,
        "sample_size": len(y_true),
        "overall_status": "CERTIFIED COMPLIANT" if overall_pass else "ACTION REQUIRED",
        "gender_audit": gender_audit,
        "age_audit": age_audit,
        "regulatory_notes": {
            "four_fifths_rule": "Under EEOC/CFPB guidelines, a selection rate for any group that is less than 80% (4/5ths) of the rate for the group with the highest rate constitutes evidence of disparate impact.",
            "equal_opportunity": "Measures whether creditworthy borrowers of all demographics have equal probability of obtaining loan approval.",
            "ecoa_fcra": "Equal Credit Opportunity Act (ECOA) & Fair Credit Reporting Act (FCRA) prohibit discrimination based on protected demographics.",
        },
    }

    # Print Executive Summary
    print("\n" + "=" * 80)
    print("                EXECUTIVE FAIRNESS & REGULATORY COMPLIANCE REPORT                ")
    print("=" * 80)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Underwriting Threshold (Default Cutoff): {threshold:.2f}\n")

    for audit in [gender_audit, age_audit]:
        print(f"--- {audit['attribute_name']} ---")
        print(f"  Privileged ({audit['privileged_group']}) Approval Rate:    {audit['privileged_metrics']['approval_rate']:.2%}")
        print(f"  Unprivileged ({audit['unprivileged_group']}) Approval Rate:  {audit['unprivileged_metrics']['approval_rate']:.2%}")
        print(f"  Disparate Impact Ratio:               {audit['disparate_impact_ratio']:.3f} (Threshold >= 0.80)")
        print(f"  Demographic Parity Difference:        {audit['demographic_parity_difference']:.3f}")
        print(f"  Equal Opportunity Difference (TPR):   {audit['equal_opportunity_difference']:.3f}")
        print(f"  Audit Status:                         {audit['status']}")
        print()

    print("=" * 80 + "\n")

    # Persist JSON report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Fairness report saved to {output_path}")

    return report


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    y_t = np.random.binomial(1, 0.08, size=n)
    y_p = np.random.beta(2, 20, size=n)
    df_dummy = pd.DataFrame({
        "CODE_GENDER": np.random.choice(["M", "F"], size=n),
        "DAYS_BIRTH": -np.random.randint(7500, 24000, size=n),
    })
    perform_fairness_audit(y_t, y_p, df_dummy, output_path="models/fairness_report.json")
