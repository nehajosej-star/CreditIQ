import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.data_loader import load_data
from src.feature_engineering import CreditRiskPreprocessor
from src.explainers import setup_explainers
from src.fairness_audit import perform_fairness_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.35,
) -> Dict[str, float]:
    """
    Computes key financial credit scoring metrics:
    - ROC-AUC
    - PR-AUC (Average Precision)
    - F1-Score (at decision threshold)
    - Brier Score Loss (lower is better calibrated)
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_pred_proba)

    metrics = {
        "model": model_name,
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_score": round(float(f1), 4),
        "brier_score": round(float(brier), 4),
    }
    return metrics


def train_and_benchmark(
    data_path: str = None,
    output_dir: str = "models",
    n_samples: int = 10000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end ML pipeline:
    1. Loads / generates synthetic dataset
    2. 80/20 Stratified train-test split
    3. Fits leak-free feature preprocessor
    4. Benchmarks SMOTE vs Class-Weighted LightGBM, XGBoost, CatBoost
    5. Calibrates top model using Isotonic regression
    6. Generates SHAP & LIME explainers
    7. Runs Regulatory Fairness Audit
    8. Exports all artifacts to models/ directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    logger.info("Step 1: Loading loan application dataset...")
    df = load_data(n_samples=n_samples, random_state=random_state)
    logger.info(f"Dataset loaded: {df.shape[0]} applicants, default rate: {df['TARGET'].mean():.2%}")

    # Step 2: Stratified Split (80% dev, 20% test)
    logger.info("Step 2: Performing stratified train/test split...")
    df_dev, df_test = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df["TARGET"],
    )

    # Further split dev into 70% train (for model fit) and 10% calibration (for isotonic prefit)
    df_train, df_calib = train_test_split(
        df_dev,
        test_size=0.125,  # 0.125 of 80% = 10% of total
        random_state=random_state,
        stratify=df_dev["TARGET"],
    )

    y_train = df_train["TARGET"].values
    y_calib = df_calib["TARGET"].values
    y_test = df_test["TARGET"].values

    # Step 3: Feature Engineering Preprocessing (fitted strictly on df_train)
    logger.info("Step 3: Fitting CreditRiskPreprocessor on training data...")
    preprocessor = CreditRiskPreprocessor()
    X_train_prep = preprocessor.fit_transform(df_train)
    X_calib_prep = preprocessor.transform(df_calib)
    X_test_prep = preprocessor.transform(df_test)

    feature_names = preprocessor.feature_names_
    logger.info(f"Engineered {len(feature_names)} features: {feature_names}")

    # Step 4: Handle Class Imbalance via SMOTE
    logger.info("Step 4: Balancing classes with SMOTE on training partition...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_prep.values, y_train)
    logger.info(f"Post-SMOTE class distribution: {np.bincount(y_train_res)}")

    # Step 5: Multi-Model Ensemble Benchmarking
    logger.info("Step 5: Training and benchmarking LightGBM, XGBoost, and CatBoost...")

    candidates = {
        "LightGBM": LGBMClassifier(
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.04,
            depth=5,
            subsample=0.85,
            random_seed=random_state,
            verbose=0,
        ),
    }

    benchmark_results = []
    fitted_models = {}

    for name, model in candidates.items():
        logger.info(f"Fitting candidate algorithm: {name}...")
        model.fit(X_train_res, y_train_res)
        fitted_models[name] = model

        eval_res = evaluate_model(model, X_test_prep.values, y_test, model_name=name)
        benchmark_results.append(eval_res)
        logger.info(
            f"Candidate {name} -> ROC-AUC: {eval_res['roc_auc']:.4f}, "
            f"PR-AUC: {eval_res['pr_auc']:.4f}, Brier: {eval_res['brier_score']:.4f}"
        )

    # Select winning model based on highest ROC-AUC + PR-AUC score
    best_candidate_name = max(benchmark_results, key=lambda x: (x["roc_auc"] + x["pr_auc"]))["model"]
    best_base_model = fitted_models[best_candidate_name]
    logger.info(f"Winning base ensemble: {best_candidate_name}")

    # Step 6: Probability Calibration with Isotonic Regression
    logger.info("Step 6: Calibrating winning model with Isotonic Regression...")
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_base_model),
            method="isotonic",
        )
    except (ImportError, TypeError, ValueError):
        calibrated_model = CalibratedClassifierCV(
            estimator=best_base_model,
            method="isotonic",
            cv="prefit",
        )

    calibrated_model.fit(X_calib_prep.values, y_calib)

    # Evaluate Calibrated Model
    calib_eval = evaluate_model(
        calibrated_model,
        X_test_prep.values,
        y_test,
        model_name=f"{best_candidate_name} (Isotonic Calibrated)",
    )
    logger.info(
        f"Calibrated Model -> ROC-AUC: {calib_eval['roc_auc']:.4f}, "
        f"PR-AUC: {calib_eval['pr_auc']:.4f}, Brier: {calib_eval['brier_score']:.4f}"
    )
    benchmark_results.append(calib_eval)

    # Reliability Diagram Comparison (Calibration Curves)
    prob_true_raw, prob_pred_raw = calibration_curve(
        y_test, best_base_model.predict_proba(X_test_prep.values)[:, 1], n_bins=10
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, calibrated_model.predict_proba(X_test_prep.values)[:, 1], n_bins=10
    )

    calibration_curves_data = {
        "raw": {"prob_pred": prob_pred_raw.tolist(), "prob_true": prob_true_raw.tolist()},
        "calibrated": {"prob_pred": prob_pred_cal.tolist(), "prob_true": prob_true_cal.tolist()},
    }

    # Step 7: Predictions on Test Cohort for Cockpit
    y_test_pred_proba = calibrated_model.predict_proba(X_test_prep.values)[:, 1]
    y_test_raw_proba = best_base_model.predict_proba(X_test_prep.values)[:, 1]

    # Credit risk tiers and system recommendation
    def get_tier_and_action(prob: float) -> Tuple[str, str]:
        if prob < 0.25:
            return "Low Risk", "Approve"
        elif prob <= 0.50:
            return "Medium Risk", "Manual Underwrite"
        else:
            return "High Risk", "Reject"

    tiers, actions = zip(*[get_tier_and_action(p) for p in y_test_pred_proba])

    # Enrich df_test with calculated ratios and decision scores
    df_test_enriched = df_test.copy()
    df_test_enriched["AGE_YEARS"] = np.round(np.abs(df_test["DAYS_BIRTH"]) / 365.25, 1)
    days_emp_clean = np.where(df_test["DAYS_EMPLOYED"] == 365243, 0, np.abs(df_test["DAYS_EMPLOYED"]))
    df_test_enriched["EMPLOYMENT_YEARS"] = np.round(days_emp_clean / 365.25, 1)
    df_test_enriched["CREDIT_TO_INCOME_RATIO"] = np.round(df_test["AMT_CREDIT"] / df_test["AMT_INCOME_TOTAL"], 2)
    df_test_enriched["ANNUITY_TO_INCOME_RATIO"] = np.round(df_test["AMT_ANNUITY"] / df_test["AMT_INCOME_TOTAL"], 3)
    df_test_enriched["CREDIT_TERM"] = np.round(df_test["AMT_ANNUITY"] / df_test["AMT_CREDIT"], 4)

    df_test_enriched["CALIBRATED_PROB_DEFAULT"] = np.round(y_test_pred_proba, 4)
    df_test_enriched["RAW_PROB_DEFAULT"] = np.round(y_test_raw_proba, 4)
    df_test_enriched["RISK_TIER"] = list(tiers)
    df_test_enriched["RECOMMENDATION"] = list(actions)

    # Step 8: Save Test Dataset to Parquet
    test_parquet_path = os.path.join(output_dir, "test_data.parquet")
    df_test_enriched.to_parquet(test_parquet_path, index=False)
    logger.info(f"Saved test dataset to {test_parquet_path}")

    # Also save preprocessed test features dataframe
    test_prep_parquet = os.path.join(output_dir, "test_features_prep.parquet")
    X_test_prep.to_parquet(test_prep_parquet, index=False)

    # Step 9: Setup Explainability (SHAP & LIME)
    logger.info("Step 9: Setting up SHAP and LIME explainer artifacts...")
    setup_explainers(
        base_tree_model=best_base_model,
        X_train_transformed=X_train_prep,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    # Step 10: Regulatory Fairness Audit
    logger.info("Step 10: Executing Algorithmic Fairness Audit on Test Cohort...")
    fairness_report = perform_fairness_audit(
        y_true=y_test,
        y_pred_proba=y_test_pred_proba,
        df_raw=df_test,
        threshold=0.35,
        output_path=os.path.join(output_dir, "fairness_report.json"),
    )

    # Step 11: Serialize Model & Pipeline Artifacts
    logger.info("Step 11: Persisting models and metadata...")
    joblib.dump(calibrated_model, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(best_base_model, os.path.join(output_dir, "base_model.pkl"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))

    metrics_payload = {
        "best_algorithm": best_candidate_name,
        "benchmark_comparison": benchmark_results,
        "calibration_curves": calibration_curves_data,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "test_sample_size": len(y_test),
        "test_default_rate": round(float(np.mean(y_test)), 4),
    }

    with open(os.path.join(output_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    logger.info("Training, calibration, explainability, and fairness pipeline complete!")
    return metrics_payload


if __name__ == "__main__":
    train_and_benchmark()
