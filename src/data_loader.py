"""
Data Loader and Synthetic Generator for Home Credit Default Risk.
Generates realistic financial loan applicant data conforming to the Kaggle Home Credit schema,
performs automated validation checks, and handles raw dataset caching.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Schema definition for validation
EXPECTED_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
]

EDUCATION_LEVELS = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]


def generate_synthetic_home_credit_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic loan applicant dataset matching the Kaggle Home Credit Default Risk schema.
    Incorporates empirical domain distributions, external credit bureau scores, and realistic default probabilities.
    """
    rng = np.random.default_rng(random_state)
    logger.info(f"Generating synthetic Home Credit dataset with {n_samples} applicants...")

    sk_id_curr = np.arange(100001, 100001 + n_samples)

    # Demographics
    gender_prob = [0.65, 0.35]  # ~65% F, 35% M like Home Credit
    code_gender = rng.choice(["F", "M"], size=n_samples, p=gender_prob)

    edu_prob = [0.71, 0.24, 0.035, 0.012, 0.003]
    name_education = rng.choice(EDUCATION_LEVELS, size=n_samples, p=edu_prob)

    # Age in days (negative): age between 21 and 68
    age_years = rng.uniform(21, 68, size=n_samples)
    days_birth = -np.round(age_years * 365.25).astype(int)

    # Employment days (negative): employment years up to (age - 18)
    max_emp_years = np.maximum(0.5, age_years - 18)
    emp_years = rng.exponential(scale=5.0, size=n_samples)
    emp_years = np.minimum(emp_years, max_emp_years)
    # ~15% unemployed / pensioners (Home Credit sentinel 365243)
    is_pensioner = (age_years > 58) & (rng.random(size=n_samples) < 0.5)
    days_employed = -np.round(emp_years * 365.25).astype(int)
    days_employed[is_pensioner] = 365243

    # Financials (Lognormal income: median ~145,000, realistic skew)
    income_log = rng.normal(loc=11.85, scale=0.55, size=n_samples)
    amt_income_total = np.round(np.exp(income_log) / 1000) * 1000
    amt_income_total = np.clip(amt_income_total, 25000, 1500000)

    # Credit amount: typically 2x to 6x annual income, median ~500,000
    credit_multiplier = rng.uniform(1.8, 5.5, size=n_samples)
    amt_credit = np.round((amt_income_total * credit_multiplier) / 5000) * 5000
    amt_credit = np.clip(amt_credit, 45000, 3500000)

    # Goods price: typically close to or slightly less than credit amount
    goods_discount = rng.uniform(0.85, 1.0, size=n_samples)
    amt_goods_price = np.round((amt_credit * goods_discount) / 5000) * 5000

    # Annuity: loan payment (typically 4% to 10% of credit amount)
    annuity_rate = rng.uniform(0.038, 0.085, size=n_samples)
    amt_annuity = np.round((amt_credit * annuity_rate) / 100) * 100

    # External Bureau Scores (Beta distributions scaled [0, 1])
    ext_source_1 = rng.beta(a=3.0, b=3.0, size=n_samples)
    ext_source_2 = rng.beta(a=3.5, b=2.8, size=n_samples)
    ext_source_3 = rng.beta(a=2.8, b=3.2, size=n_samples)

    # Realistic missingness patterns (similar to Kaggle Home Credit)
    mask_ext_1 = rng.random(size=n_samples) < 0.25
    mask_ext_3 = rng.random(size=n_samples) < 0.15
    ext_source_1_clean = np.where(mask_ext_1, np.nan, ext_source_1)
    ext_source_3_clean = np.where(mask_ext_3, np.nan, ext_source_3)

    # Realistic Ground Truth Risk Function (Latent Log-Odds)
    # Ratios
    annuity_to_income = amt_annuity / amt_income_total
    credit_to_income = amt_credit / amt_income_total
    valid_emp_years = np.where(days_employed == 365243, 0, np.abs(days_employed) / 365.25)
    mean_ext = np.nanmean([ext_source_1, ext_source_2, ext_source_3], axis=0)

    # Risk latent score components (Mirroring empirical Home Credit risk drivers)
    base_log_odds = -2.40  # Natural default rate ~8.5%
    ext_effect = -4.8 * (mean_ext - 0.5)  # External bureau scores are primary driver
    dti_effect = 4.2 * (annuity_to_income - 0.22)  # High debt burden increases default
    credit_ratio_effect = 0.45 * (credit_to_income - 3.2)  # High credit-to-income increases default
    emp_effect = -0.12 * np.minimum(valid_emp_years, 12)  # Stable employment reduces default
    age_effect = np.where(age_years < 28, 0.45, 0.0)  # Younger borrowers have less credit maturity
    edu_discount = np.where(name_education == "Higher education", -0.40, 0.0)  # Education credit stability
    noise = rng.normal(loc=0.0, scale=0.35, size=n_samples)

    latent_z = (
        base_log_odds
        + ext_effect
        + dti_effect
        + credit_ratio_effect
        + emp_effect
        + age_effect
        + edu_discount
        + noise
    )
    prob_default = 1.0 / (1.0 + np.exp(-latent_z))

    # Binary outcome based on Bernoulli draw
    target = (rng.random(size=n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr,
        "TARGET": target,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": ext_source_1_clean,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3_clean,
        "CODE_GENDER": code_gender,
        "NAME_EDUCATION_TYPE": name_education,
    })

    logger.info(f"Generated synthetic dataset: {df.shape[0]} rows, default rate = {df['TARGET'].mean():.2%}")
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates dataset integrity, schema conformity, value ranges, and missingness metrics.
    """
    validation_report = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": [],
        "null_percentages": {},
        "target_distribution": {},
        "warnings": [],
    }

    # Column presence check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        validation_report["is_valid"] = False
        validation_report["missing_columns"] = missing_cols
        logger.error(f"Dataset validation failed: missing columns {missing_cols}")
        return False, validation_report

    # Null percentage check
    for col in EXPECTED_COLUMNS:
        null_pct = df[col].isnull().mean()
        validation_report["null_percentages"][col] = round(null_pct, 4)
        if col in ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "CODE_GENDER"] and null_pct > 0.05:
            validation_report["warnings"].append(f"Critical column {col} has unexpected null rate: {null_pct:.2%}")

    # Range and sanity checks
    if (df["AMT_INCOME_TOTAL"] <= 0).any():
        validation_report["warnings"].append("Non-positive AMT_INCOME_TOTAL values detected.")
    if (df["AMT_CREDIT"] <= 0).any():
        validation_report["warnings"].append("Non-positive AMT_CREDIT values detected.")

    # Target sanity
    target_counts = df["TARGET"].value_counts(normalize=True).to_dict()
    validation_report["target_distribution"] = {str(k): round(v, 4) for k, v in target_counts.items()}
    if df["TARGET"].nunique() != 2:
        validation_report["is_valid"] = False
        validation_report["warnings"].append("TARGET column must contain exactly binary classes (0 and 1).")

    logger.info(f"Validation completed. Status: {'PASS' if validation_report['is_valid'] else 'FAIL'}")
    return validation_report["is_valid"], validation_report


def load_data(
    data_dir: str = "data",
    n_samples: int = 10000,
    force_generate: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Main entrypoint: Loads application_train.csv if present, or raw_data.csv,
    or generates a synthetic dataset matching Kaggle Home Credit Default Risk schema.
    """
    os.makedirs(data_dir, exist_ok=True)
    real_csv = os.path.join(data_dir, "application_train.csv")
    raw_csv = os.path.join(data_dir, "raw_data.csv")

    if not force_generate and os.path.exists(real_csv):
        logger.info(f"Loading existing Kaggle dataset from {real_csv}...")
        df = pd.read_csv(real_csv)
    elif not force_generate and os.path.exists(raw_csv):
        logger.info(f"Loading existing raw dataset from {raw_csv}...")
        df = pd.read_csv(raw_csv)
    else:
        logger.info("Generating new baseline synthetic dataset...")
        df = generate_synthetic_home_credit_data(n_samples=n_samples, random_state=random_state)
        df.to_csv(raw_csv, index=False)
        logger.info(f"Saved generated raw dataset to {raw_csv}")

    # Validate dataset
    is_valid, report = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset failed automated validation: {report}")

    return df


if __name__ == "__main__":
    df_sample = load_data(force_generate=True, n_samples=10000)
    print("\nDataset Summary:")
    print(df_sample.info())
    print("\nTarget Class Distribution:")
    print(df_sample["TARGET"].value_counts(normalize=True))
    print("\nSample Rows:")
    print(df_sample.head(3))
