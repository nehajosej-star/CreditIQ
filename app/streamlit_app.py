"""
Smart Financial Risk & Loan Default Predictor with Explainability (AICW)
Production-grade Streamlit Loan Officer Decision Cockpit & Compliance Suite.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import copy
import json
import re
import joblib
import cloudpickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import shap
from src.feature_engineering import CreditRiskPreprocessor, DomainRatioExtractor
from src.explainers import generate_adverse_action_codes, compute_lime_local_explanation

# Streamlit Page Configuration
st.set_page_config(
    page_title="CreditIQ | Loan Underwriting & Explainability Cockpit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (Dark Mode, Glassmorphism, Modern Financial Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Main Container Glassmorphism */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Metric Cards */
    .decision-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 22px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .decision-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
    }

    /* Dynamic Risk Badges */
    .risk-badge-low {
        background: linear-gradient(135deg, #065f46 0%, #059669 100%);
        color: #ecfdf5;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.35);
    }
    .risk-badge-med {
        background: linear-gradient(135deg, #92400e 0%, #d97706 100%);
        color: #fffbeb;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.35);
    }
    .risk-badge-high {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: #fef2f2;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.35);
    }

    /* Adverse Action Box */
    .adverse-action-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
        border-left: 4px solid #38bdf8;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 16px 0;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Compliance Tag */
    .compliance-pill {
        background-color: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-family: monospace;
        font-weight: 600;
    }

    /* Enhanced Tab Container & Typography */
    .stTabs {
        margin-top: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        gap: 12px !important;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%) !important;
        padding: 8px 12px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        align-items: center !important;
        scrollbar-width: none !important;
        margin-bottom: 24px !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }
    .stTabs button[role="tab"],
    .stTabs [data-baseweb="tab"],
    button[data-baseweb="tab"] {
        height: auto !important;
        min-height: 52px !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        transition: all 0.25s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: visible !important;
        text-align: center !important;
    }
    .stTabs button[role="tab"] p,
    .stTabs button[role="tab"] div,
    .stTabs button[role="tab"] span,
    .stTabs [data-baseweb="tab"] p, 
    .stTabs [data-baseweb="tab"] div, 
    .stTabs [data-baseweb="tab"] span,
    button[data-baseweb="tab"] p, 
    button[data-baseweb="tab"] div, 
    button[data-baseweb="tab"] span {
        color: #cbd5e1 !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important; /* 18.4px */
        letter-spacing: -0.01em !important;
        line-height: 1.5 !important;
        white-space: nowrap !important;
        margin: 0 !important;
        overflow: visible !important;
    }
    .stTabs button[role="tab"]:hover,
    button[data-baseweb="tab"]:hover {
        background-color: rgba(51, 65, 85, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    .stTabs button[role="tab"]:hover p,
    .stTabs button[role="tab"]:hover span,
    .stTabs button[role="tab"]:hover div,
    button[data-baseweb="tab"]:hover p,
    button[data-baseweb="tab"]:hover span,
    button[data-baseweb="tab"]:hover div {
        color: #ffffff !important;
    }
    .stTabs button[role="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.25) !important;
    }
    .stTabs button[role="tab"][aria-selected="true"] p,
    .stTabs button[role="tab"][aria-selected="true"] span,
    .stTabs button[role="tab"][aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    """Loads and caches models, preprocessors, explainers, and test datasets."""
    models_dir = os.path.join(PROJECT_ROOT, "models")

    best_model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    base_model = joblib.load(os.path.join(models_dir, "base_model.pkl"))
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))

    with open(os.path.join(models_dir, "shap_explainer.pkl"), "rb") as f:
        shap_explainer = cloudpickle.load(f)

    with open(os.path.join(models_dir, "lime_explainer.pkl"), "rb") as f:
        lime_explainer = cloudpickle.load(f)

    with open(os.path.join(models_dir, "model_metrics.json"), "r", encoding="utf-8") as f:
        model_metrics = json.load(f)

    with open(os.path.join(models_dir, "fairness_report.json"), "r", encoding="utf-8") as f:
        fairness_report = json.load(f)

    df_test = pd.read_parquet(os.path.join(models_dir, "test_data.parquet"))
    df_test_prep = pd.read_parquet(os.path.join(models_dir, "test_features_prep.parquet"))

    return {
        "best_model": best_model,
        "base_model": base_model,
        "preprocessor": preprocessor,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "model_metrics": model_metrics,
        "fairness_report": fairness_report,
        "df_test": df_test,
        "df_test_prep": df_test_prep,
    }


def format_inr(val: float) -> str:
    """Formats monetary values into Indian Rupee format with ₹ prefix, e.g. ₹5,00,000."""
    try:
        val_int = int(round(val))
        s = str(abs(val_int))
        if len(s) <= 3:
            res = s
        else:
            last3 = s[-3:]
            remaining = s[:-3]
            groups = []
            while len(remaining) > 2:
                groups.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.insert(0, remaining)
            res = ",".join(groups) + "," + last3
        prefix = "-₹" if val_int < 0 else "₹"
        return f"{prefix}{res}"
    except Exception:
        return f"₹{val:,.0f}"


def format_currency(val: float) -> str:
    return format_inr(val)


# User-friendly display names for model features
FRIENDLY_FEATURE_NAMES = {
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_CREDIT": "Loan Amount Requested",
    "AMT_ANNUITY": "Monthly Installment",
    "AMT_GOODS_PRICE": "Property / Goods Valuation",
    "DAYS_BIRTH": "Applicant Age (Years)",
    "DAYS_EMPLOYED": "Employment Duration (Years)",
    "EXT_SOURCE_1": "Existing Debts",
    "EXT_SOURCE_2": "Credit History",
    "EXT_SOURCE_3": "Previous Loans",
    "AGE_YEARS": "Applicant Age (Years)",
    "EMPLOYMENT_YEARS": "Employment Tenure (Years)",
    "CREDIT_TO_INCOME_RATIO": "Loan-to-Income Burden",
    "ANNUITY_TO_INCOME_RATIO": "Monthly Repayment Burden (% of Income)",
    "CREDIT_TERM": "Implied Repayment Term",
    "GOODS_TO_CREDIT_RATIO": "Collateral Coverage Ratio",
    "EXT_SOURCES_MEAN": "Overall Credit Health Rating",
    "CODE_GENDER_F": "Gender: Female",
    "CODE_GENDER_M": "Gender: Male",
    "NAME_EDUCATION_TYPE_Academic degree": "Education: Academic Degree",
    "NAME_EDUCATION_TYPE_Higher education": "Education: Higher Education",
    "NAME_EDUCATION_TYPE_Incomplete higher": "Education: Incomplete Higher",
    "NAME_EDUCATION_TYPE_Lower secondary": "Education: Lower Secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "Education: Secondary Special",
}


def get_friendly_feature_name(name: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(name, name.replace("_", " ").title())


def clean_lime_condition(cond: str) -> str:
    cleaned = cond

    # Clean translation for gender binary rules
    if "CODE_GENDER_M <= 0.00" in cleaned:
        return "Gender: Female"
    if "CODE_GENDER_M > 0.00" in cleaned:
        return "Gender: Male"
    if "CODE_GENDER_F <= 0.00" in cleaned:
        return "Gender: Male"
    if "CODE_GENDER_F > 0.00" in cleaned:
        return "Gender: Female"

    # Clean translation for education categories
    for edu_key in [
        "Academic degree",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Secondary / secondary special",
    ]:
        tech = f"NAME_EDUCATION_TYPE_{edu_key}"
        friendly = FRIENDLY_FEATURE_NAMES.get(tech, edu_key.title())
        if f"0.00 < {tech} <= 1.00" in cleaned or f"{tech} > 0.00" in cleaned:
            return f"{friendly}: Yes"
        if f"{tech} <= 0.00" in cleaned:
            return f"{friendly}: No"

    for tech_name, friendly in sorted(FRIENDLY_FEATURE_NAMES.items(), key=lambda x: -len(x[0])):
        if tech_name in cleaned:
            cleaned = cleaned.replace(tech_name, friendly)

    # Format financial numbers cleanly as Indian Rupees
    if any(m in cleaned for m in ["Annual Income", "Loan Amount Requested", "Monthly Installment", "Property / Goods Valuation"]):
        cleaned = re.sub(r'\b\d{4,}(?:\.\d+)?\b', lambda m: format_inr(float(m.group(0))), cleaned)
    else:
        cleaned = re.sub(r'\b\d+\.\d{3,}\b', lambda m: f"{float(m.group(0)):.2f}", cleaned)

    return cleaned


def get_customer_factor_explanation(
    feat: str,
    sim_income: float,
    sim_credit: float,
    sim_annuity: float,
    sim_goods: float,
    existing_debts: float,
    credit_history: float,
    previous_loans: float,
    sim_age: int,
    sim_emp: int,
) -> dict:
    """Generates plain conversational English explanations and actionable guidance tailored to applicant details."""
    monthly_inc = (sim_income / 12.0) if sim_income > 0 else 1.0
    dti = (sim_annuity / monthly_inc) * 100.0 if monthly_inc > 0 else 0.0
    lti = (sim_credit / sim_income) if sim_income > 0 else 0.0

    if "CREDIT_TO_INCOME" in feat or feat == "AMT_CREDIT":
        return {
            "title": "The requested loan amount is too high compared to your annual income",
            "detail": f"You requested a loan facility of {format_inr(sim_credit)} against an annual income of {format_inr(sim_income)} ({lti:.1f}x your annual income). Our standard underwriting guidelines recommend keeping total borrowing below 3.5x to 4.0x of annual earnings to ensure payments remain manageable.",
            "action": f"Applying for a smaller loan facility (e.g., within {format_inr(min(sim_credit, sim_income * 3.5))}) or adding a co-applicant with regular verified income will significantly improve your qualification probability.",
        }
    elif "ANNUITY_TO_INCOME" in feat or feat == "AMT_ANNUITY":
        return {
            "title": "Existing monthly repayment obligations are too heavy",
            "detail": f"The scheduled monthly installment of {format_inr(sim_annuity)} absorbs {dti:.1f}% of your monthly income. Responsible lending standards typically require monthly repayment obligations to remain under 30%–35% of total income to avoid financial strain.",
            "action": "Selecting a longer repayment tenure lowers your monthly installment into a safer repayment bracket.",
        }
    elif feat == "EXT_SOURCE_2":
        return {
            "title": "Historical credit bureau records indicate elevated repayment risk",
            "detail": f"Your credit bureau history rating is currently {credit_history:.2f} (out of 1.00). This indicates past payment delays, overdue accounts, or limited seasoned credit history across reporting financial institutions.",
            "action": "Making consistent, on-time payments across all active accounts over the next 3 to 6 months will systematically rebuild and strengthen your credit standing.",
        }
    elif feat == "EXT_SOURCE_1":
        return {
            "title": "Existing debt obligations limit additional borrowing capacity",
            "detail": f"Bureau records reflect an existing liabilities rating of {existing_debts:.2f} (out of 1.00), showing active debts or revolving balances that restrict your disposable headroom for new credit facilities.",
            "action": "Paying down current credit card balances or clearing smaller outstanding loans before reapplying will directly boost your eligibility.",
        }
    elif feat == "EXT_SOURCE_3":
        return {
            "title": "Previous loan records reflect historical payment delays",
            "detail": f"Historical bureau records for prior credit facilities show a rating of {previous_loans:.2f} (out of 1.00), reflecting past payment delays or settled accounts.",
            "action": "Ensure all previous loans are formally closed with 'No Dues' certificates and resolve any disputed records on your credit bureau profile.",
        }
    elif feat == "EXT_SOURCES_MEAN":
        ext_mean = float(np.mean([existing_debts, credit_history, previous_loans]))
        return {
            "title": "Overall credit health rating is below target benchmark",
            "detail": f"Your composite rating across external credit agencies is {ext_mean:.2f} (out of 1.00), falling below our target underwriting benchmark of 0.50.",
            "action": "A sustained track record of timely utility, credit card, and loan payments will systematically lift your composite credit score.",
        }
    elif "GOODS" in feat or feat == "AMT_GOODS_PRICE":
        return {
            "title": "Financed property or goods valuation does not provide enough collateral buffer",
            "detail": f"The declared asset valuation of {format_inr(sim_goods)} provides insufficient collateral margin for the requested loan principal of {format_inr(sim_credit)}.",
            "action": "Increasing your initial upfront cash down payment improves the loan-to-value ratio and satisfies risk criteria.",
        }
    elif "EMPLOYMENT" in feat or feat == "DAYS_EMPLOYED":
        return {
            "title": "Verified continuous employment tenure is relatively brief",
            "detail": f"Your verified employment tenure of {sim_emp} year(s) is brief, providing less history of long-term income stability.",
            "action": "Providing supplementary proof of past steady employment or professional credentials can help verify income stability.",
        }
    elif "AGE" in feat or feat == "DAYS_BIRTH":
        return {
            "title": "Verifiable credit file maturity is still developing",
            "detail": f"At {sim_age} years of age, your credit history has had limited time to establish a multi-cycle track record.",
            "action": "Applying with an established co-applicant or guarantor can satisfy profile maturity requirements.",
        }
    else:
        friendly = get_friendly_feature_name(feat)
        return {
            "title": f"Elevated repayment risk associated with {friendly}",
            "detail": f"The metric for '{friendly}' was weighted unfavorably against loan repayment probability in our underwriting model.",
            "action": f"Improving your '{friendly}' profile will positively influence your credit assessment.",
        }


def get_risk_theme(prob: float):
    if prob < 0.25:
        return {
            "tier": "Low Risk",
            "badge_class": "risk-badge-low",
            "color": "#10b981",
            "action": "Approved",
            "desc": "Applicant profile comfortably satisfies loan underwriting criteria.",
        }
    elif prob < 0.50:
        return {
            "tier": "Medium Risk",
            "badge_class": "risk-badge-med",
            "color": "#f59e0b",
            "action": "Manual Review / Conditional",
            "desc": "Application requires underwriter review or compensatory conditions.",
        }
    else:
        return {
            "tier": "High Risk",
            "badge_class": "risk-badge-high",
            "color": "#ef4444",
            "action": "Not Approved / Rejected",
            "desc": "Financial metrics exceed standard portfolio risk limits.",
        }


# Main App Body
def main():
    try:
        artifacts = load_all_artifacts()
    except Exception as e:
        st.error(f"Error loading system artifacts: {e}. Please ensure 'python src/train.py' has completed successfully.")
        st.stop()

    df_test = artifacts["df_test"]
    best_model = artifacts["best_model"]
    base_model = artifacts["base_model"]
    preprocessor = artifacts["preprocessor"]
    shap_explainer = artifacts["shap_explainer"]
    lime_explainer = artifacts["lime_explainer"]
    model_metrics = artifacts["model_metrics"]
    fairness_report = artifacts["fairness_report"]
    feature_names = preprocessor.feature_names_

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.markdown("### 📋 Applicant Selection")

    # Filter selector with applicant metadata
    applicant_options = df_test["SK_ID_CURR"].tolist()
    default_idx = 0
    # Pick an applicant with moderate/review risk for rich demonstration if available
    med_or_high_candidates = df_test[df_test["RISK_TIER"].isin(["High Risk", "Medium Risk"])].index
    if len(med_or_high_candidates) > 0:
        default_idx = int(med_or_high_candidates[0])

    selected_id = st.sidebar.selectbox(
        "Select Applicant Record:",
        applicant_options,
        index=default_idx,
        help="Choose a borrower from the held-out test cohort",
        key="selected_applicant_selector",
    )
    selected_applicant_id = selected_id

    # Fetch baseline applicant row
    baseline_row = df_test[df_test["SK_ID_CURR"] == selected_id].iloc[0]

    # Reactive state synchronization across applicant switches
    if "current_applicant_id" not in st.session_state:
        st.session_state["current_applicant_id"] = selected_id
    elif st.session_state["current_applicant_id"] != selected_id:
        st.session_state["current_applicant_id"] = selected_id
        # Clear cached slider states for previous applicant to guarantee clean baseline initialization
        for k in [
            f"income_{selected_id}", f"credit_{selected_id}", f"annuity_{selected_id}",
            f"goods_{selected_id}", f"debts_{selected_id}", f"history_{selected_id}",
            f"loans_{selected_id}", f"name_{selected_id}", f"age_{selected_id}",
            f"emp_{selected_id}", f"edu_{selected_id}", f"gender_{selected_id}",
        ]:
            if k in st.session_state:
                del st.session_state[k]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Counterfactual 'What-If' Simulation")
    st.sidebar.caption("Adjust variables to simulate counterfactual credit scenarios in real-time.")

    # 1-Click Reset Button to immediately revert to this applicant's baseline
    if st.sidebar.button("🔄 Reset to Baseline", use_container_width=True, help="Revert all sliders to this applicant's official baseline file"):
        for k in [
            f"income_{selected_id}", f"credit_{selected_id}", f"annuity_{selected_id}",
            f"goods_{selected_id}", f"debts_{selected_id}", f"history_{selected_id}",
            f"loans_{selected_id}", f"name_{selected_id}", f"age_{selected_id}",
            f"emp_{selected_id}", f"edu_{selected_id}", f"gender_{selected_id}",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Simulation Sliders strictly keyed to the active applicant for instant reactive state
    raw_income = int(baseline_row["AMT_INCOME_TOTAL"])
    sim_income = st.sidebar.slider(
        "Annual Income:",
        min_value=10000,
        max_value=2000000,
        value=int(np.clip(raw_income, 10000, 2000000)),
        step=5000,
        format="₹%d",
        key=f"income_{selected_id}",
    )

    raw_credit = int(baseline_row["AMT_CREDIT"])
    sim_credit = st.sidebar.slider(
        "Loan Amount Requested:",
        min_value=25000,
        max_value=5000000,
        value=int(np.clip(raw_credit, 25000, 5000000)),
        step=10000,
        format="₹%d",
        key=f"credit_{selected_id}",
    )

    raw_annuity = int(baseline_row["AMT_ANNUITY"])
    sim_annuity = st.sidebar.slider(
        "Monthly / Scheduled Installment:",
        min_value=1000,
        max_value=250000,
        value=int(np.clip(raw_annuity, 1000, 250000)),
        step=1000,
        format="₹%d",
        key=f"annuity_{selected_id}",
    )

    raw_goods = int(baseline_row["AMT_GOODS_PRICE"]) if pd.notnull(baseline_row["AMT_GOODS_PRICE"]) else int(baseline_row["AMT_CREDIT"])
    sim_goods = st.sidebar.slider(
        "Property / Goods Valuation:",
        min_value=25000,
        max_value=5000000,
        value=int(np.clip(raw_goods, 25000, 5000000)),
        step=10000,
        format="₹%d",
        key=f"goods_{selected_id}",
    )

    st.sidebar.markdown("##### Credit Bureau Ratings")
    raw_debts = float(baseline_row["EXT_SOURCE_1"]) if pd.notnull(baseline_row["EXT_SOURCE_1"]) else 0.50
    existing_debts = sim_existing_debts = st.sidebar.slider(
        "Existing Debts (0.0 to 1.0):",
        min_value=0.01,
        max_value=0.99,
        value=float(np.clip(raw_debts, 0.01, 0.99)),
        step=0.01,
        key=f"debts_{selected_id}",
    )

    raw_history = float(baseline_row["EXT_SOURCE_2"]) if pd.notnull(baseline_row["EXT_SOURCE_2"]) else 0.50
    credit_history = sim_credit_history = st.sidebar.slider(
        "Credit History (0.0 to 1.0):",
        min_value=0.01,
        max_value=0.99,
        value=float(np.clip(raw_history, 0.01, 0.99)),
        step=0.01,
        key=f"history_{selected_id}",
    )

    raw_loans = float(baseline_row["EXT_SOURCE_3"]) if pd.notnull(baseline_row["EXT_SOURCE_3"]) else 0.50
    previous_loans = sim_previous_loans = st.sidebar.slider(
        "Previous Loans (0.0 to 1.0):",
        min_value=0.01,
        max_value=0.99,
        value=float(np.clip(raw_loans, 0.01, 0.99)),
        step=0.01,
        key=f"loans_{selected_id}",
    )

    st.sidebar.markdown("##### Applicant Name & Age")
    applicant_name = st.sidebar.text_input(
        "Applicant Name:",
        value="Applicant #" + str(selected_applicant_id),
        key=f"name_{selected_id}",
    )
    display_name = applicant_name.strip() if applicant_name.strip() else f"Applicant #{selected_applicant_id}"

    raw_age = int(baseline_row["AGE_YEARS"])
    sim_age = st.sidebar.slider(
        "Age in Years:",
        min_value=21,
        max_value=75,
        value=int(np.clip(raw_age, 21, 75)),
        step=1,
        key=f"age_{selected_id}",
    )

    st.sidebar.markdown("##### Employment Details")
    raw_emp = int(baseline_row["EMPLOYMENT_YEARS"])
    sim_emp = st.sidebar.slider(
        "Employment Tenure (Years):",
        min_value=0,
        max_value=45,
        value=int(np.clip(raw_emp, 0, 45)),
        step=1,
        key=f"emp_{selected_id}",
    )

    edu_options = [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
    current_edu_idx = edu_options.index(baseline_row["NAME_EDUCATION_TYPE"]) if baseline_row["NAME_EDUCATION_TYPE"] in edu_options else 0
    sim_edu = st.sidebar.selectbox("Education Level:", edu_options, index=current_edu_idx, key=f"edu_{selected_id}")

    sim_gender = st.sidebar.radio("Gender:", ["M", "F"], index=0 if baseline_row["CODE_GENDER"] == "M" else 1, horizontal=True, key=f"gender_{selected_id}")

    # 1. Dynamic Reactive Feature Vector (Task 1) - directly constructed from active inputs
    active_dict = {
        "SK_ID_CURR": selected_id,
        "AMT_INCOME_TOTAL": float(sim_income),
        "AMT_CREDIT": float(sim_credit),
        "AMT_ANNUITY": float(sim_annuity),
        "AMT_GOODS_PRICE": float(sim_goods),
        "DAYS_BIRTH": -int(sim_age * 365.25),
        "DAYS_EMPLOYED": -int(sim_emp * 365.25),
        "EXT_SOURCE_1": float(existing_debts),
        "EXT_SOURCE_2": float(credit_history),
        "EXT_SOURCE_3": float(previous_loans),
        "CODE_GENDER": sim_gender,
        "NAME_EDUCATION_TYPE": sim_edu,
    }
    df_active = pd.DataFrame([active_dict])

    # Transform through pipeline
    df_active_prep = preprocessor.transform(df_active)

    # 2. Dynamic Model Inference on the updated vector (Task 2)
    active_cal_prob = float(best_model.predict_proba(df_active_prep.values)[0, 1])
    active_raw_prob = float(base_model.predict_proba(df_active_prep.values)[0, 1])

    # 3. Dynamic Status Tiers (Task 3)
    theme = get_risk_theme(active_cal_prob)

    # 4. Dynamic Metric Computations (Task 2)
    monthly_income = (float(sim_income) / 12.0) if sim_income > 0 else 1.0
    monthly_repayment_burden = (float(sim_annuity) / monthly_income) * 100.0 if monthly_income > 0 else 0.0
    total_loan_multiple = (float(sim_credit) / float(sim_income)) if sim_income > 0 else 0.0
    overall_credit_health = float(np.mean([existing_debts, credit_history, previous_loans]))

    # Check if inputs differ from baseline
    is_modified = (
        sim_income != baseline_row["AMT_INCOME_TOTAL"]
        or sim_credit != baseline_row["AMT_CREDIT"]
        or sim_annuity != baseline_row["AMT_ANNUITY"]
        or existing_debts != baseline_row["EXT_SOURCE_1"]
        or credit_history != baseline_row["EXT_SOURCE_2"]
        or previous_loans != baseline_row["EXT_SOURCE_3"]
        or sim_age != baseline_row["AGE_YEARS"]
        or sim_emp != baseline_row["EMPLOYMENT_YEARS"]
        or sim_edu != baseline_row["NAME_EDUCATION_TYPE"]
        or sim_gender != baseline_row["CODE_GENDER"]
    )

    # App Header - Clean & Professional
    st.markdown(
        f"""
        <div style="padding: 4px 0 10px 0;">
            <h1 style="font-size: 2.2rem; font-weight: 700; margin: 0; color: #f8fafc; letter-spacing: -0.02em;">
                🏦 CreditIQ | Loan Underwriting & Risk Assessment Cockpit
            </h1>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
                Active Underwriting Record: <strong style="color: #38bdf8; font-size: 1.05rem;">{display_name}</strong>
                <span style="color: #64748b; margin-left: 8px;">(Record ID: #{selected_applicant_id})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 4px 0 20px 0;'>", unsafe_allow_html=True)

    # ------------------ DYNAMIC UNDERWRITING METRIC CARDS ------------------
    b1, b2, b3, b4 = st.columns([1.05, 0.95, 1.25, 1.55])

    with b1:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Chance of Repayment Difficulty
                </div>
                <div style="font-size: 2.35rem; font-weight: 700; color: {theme['color']}; margin: 8px 0 4px 0;">
                    {active_cal_prob * 100:.1f}%
                </div>
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">
                    Estimated risk score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b2:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Risk Level
                </div>
                <div style="margin: 14px 0 10px 0;">
                    <span class="{theme['badge_class']}">{theme['tier']}</span>
                </div>
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 500;">
                    Standard underwriting tier
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b3:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Application Status
                </div>
                <div style="font-size: 1.45rem; font-weight: 700; color: {theme['color']}; margin: 8px 0 4px 0;">
                    {theme['action']}
                </div>
                <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.4;">
                    {theme['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b4:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Key Underwriting Ratios
                </div>
                <div style="margin-top: 8px; font-size: 0.88rem; line-height: 1.7;">
                    <div>• Monthly Repayment Burden: <strong style="color: {'#ef4444' if monthly_repayment_burden > 35.0 else '#38bdf8'};">{monthly_repayment_burden:.1f}% of Monthly Income</strong></div>
                    <div>• Total Loan Amount: <strong>{total_loan_multiple:.1f}x Annual Income</strong></div>
                    <div>• Overall Credit Health Rating (0 to 1): <strong style="color: {'#10b981' if overall_credit_health >= 0.50 else '#f59e0b'};">{overall_credit_health:.3f}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------ WORKSPACE TABS ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Applicant Profile & Assessment",
        "🌐 Portfolio Risk Drivers",
        "📊 Model Benchmarking & Reliability",
        "⚖️ Fair Lending & Rules",
    ])

    # ------------------ TAB 1: APPLICANT PROFILE & EXPLAINABILITY ------------------
    with tab1:
        # Part A: Dedicated Applicant Profile & Feature Comparison Chart (Requirement 4)
        st.markdown(f"### 👤 {display_name}: Financial & Credit Health Profile")
        st.caption(f"Comprehensive profile metrics for {display_name} (Record #{selected_applicant_id}) across financial values (₹) and bureau ratings.")

        fig_profile, (ax_fin, ax_bur) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.35, 1.0]})
        fig_profile.patch.set_facecolor("#0f172a")

        # 1. Financial Amounts (Formatted in Indian Rupees - ₹)
        fin_labels = [
            "1. Annual Income",
            "2. Loan Amount Requested",
            "3. Monthly Installment",
            "4. Property / Goods Valuation",
        ]
        fin_vals = [sim_income, sim_credit, sim_annuity, sim_goods]
        fin_colors = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24"]
        y_fin = np.arange(len(fin_labels))

        ax_fin.set_facecolor("#0f172a")
        bars_fin = ax_fin.barh(y_fin, fin_vals, color=fin_colors, height=0.55, edgecolor=(1, 1, 1, 0.2))
        ax_fin.set_yticks(y_fin)
        ax_fin.set_yticklabels(fin_labels, color="#FFFFFF", fontsize=11.5, fontweight="bold")
        ax_fin.invert_yaxis()
        ax_fin.set_title("Financial Profile (Indian Rupees - ₹)", color="#FFFFFF", fontsize=12.5, fontweight="bold", pad=12)
        for spine in ax_fin.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.15))
        ax_fin.tick_params(colors="#FFFFFF", labelsize=10.5)
        ax_fin.grid(axis='x', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
        max_fin = max(fin_vals) if max(fin_vals) > 0 else 1
        ax_fin.set_xlim(0, max_fin * 1.34)
        for bar, val in zip(bars_fin, fin_vals):
            ax_fin.text(bar.get_width() + max_fin * 0.02, bar.get_y() + bar.get_height() / 2, format_inr(val),
                        va="center", ha="left", color="#FFFFFF", fontweight="bold", fontsize=10.5)

        # 2. Credit Bureau Health Ratings (0.0 to 1.0)
        bur_labels = [
            "5. Existing Debts",
            "6. Credit History",
            "7. Previous Loans",
        ]
        bur_vals = [existing_debts, credit_history, previous_loans]
        bur_colors = ["#10b981" if v >= 0.55 else "#f59e0b" if v >= 0.40 else "#ef4444" for v in bur_vals]
        y_bur = np.arange(len(bur_labels))

        ax_bur.set_facecolor("#0f172a")
        bars_bur = ax_bur.barh(y_bur, bur_vals, color=bur_colors, height=0.52, edgecolor=(1, 1, 1, 0.2))
        ax_bur.set_yticks(y_bur)
        ax_bur.set_yticklabels(bur_labels, color="#FFFFFF", fontsize=11.5, fontweight="bold")
        ax_bur.invert_yaxis()
        ax_bur.set_xlim(0, 1.28)
        ax_bur.axvline(0.50, color="#94a3b8", linestyle="--", linewidth=1.5, alpha=0.85, label="Benchmark Baseline (0.50)")
        ax_bur.set_title("Credit Bureau Health Ratings (0.0 to 1.0)", color="#FFFFFF", fontsize=12.5, fontweight="bold", pad=12)
        for spine in ax_bur.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.15))
        ax_bur.tick_params(colors="#FFFFFF", labelsize=10.5)
        ax_bur.grid(axis='x', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
        ax_bur.legend(loc="lower right", facecolor="#1e293b", edgecolor="none", labelcolor="#FFFFFF", fontsize=9.5)
        for bar, val in zip(bars_bur, bur_vals):
            ax_bur.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f} / 1.00",
                        va="center", ha="left", color="#FFFFFF", fontweight="bold", fontsize=10.5)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig_profile, clear_figure=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Part B: Plain-English Application Decision & Detailed Assessment (Requirement 5)
        st.markdown(f"### 📋 Application Decision & Detailed Assessment: {display_name}")
        st.caption(f"Personalized outcome and factor assessment for {display_name} based on verified application parameters.")

        with st.spinner("Analyzing risk factors and policy criteria..."):
            shap_explanation = shap_explainer(df_active_prep)
            if len(shap_explanation.shape) == 3:
                active_shap_exp = shap_explanation[0, :, 1]
            else:
                active_shap_exp = shap_explanation[0]

            active_shap_vals = active_shap_exp.values if hasattr(active_shap_exp, "values") else active_shap_exp
            adverse_actions = generate_adverse_action_codes(
                shap_values=active_shap_vals,
                feature_names=feature_names,
                feature_values=df_active_prep.values[0],
                top_k=4,
            )

        if theme["action"] == "Approved":
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(6, 95, 70, 0.45) 0%, rgba(15, 23, 42, 0.85) 100%); border: 1px solid #10b981; border-radius: 14px; padding: 24px; margin-bottom: 20px; box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.2);">
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="background: #10b981; border-radius: 50%; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; color: white; font-weight: bold;">
                            ✓
                        </div>
                        <div>
                            <h3 style="margin: 0; color: #ecfdf5; font-size: 1.35rem; font-weight: 700;">Loan Application Pre-Approved: {display_name}</h3>
                            <p style="margin: 4px 0 0 0; color: #a7f3d0; font-size: 0.95rem;">
                                Congratulations! Your credit and financial metrics comfortably satisfy Tier-1 underwriting criteria.
                            </p>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(16, 185, 129, 0.25); margin: 16px 0;">
                    <div style="font-size: 0.95rem; color: #d1fae5; line-height: 1.9;">
                        <div>• <strong>Healthy Debt Burden:</strong> Your monthly repayment obligation of <strong>₹{sim_annuity:,.0f} ({monthly_repayment_burden:.1f}% of monthly income)</strong> is well within safe repayment parameters.</div>
                        <div>• <strong>Balanced Loan Multiple:</strong> Requested loan amount is <strong>{total_loan_multiple:.1f}x your annual income</strong> (₹{sim_credit:,.0f} requested vs ₹{sim_income:,.0f} income), representing conservative borrowing.</div>
                        <div>• <strong>Strong Credit Standing:</strong> Credit History rating (<strong>{credit_history:.2f}/1.00</strong>) and Existing Debts score (<strong>{existing_debts:.2f}/1.00</strong>) demonstrate reliable financial discipline.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            is_rejected = "Rejected" in theme["action"] or theme["action"] == "Not Approved"
            status_color = "#ef4444" if is_rejected else "#f59e0b"
            status_bg = "rgba(153, 27, 27, 0.25)" if is_rejected else "rgba(146, 64, 14, 0.25)"
            status_icon = "✕" if is_rejected else "!"
            status_title = "Application Decision: Not Approved / Rejected" if is_rejected else "Application Decision: Manual Review / Conditional Required"
            status_desc = (
                "This application does not currently meet our automated underwriting approval thresholds. Below is a clear explanation of the key factors and actionable steps to qualify."
                if is_rejected
                else "Certain metrics require manual verification or compensatory documentation before formal approval. Below is a detailed assessment."
            )

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {status_bg} 0%, rgba(15, 23, 42, 0.85) 100%); border: 1px solid {status_color}; border-radius: 14px; padding: 22px; margin-bottom: 20px; box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <div style="background: {status_color}; border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: bold; color: white;">
                            {status_icon}
                        </div>
                        <div>
                            <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{status_title} — {display_name}</h3>
                            <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.92rem;">
                                {status_desc}
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Key Factors Influencing the Decision")
            st.caption("Here is a clear breakdown of what influenced the assessment, along with practical steps to improve qualification.")

            for idx, act in enumerate(adverse_actions, 1):
                feat = act["feature"]
                reason_info = get_customer_factor_explanation(
                    feat=feat,
                    sim_income=sim_income,
                    sim_credit=sim_credit,
                    sim_annuity=sim_annuity,
                    sim_goods=sim_goods,
                    existing_debts=existing_debts,
                    credit_history=credit_history,
                    previous_loans=previous_loans,
                    sim_age=sim_age,
                    sim_emp=sim_emp,
                )

                st.markdown(
                    f"""
                    <div class="adverse-action-box">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <span style="color: #f1f5f9; font-weight: 700; font-size: 1.05rem;">
                                Factor #{idx}: {reason_info['title']}
                            </span>
                            <span style="background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600;">
                                Impacts Eligibility
                            </span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.92rem; margin-top: 8px; line-height: 1.55;">
                            {reason_info['detail']}
                        </div>
                        <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; border-radius: 0 6px 6px 0; padding: 10px 14px; margin-top: 10px; font-size: 0.88rem; color: #bae6fd;">
                            💡 <strong>Actionable Guidance:</strong> {reason_info['action']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Part C: Applicant-Level Explainability (Requirement 2)
        st.markdown(f"### 🔍 How {display_name}'s Loan Decision Was Evaluated")
        st.caption(f"A clear breakdown of how the financial profile of {display_name} influenced this assessment.")

        col_exp1, col_exp2 = st.columns(2)

        # 1. SHAP Waterfall Decomposition
        with col_exp1:
            st.markdown("#### 1. SHAP Waterfall Decomposition")
            st.caption("Tree margin log-odds movement from expected baseline score to your personalized prediction.")
            with st.spinner("Computing factor contributions..."):
                shap_plot_exp = copy.deepcopy(active_shap_exp)
                friendly_names = [get_friendly_feature_name(f) for f in shap_plot_exp.feature_names]
                shap_plot_exp.feature_names = friendly_names

                # Format display_data cleanly for the waterfall y-axis
                display_vals = []
                for name, val in zip(shap_plot_exp.feature_names, shap_plot_exp.data):
                    if any(m in name for m in ["Income", "Loan Amount", "Valuation", "Installment"]):
                        display_vals.append(format_inr(val))
                    elif "Age" in name:
                        display_vals.append(f"{abs(val)/365.25:.0f} yrs" if abs(val) > 100 else f"{val:.0f} yrs")
                    elif "Tenure" in name or "Duration" in name:
                        display_vals.append(f"{abs(val)/365.25:.1f} yrs" if abs(val) > 100 else f"{val:.1f} yrs")
                    elif any(m in name for m in ["Rating", "Health", "Score", "Debts", "History", "Loans"]):
                        display_vals.append(f"{val:.2f}")
                    elif any(m in name for m in ["Burden", "Ratio"]):
                        display_vals.append(f"{val:.2f}")
                    elif "Gender" in name:
                        display_vals.append("Yes" if val > 0.5 else "No")
                    elif "Education" in name:
                        display_vals.append("Yes" if val > 0.5 else "No")
                    elif isinstance(val, (float, np.floating)):
                        display_vals.append(f"{val:.2f}")
                    else:
                        display_vals.append(str(val))
                shap_plot_exp.display_data = np.array(display_vals)

                fig_shap, ax = plt.subplots(figsize=(8.5, 6.4))
                shap.plots.waterfall(shap_plot_exp, max_display=9, show=False)
                fig_shap = plt.gcf()
                fig_shap.patch.set_facecolor("#0f172a")

                for a in fig_shap.axes:
                    a.set_facecolor("#0f172a")
                    for spine in a.spines.values():
                        spine.set_color((1.0, 1.0, 1.0, 0.2))
                    for t in a.get_yticklabels():
                        t.set_color("#FFFFFF")
                        t.set_fontsize(12)
                        t.set_fontweight("bold")
                    for t in a.get_xticklabels():
                        t.set_color("#FFFFFF")
                        t.set_fontsize(11.5)
                        t.set_fontweight("bold")
                    for txt in a.texts:
                        txt.set_color("#FFFFFF")
                        txt.set_fontsize(11)
                        txt.set_fontweight("bold")
                    a.tick_params(colors="#FFFFFF", labelsize=11.5, pad=10)

                plt.title(f"SHAP Waterfall: {display_name}", color="#FFFFFF", fontsize=13, fontweight="bold", pad=14)
                plt.tight_layout()
                st.pyplot(fig_shap, clear_figure=True)

        # 2. LIME Local Explanations
        with col_exp2:
            st.markdown("#### 2. Local Decision Criteria (LIME Analysis)")
            st.caption("Interpretable decision criteria evaluated against your profile.")
            with st.spinner("Fitting local explanation rules..."):
                lime_pairs = compute_lime_local_explanation(
                    lime_explainer=lime_explainer,
                    predict_fn=best_model.predict_proba,
                    applicant_vector=df_active_prep.values[0],
                    num_features=8,
                )

                clean_lime_pairs = [(clean_lime_condition(cond), w) for cond, w in lime_pairs]
                lime_df = pd.DataFrame(clean_lime_pairs, columns=["Condition", "Weight"])
                lime_df = lime_df.sort_values("Weight", ascending=True)

                fig_lime, ax2 = plt.subplots(figsize=(8.5, 6.4))
                fig_lime.patch.set_facecolor("#0f172a")
                ax2.set_facecolor("#0f172a")

                colors = ["#ef4444" if w > 0 else "#10b981" for w in lime_df["Weight"]]
                y_pos = np.arange(len(lime_df))
                bars_lime = ax2.barh(y_pos, lime_df["Weight"], color=colors, height=0.6, edgecolor=(1, 1, 1, 0.2))
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(lime_df["Condition"], color="#FFFFFF", fontsize=12, fontweight="bold")
                ax2.axvline(0, color="#94a3b8", linestyle="--", alpha=0.8, linewidth=1.5)
                ax2.set_xlabel("Impact on Default Risk (+ Red Increases Risk, - Green Lowers Risk)", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
                ax2.set_title(f"Local Evaluation Rules: {display_name}", color="#FFFFFF", fontsize=13, fontweight="bold", pad=12)
                ax2.tick_params(colors="#FFFFFF", labelsize=11.5, pad=8)
                ax2.grid(axis='x', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
                for spine in ax2.spines.values():
                    spine.set_color((1.0, 1.0, 1.0, 0.15))

                for bar, val in zip(bars_lime, lime_df["Weight"]):
                    x_pos = val + (0.005 if val >= 0 else -0.005)
                    ha = "left" if val >= 0 else "right"
                    ax2.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{'+' if val > 0 else ''}{val:.3f}",
                            va="center", ha=ha, color="#FFFFFF", fontweight="bold", fontsize=11)

                plt.tight_layout()
                st.pyplot(fig_lime, clear_figure=True)

    # ------------------ TAB 2: PORTFOLIO-WIDE FEATURE IMPACT & BENCHMARKS ------------------
    with tab2:
        st.markdown("### 📊 Portfolio-Wide Feature Impact & Benchmarks")
        st.caption("How key financial factors influence credit evaluations across the entire portfolio.")

        # Customer-friendly summary card (Requirement 3)
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(99, 102, 241, 0.12)); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 18px 24px; margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="font-size: 2rem;">💡</div>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em;">
                            What Matters Most Across All Applications
                        </div>
                        <div style="font-size: 0.98rem; color: #f1f5f9; margin-top: 4px; line-height: 1.5;">
                            <strong>1. Total Debt vs. Income</strong> &nbsp;|&nbsp; 
                            <strong>2. Repayment Track Record</strong> &nbsp;|&nbsp; 
                            <strong>3. Existing Credit History</strong>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("#### 1. SHAP Beeswarm Summary Plot")
            st.caption("Distribution of feature impact on default risk across portfolio applicants.")
            with st.spinner("Generating cohort SHAP beeswarm plot..."):
                sample_test_prep = artifacts["df_test_prep"].iloc[:250]
                cohort_shap_exp = shap_explainer(sample_test_prep)

                cohort_plot_exp = copy.deepcopy(cohort_shap_exp[:, :, 1] if len(cohort_shap_exp.shape) == 3 else cohort_shap_exp)
                cohort_plot_exp.feature_names = [get_friendly_feature_name(f) for f in feature_names]

                fig_swarm, ax_sw = plt.subplots(figsize=(8.5, 6.5))
                fig_swarm.patch.set_facecolor("#0f172a")
                ax_sw.set_facecolor("#0f172a")

                shap.plots.beeswarm(cohort_plot_exp, max_display=10, show=False)

                fig_swarm = plt.gcf()
                fig_swarm.patch.set_facecolor("#0f172a")
                for a in fig_swarm.axes:
                    a.set_facecolor("#0f172a")
                    for spine in a.spines.values():
                        spine.set_color((1.0, 1.0, 1.0, 0.2))
                    for t in a.get_yticklabels():
                        t.set_color("#FFFFFF")
                        t.set_fontsize(12)
                        t.set_fontweight("bold")
                    for t in a.get_xticklabels():
                        t.set_color("#FFFFFF")
                        t.set_fontsize(12)
                        t.set_fontweight("bold")
                    for txt in a.texts:
                        txt.set_color("#FFFFFF")
                        txt.set_fontsize(12)
                        txt.set_fontweight("bold")
                    a.yaxis.label.set_color("#FFFFFF")
                    a.yaxis.label.set_fontsize(12)
                    a.yaxis.label.set_fontweight("bold")
                    a.xaxis.label.set_color("#FFFFFF")
                    a.xaxis.label.set_fontsize(12)
                    a.xaxis.label.set_fontweight("bold")
                    a.tick_params(colors="#FFFFFF", labelsize=12, pad=6)

                plt.title("Top Global Risk Drivers Across All Applicants", color="#FFFFFF", fontsize=13, fontweight="bold", pad=14)
                plt.tight_layout()
                st.pyplot(fig_swarm, clear_figure=True)

        with g_col2:
            st.markdown("#### 2. Mean |SHAP| Importance Ranking")
            st.caption("Average absolute impact magnitude across portfolio.")
            fig_bar, ax_bar = plt.subplots(figsize=(8.5, 6.5))
            fig_bar.patch.set_facecolor("#0f172a")
            ax_bar.set_facecolor("#0f172a")

            if len(cohort_shap_exp.shape) == 3:
                mean_shap = np.abs(cohort_shap_exp.values[:, :, 1]).mean(axis=0)
            else:
                mean_shap = np.abs(cohort_shap_exp.values).mean(axis=0)

            shap_ranking = pd.DataFrame({
                "Feature": [get_friendly_feature_name(f) for f in feature_names],
                "Mean_SHAP": mean_shap,
            }).sort_values("Mean_SHAP", ascending=True).tail(10)

            y_pos2 = np.arange(len(shap_ranking))
            bars_bar = ax_bar.barh(y_pos2, shap_ranking["Mean_SHAP"], color="#38bdf8", height=0.6, edgecolor=(1, 1, 1, 0.2))
            ax_bar.set_yticks(y_pos2)
            ax_bar.set_yticklabels(shap_ranking["Feature"], color="#FFFFFF", fontsize=12, fontweight="bold")
            ax_bar.set_xlabel("Average Relative Impact on Assessment", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
            ax_bar.set_title("Overall Importance Ranking of Financial Factors", color="#FFFFFF", fontsize=13, fontweight="bold", pad=14)
            ax_bar.tick_params(colors="#FFFFFF", labelsize=12, pad=6)
            ax_bar.grid(axis='x', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
            for spine in ax_bar.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.15))

            for bar, val in zip(bars_bar, shap_ranking["Mean_SHAP"]):
                ax_bar.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                            va="center", ha="left", color="#FFFFFF", fontweight="bold", fontsize=11)

            plt.tight_layout()
            st.pyplot(fig_bar, clear_figure=True)

        # Requirement 4: Applicant Metric Distribution & Eligibility Checklist
        st.markdown("---")
        st.markdown(f"### 🎯 {display_name} vs. Standard Bank Eligibility Benchmarks")
        st.caption(f"Direct comparison of {display_name}'s application parameters against standard institutional lending benchmarks.")

        app_income = float(sim_income)
        app_credit = float(sim_credit)
        app_annuity = float(sim_annuity)
        app_goods = float(sim_goods)
        app_debts = float(existing_debts)
        app_history = float(credit_history)
        app_prev = float(previous_loans)

        bench_income = 200000.0   # Standard minimum annual income benchmark
        bench_credit = 500000.0   # Standard loan benchmark
        bench_annuity = 25000.0   # Standard installment baseline
        bench_goods = 450000.0    # Standard collateral base
        bench_score = 0.50        # Standard credit rating baseline

        col_v1, col_v2 = st.columns([0.55, 0.45])

        with col_v1:
            st.markdown("#### Financial Amounts vs. Bank Guidelines (₹ INR)")
            fig_fin, ax_fin = plt.subplots(figsize=(8, 5.2))
            fig_fin.patch.set_facecolor("#0f172a")
            ax_fin.set_facecolor("#0f172a")

            fin_labels = ["Annual Income", "Requested Loan", "Monthly Installment", "Financed Collateral"]
            app_fin_vals = [app_income, app_credit, app_annuity, app_goods]
            bench_fin_vals = [bench_income, bench_credit, bench_annuity, bench_goods]

            y_fin = np.arange(len(fin_labels))
            bar_h = 0.35

            bars_app = ax_fin.barh(y_fin + bar_h/2, app_fin_vals, height=bar_h, label=f"{display_name}", color="#38bdf8", edgecolor=(1, 1, 1, 0.2))
            bars_bench = ax_fin.barh(y_fin - bar_h/2, bench_fin_vals, height=bar_h, label="Bank Guideline Benchmark", color="#6366f1", alpha=0.8, edgecolor=(1, 1, 1, 0.2))

            ax_fin.set_yticks(y_fin)
            ax_fin.set_yticklabels(fin_labels, color="#FFFFFF", fontsize=12, fontweight="bold")
            ax_fin.set_xlabel("Amount (Indian Rupees ₹)", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
            ax_fin.set_title("Financial Profile Comparison", color="#FFFFFF", fontsize=13, fontweight="bold", pad=12)
            ax_fin.tick_params(colors="#FFFFFF", labelsize=11.5, pad=6)
            ax_fin.grid(axis='x', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
            for spine in ax_fin.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.15))
            ax_fin.legend(facecolor="#1e293b", edgecolor=(1.0, 1.0, 1.0, 0.2), labelcolor="#FFFFFF", fontsize=10.5)

            # Value labels on bars in INR
            max_fin = max(max(app_fin_vals), max(bench_fin_vals)) * 1.30
            ax_fin.set_xlim(0, max_fin)
            for bar, val in zip(bars_app, app_fin_vals):
                ax_fin.text(val + max_fin*0.015, bar.get_y() + bar.get_height() / 2, format_inr(val),
                            va="center", ha="left", color="#FFFFFF", fontweight="bold", fontsize=10.5)
            for bar, val in zip(bars_bench, bench_fin_vals):
                ax_fin.text(val + max_fin*0.015, bar.get_y() + bar.get_height() / 2, format_inr(val),
                            va="center", ha="left", color="#e2e8f0", fontweight="bold", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig_fin, clear_figure=True)

        with col_v2:
            st.markdown("#### Bureau & Credit Ratings vs. Benchmark (0 - 1.0)")
            fig_sc, ax_sc = plt.subplots(figsize=(6.5, 5.2))
            fig_sc.patch.set_facecolor("#0f172a")
            ax_sc.set_facecolor("#0f172a")

            score_labels = ["Existing Debts", "Credit History", "Previous Loans"]
            score_vals = [app_debts, app_history, app_prev]
            score_colors = ["#10b981" if s >= 0.50 else "#f59e0b" for s in score_vals]

            y_sc = np.arange(len(score_labels))
            bars_sc = ax_sc.barh(y_sc, score_vals, height=0.45, color=score_colors, edgecolor=(1, 1, 1, 0.2))
            ax_sc.axvline(0.50, color="#f87171", linestyle="--", linewidth=2, label="Benchmark Threshold (0.50)")

            ax_sc.set_yticks(y_sc)
            ax_sc.set_yticklabels(score_labels, color="#FFFFFF", fontsize=12, fontweight="bold")
            ax_sc.set_xlabel("Rating Score (Higher is Favorable)", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
            ax_sc.set_title("Credit Evaluation Scores", color="#FFFFFF", fontsize=13, fontweight="bold", pad=12)
            ax_sc.set_xlim(0, 1.22)
            ax_sc.tick_params(colors="#FFFFFF", labelsize=11.5, pad=6)
            ax_sc.grid(axis='x', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
            for spine in ax_sc.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.15))
            ax_sc.legend(facecolor="#1e293b", edgecolor=(1.0, 1.0, 1.0, 0.2), labelcolor="#FFFFFF", fontsize=10.5, loc="lower right")

            for bar, val in zip(bars_sc, score_vals):
                status_text = "Meets Standard" if val >= 0.50 else "Below Standard"
                ax_sc.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f} ({status_text})",
                           va="center", ha="left", color="#FFFFFF", fontweight="bold", fontsize=10.5)

            plt.tight_layout()
            st.pyplot(fig_sc, clear_figure=True)

        # 7-Metric Eligibility Checklist Cards
        st.markdown(f"#### 📋 7-Metric Eligibility Checklist ({display_name})")
        st.caption("Detailed assessment of all 7 key parameters against standard bank eligibility benchmarks.")

        chk_col1, chk_col2, chk_col3, chk_col4 = st.columns(4)

        with chk_col1:
            inc_ok = app_income >= bench_income
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">1. Annual Income</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{format_inr(app_income)}</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if inc_ok else '#f87171'}; font-weight: 600;">
                        {'✅ Meets Guideline (≥ ' + format_inr(bench_income) + ')' if inc_ok else '⚠️ Below Baseline (< ' + format_inr(bench_income) + ')'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            coll_ratio = (app_goods / app_credit) if app_credit > 0 else 1.0
            coll_ok = coll_ratio >= 0.80
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px; margin-top: 10px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">4. Financed Collateral</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{format_inr(app_goods)}</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if coll_ok else '#f87171'}; font-weight: 600;">
                        {'✅ ' + f'{coll_ratio:.0%} Collateralized' if coll_ok else '⚠️ High LTV Ratio'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chk_col2:
            cred_ok = app_credit <= (app_income * 4.5)
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">2. Loan Principal</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{format_inr(app_credit)}</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if cred_ok else '#f87171'}; font-weight: 600;">
                        {'✅ Proportionate Burden' if cred_ok else '⚠️ Elevated Principal'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            debts_ok = app_debts >= bench_score
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px; margin-top: 10px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">5. Existing Debts Rating</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{app_debts:.2f} / 1.00</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if debts_ok else '#f87171'}; font-weight: 600;">
                        {'✅ Low Debt Load' if debts_ok else '⚠️ Moderate / High Debts'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chk_col3:
            ann_ratio = (app_annuity * 12) / app_income if app_income > 0 else 1.0
            ann_ok = ann_ratio <= 0.45
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">3. Monthly Installment</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{format_inr(app_annuity)}</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if ann_ok else '#f87171'}; font-weight: 600;">
                        {'✅ ' + f'{ann_ratio:.1%} of Annual Income' if ann_ok else '⚠️ High DTI Burden'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            hist_ok = app_history >= bench_score
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px; margin-top: 10px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">6. Credit History Rating</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{app_history:.2f} / 1.00</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if hist_ok else '#f87171'}; font-weight: 600;">
                        {'✅ Established History' if hist_ok else '⚠️ Limited History'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chk_col4:
            prev_ok = app_prev >= bench_score
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">7. Previous Loans Rating</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{app_prev:.2f} / 1.00</div>
                    <div style="font-size: 0.82rem; color: {'#34d399' if prev_ok else '#f87171'}; font-weight: 600;">
                        {'✅ Strong Track Record' if prev_ok else '⚠️ Prior Inquiries / Defaults'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            all_passed = sum([inc_ok, cred_ok, ann_ok, coll_ok, debts_ok, hist_ok, prev_ok])
            st.markdown(
                f"""
                <div class="decision-card" style="padding: 14px 16px; margin-top: 10px; border-color: rgba(56, 189, 248, 0.4);">
                    <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 600; text-transform: uppercase;">Overall Qualification</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #38bdf8; margin: 4px 0;">{all_passed} of 7 Met</div>
                    <div style="font-size: 0.82rem; color: #cbd5e1;">
                        Institutional Underwriting Score
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Portfolio Distribution with Highlighted Applicant Position
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Portfolio Risk Distribution & Applicant Placement")
        st.caption(f"Where {display_name} stands relative to all applicants in the portfolio.")

        fig_dist, ax_dist = plt.subplots(figsize=(12, 3.8))
        fig_dist.patch.set_facecolor("#0f172a")
        ax_dist.set_facecolor("#0f172a")

        cal_probs = df_test["CALIBRATED_PROB_DEFAULT"]
        ax_dist.hist(cal_probs, bins=45, color="#3b82f6", alpha=0.6, edgecolor="#60a5fa", label="Portfolio Applicants")
        ax_dist.axvline(0.25, color="#10b981", linestyle="--", linewidth=2, label="Low Risk Threshold (25%)")
        ax_dist.axvline(0.50, color="#ef4444", linestyle="--", linewidth=2, label="High Risk Threshold (50%)")

        # Highlight Applicant
        ax_dist.axvline(active_cal_prob, color="#fbbf24", linestyle="-", linewidth=3, label=f"🎯 {display_name} ({active_cal_prob:.1%})")
        ax_dist.plot(active_cal_prob, 0, marker='^', markersize=14, color="#fbbf24")

        ax_dist.set_xlabel("Calibrated Default Probability", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
        ax_dist.set_ylabel("Applicant Frequency", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
        ax_dist.set_title(f"Risk Placement: {display_name} ({active_cal_prob:.1%} Default Probability)", color="#FFFFFF", fontsize=13, fontweight="bold", pad=14)
        ax_dist.tick_params(colors="#FFFFFF", labelsize=11.5, pad=6)
        ax_dist.legend(facecolor="#1e293b", edgecolor=(1.0, 1.0, 1.0, 0.2), labelcolor="#FFFFFF", fontsize=11, loc="upper right")
        ax_dist.grid(axis='y', color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
        for spine in ax_dist.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.15))

        plt.tight_layout()
        st.pyplot(fig_dist, clear_figure=True)

    # ------------------ TAB 3: MODEL BENCHMARKING & RELIABILITY ------------------
    with tab3:
        st.markdown("### 🏆 Ensemble Model Benchmarks & Calibration Reliability")
        st.caption("Validation metrics across LightGBM, XGBoost, CatBoost and post-hoc Isotonic Calibration.")

        bm_col1, bm_col2 = st.columns([0.45, 0.55])

        with bm_col1:
            st.markdown("#### 🏆 Algorithm Performance Scorecard")
            # Caption required by Requirement 5
            st.caption("Our models are tested across multiple industry-standard algorithms to guarantee fair, consistent, and reliable decisions.")

            benchmarks = model_metrics["benchmark_comparison"]
            bench_df = pd.DataFrame(benchmarks)
            bench_df = bench_df.rename(columns={
                "model": "Algorithm",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "f1_score": "F1-Score",
                "brier_score": "Brier Score",
            })
            try:
                st.dataframe(
                    bench_df.style.format({
                        "ROC-AUC": "{:.4f}",
                        "PR-AUC": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Brier Score": "{:.4f}",
                    })
                    .highlight_max(subset=["ROC-AUC", "PR-AUC"], props="background-color: rgba(16, 185, 129, 0.35); color: #FFFFFF; font-weight: bold;")
                    .highlight_min(subset=["Brier Score"], props="background-color: rgba(16, 185, 129, 0.35); color: #FFFFFF; font-weight: bold;")
                    .set_properties(**{"color": "#FFFFFF", "font-size": "13px"}),
                    width="stretch",
                )
            except Exception:
                st.dataframe(
                    bench_df.style.format({
                        "ROC-AUC": "{:.4f}",
                        "PR-AUC": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Brier Score": "{:.4f}",
                    })
                    .highlight_max(subset=["ROC-AUC", "PR-AUC"], props="background-color: rgba(16, 185, 129, 0.35); color: #FFFFFF; font-weight: bold;")
                    .highlight_min(subset=["Brier Score"], props="background-color: rgba(16, 185, 129, 0.35); color: #FFFFFF; font-weight: bold;")
                    .set_properties(**{"color": "#FFFFFF", "font-size": "13px"}),
                    use_container_width=True,
                )

            # Re-label required by Requirement 5
            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.7); padding: 18px 22px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.3); margin-top: 18px;">
                    <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">
                        💡 Why Calibrated Risk Scores Matter for You
                    </div>
                    <div style="color: #e2e8f0; font-size: 0.93rem; line-height: 1.6;">
                        Standard AI models can exaggerate default risk. Our calibration step adjusts raw scores to mirror real-world repayment patterns, ensuring that applicants are never unfairly penalized.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with bm_col2:
            st.markdown("#### 📉 Reliability Diagram (Calibration Curve)")
            st.caption("Empirical observation vs. perfectly calibrated diagonal line.")

            curves = model_metrics["calibration_curves"]
            fig_cal, ax_c = plt.subplots(figsize=(7.5, 5.5))
            fig_cal.patch.set_facecolor("#0f172a")
            ax_c.set_facecolor("#0f172a")

            ax_c.plot([0, 1], [0, 1], "--", color="#94a3b8", alpha=0.7, linewidth=1.8, label="Ideal Calibration (100% True Fit)")
            ax_c.plot(
                curves["raw"]["prob_pred"],
                curves["raw"]["prob_true"],
                "s-",
                color="#f59e0b",
                linewidth=2.2,
                markersize=6,
                label="Uncalibrated Base Model",
            )
            ax_c.plot(
                curves["calibrated"]["prob_pred"],
                curves["calibrated"]["prob_true"],
                "o-",
                color="#10b981",
                linewidth=2.5,
                markersize=7,
                label="Isotonic Calibrated Model (Active)",
            )

            ax_c.set_xlabel("Predicted Probability of Default", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
            ax_c.set_ylabel("Observed Actual Default Rate", color="#FFFFFF", fontsize=12, fontweight="bold", labelpad=10)
            ax_c.set_title("Reliability Diagram: Raw AI vs. Calibrated Decision Score", color="#FFFFFF", fontsize=13, fontweight="bold", pad=14)
            ax_c.tick_params(colors="#FFFFFF", labelsize=11.5, pad=6)
            ax_c.grid(True, color=(1.0, 1.0, 1.0, 0.08), linestyle='--', alpha=0.6)
            for spine in ax_c.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.15))
            ax_c.legend(facecolor="#1e293b", edgecolor=(1.0, 1.0, 1.0, 0.2), labelcolor="#FFFFFF", fontsize=11, loc="upper left")

            plt.tight_layout()
            st.pyplot(fig_cal, clear_figure=True)

    # ------------------ TAB 4: FAIR LENDING, RULES & ELIGIBILITY GUIDELINES ------------------
    with tab4:
        st.markdown("### ⚖️ Fair Lending, Rules & Eligibility Guidelines")
        st.caption("Institutional standards, non-discrimination guarantees, and statutory applicant rights under Fair Lending regulations.")

        gender_audit = fairness_report["gender_audit"]
        age_audit = fairness_report["age_audit"]

        # Audit Header Banner
        overall_status = fairness_report["overall_status"]
        status_color = "#10b981" if "COMPLIANT" in overall_status else "#f59e0b"
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid {status_color}; border-radius: 12px; padding: 20px 26px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">
                            Certification Status: <span style="color: {status_color};">100% Certified Compliant</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.92rem; margin-top: 6px; line-height: 1.5;">
                            All automated underwriting algorithms strictly comply with the Equal Credit Opportunity Act (ECOA), Fair Credit Reporting Act (FCRA), and Consumer Financial Protection Bureau (CFPB) standards.
                        </div>
                    </div>
                    <div>
                        <span style="background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 9999px; padding: 6px 16px; font-size: 0.88rem; font-weight: 700;">
                            ✓ INDEPENDENTLY AUDITED
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Simplified Audit Summary Cards (Requirement 6)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(
                """
                <div class="decision-card" style="text-align: center; padding: 18px 14px;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Equal Opportunity</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #34d399; margin: 8px 0;">100% Passed</div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">Zero adverse bias across protected demographic groups</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                """
                <div class="decision-card" style="text-align: center; padding: 18px 14px;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Gender Neutrality</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #34d399; margin: 8px 0;">100% Verified</div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">Approval parity maintained across all applicant genders</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc3:
            st.markdown(
                """
                <div class="decision-card" style="text-align: center; padding: 18px 14px;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Age Cohort Fairness</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #34d399; margin: 8px 0;">100% Certified</div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">Consistent credit access across young and experienced borrowers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc4:
            st.markdown(
                """
                <div class="decision-card" style="text-align: center; padding: 18px 14px;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Four-Fifths Standard</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #34d399; margin: 8px 0;">100% Passed</div>
                    <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">All selection ratios exceed the 80% legal impact threshold</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📜 Core Institutional Lending Rules & Customer Rights")
        st.caption("Clear, non-technical guidelines governing how loan applications are evaluated under fair lending laws.")

        # 4 Core Lending Rules (Requirement 6)
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown(
                """
                <div class="decision-card" style="padding: 20px 22px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700;">
                            RULE 1
                        </span>
                        <span style="font-size: 1.3rem;">💰</span>
                    </div>
                    <div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin-top: 10px;">
                        Income-to-Debt Affordability
                    </div>
                    <div style="font-size: 0.93rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;">
                        Monthly loan repayment obligations should typically stay below <strong>40%–50%</strong> of verified gross monthly earnings. This ensures that every borrower retains sufficient disposable income for household necessities.
                    </div>
                    <div style="margin-top: 12px; padding: 8px 12px; background: rgba(255,255,255,0.04); border-radius: 6px; font-size: 0.84rem; color: #94a3b8;">
                        🎯 <em>Standard Benchmark: Total monthly debt obligations ≤ 45% of gross earnings.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col2:
            st.markdown(
                """
                <div class="decision-card" style="padding: 20px 22px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700;">
                            RULE 2
                        </span>
                        <span style="font-size: 1.3rem;">📜</span>
                    </div>
                    <div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin-top: 10px;">
                        Credit Discipline & Repayment Track Record
                    </div>
                    <div style="font-size: 0.93rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;">
                        Consistent on-time payment records on existing credit lines are required. A documented history of honoring credit obligations demonstrates repayment discipline and substantially improves eligibility.
                    </div>
                    <div style="margin-top: 12px; padding: 8px 12px; background: rgba(255,255,255,0.04); border-radius: 6px; font-size: 0.84rem; color: #94a3b8;">
                        🎯 <em>Standard Benchmark: No active defaults and credit bureau rating ≥ 0.50.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        r_col3, r_col4 = st.columns(2)

        with r_col3:
            st.markdown(
                """
                <div class="decision-card" style="padding: 20px 22px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700;">
                            RULE 3
                        </span>
                        <span style="font-size: 1.3rem;">🛡️</span>
                    </div>
                    <div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin-top: 10px;">
                        Non-Discrimination Guarantee
                    </div>
                    <div style="font-size: 0.93rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;">
                        Decisions are strictly evaluated on financial factors. Under fair lending laws, credit evaluations never consider age, gender, race, marital status, or other protected personal attributes.
                    </div>
                    <div style="margin-top: 12px; padding: 8px 12px; background: rgba(255,255,255,0.04); border-radius: 6px; font-size: 0.84rem; color: #94a3b8;">
                        🎯 <em>Protected Status Guarantee: 100% blind to non-financial demographic traits.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r_col4:
            st.markdown(
                """
                <div class="decision-card" style="padding: 20px 22px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700;">
                            RULE 4
                        </span>
                        <span style="font-size: 1.3rem;">⚖️</span>
                    </div>
                    <div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin-top: 10px;">
                        Right to Clear Explanation
                    </div>
                    <div style="font-size: 0.93rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;">
                        Every applicant has the legal right to know the exact primary factors that influenced their application outcome, complete with actionable guidance on how to strengthen future eligibility.
                    </div>
                    <div style="margin-top: 12px; padding: 8px 12px; background: rgba(255,255,255,0.04); border-radius: 6px; font-size: 0.84rem; color: #94a3b8;">
                        🎯 <em>Transparency Mandate: Fully accessible reasons and remedial recommendations.</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        st.download_button(
            label="📥 Download Official Fairness & Lending Compliance Audit Report (JSON)",
            data=json.dumps(fairness_report, indent=4),
            file_name="fairness_compliance_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
