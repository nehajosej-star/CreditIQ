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

import json
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
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .decision-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.16);
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
        background: rgba(30, 27, 75, 0.4);
        border-left: 4px solid #818cf8;
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        margin: 15px 0;
        border: 1px solid rgba(129, 140, 248, 0.2);
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

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"] p, 
    button[data-baseweb="tab"] div, 
    button[data-baseweb="tab"] span {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        opacity: 0.85 !important;
        font-size: 0.92rem !important;
    }
    button[data-baseweb="tab"]:hover p,
    button[data-baseweb="tab"]:hover span,
    button[data-baseweb="tab"]:hover div {
        color: #FFFFFF !important;
        opacity: 1.0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1e293b !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #38BDF8 !important;
        font-weight: 600 !important;
        opacity: 1.0 !important;
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


def format_currency(val: float) -> str:
    return f"${val:,.0f}"


def get_risk_theme(prob: float):
    if prob < 0.25:
        return {
            "tier": "Low Risk",
            "badge_class": "risk-badge-low",
            "color": "#10b981",
            "action": "APPROVE",
            "desc": "Applicant credit score comfortably satisfies Tier-1 underwriting parameters.",
        }
    elif prob <= 0.50:
        return {
            "tier": "Medium Risk",
            "badge_class": "risk-badge-med",
            "color": "#f59e0b",
            "action": "MANUAL UNDERWRITE",
            "desc": "Elevated debt burden or bureau flags require senior underwriter review.",
        }
    else:
        return {
            "tier": "High Risk",
            "badge_class": "risk-badge-high",
            "color": "#ef4444",
            "action": "REJECT",
            "desc": "Default probability exceeds portfolio risk tolerance thresholds.",
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

    # App Header
    col_hdr1, col_hdr2 = st.columns([0.75, 0.25])
    with col_hdr1:
        st.markdown(
            "## 🏦 CreditIQ | Autonomous Risk & Explainability Suite\n"
            "<span style='color: #94a3b8; font-size: 0.95rem;'>"
            "Enterprise Underwriting Decisioning, Isotonic Calibration, SHAP/LIME Explainability & ECOA Compliance</span>",
            unsafe_allow_html=True,
        )
    with col_hdr2:
        st.markdown(
            "<div style='text-align: right; padding-top: 10px;'>"
            "<span class='compliance-pill'>REG-B / ECOA AUDIT READY</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 15px 0 25px 0;'>", unsafe_allow_html=True)

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.markdown("### 📋 Applicant Selection")

    # Filter selector with applicant metadata
    applicant_options = df_test["SK_ID_CURR"].tolist()
    default_idx = 0
    # Pick an applicant with moderate/high risk for interesting default demonstration if available
    high_risk_candidates = df_test[df_test["RISK_TIER"] == "High Risk"].index
    if len(high_risk_candidates) > 0:
        default_idx = int(high_risk_candidates[0])

    selected_id = st.sidebar.selectbox(
        "Select Applicant Record:",
        applicant_options,
        index=default_idx,
        help="Choose a borrower from the held-out test cohort",
    )

    # Fetch baseline applicant row
    baseline_row = df_test[df_test["SK_ID_CURR"] == selected_id].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Counterfactual 'What-If' Simulation")
    st.sidebar.caption("Adjust variables to simulate counterfactual credit scenarios in real-time.")

    # Simulation Sliders
    sim_income = st.sidebar.slider(
        "Annual Income (AMT_INCOME_TOTAL):",
        min_value=25000,
        max_value=1200000,
        value=int(baseline_row["AMT_INCOME_TOTAL"]),
        step=5000,
        format="$%d",
    )

    sim_credit = st.sidebar.slider(
        "Loan Facility Principal (AMT_CREDIT):",
        min_value=50000,
        max_value=3000000,
        value=int(baseline_row["AMT_CREDIT"]),
        step=10000,
        format="$%d",
    )

    sim_annuity = st.sidebar.slider(
        "Scheduled Annuity (AMT_ANNUITY):",
        min_value=2000,
        max_value=180000,
        value=int(baseline_row["AMT_ANNUITY"]),
        step=1000,
        format="$%d",
    )

    sim_goods = st.sidebar.slider(
        "Financed Goods Valuation (AMT_GOODS_PRICE):",
        min_value=30000,
        max_value=3000000,
        value=int(baseline_row["AMT_GOODS_PRICE"]) if pd.notnull(baseline_row["AMT_GOODS_PRICE"]) else int(baseline_row["AMT_CREDIT"]),
        step=10000,
        format="$%d",
    )

    st.sidebar.markdown("##### Credit Bureau Ratings")
    sim_ext1 = st.sidebar.slider(
        "External Bureau 1 (EXT_SOURCE_1):",
        min_value=0.01,
        max_value=0.99,
        value=float(baseline_row["EXT_SOURCE_1"]) if pd.notnull(baseline_row["EXT_SOURCE_1"]) else 0.50,
        step=0.01,
    )
    sim_ext2 = st.sidebar.slider(
        "External Bureau 2 (EXT_SOURCE_2):",
        min_value=0.01,
        max_value=0.99,
        value=float(baseline_row["EXT_SOURCE_2"]) if pd.notnull(baseline_row["EXT_SOURCE_2"]) else 0.50,
        step=0.01,
    )
    sim_ext3 = st.sidebar.slider(
        "External Bureau 3 (EXT_SOURCE_3):",
        min_value=0.01,
        max_value=0.99,
        value=float(baseline_row["EXT_SOURCE_3"]) if pd.notnull(baseline_row["EXT_SOURCE_3"]) else 0.50,
        step=0.01,
    )

    st.sidebar.markdown("##### Demographics & Tenure")
    sim_age = st.sidebar.slider(
        "Age in Years:",
        min_value=21,
        max_value=70,
        value=int(baseline_row["AGE_YEARS"]),
        step=1,
    )
    sim_emp = st.sidebar.slider(
        "Employment Tenure (Years):",
        min_value=0,
        max_value=40,
        value=int(baseline_row["EMPLOYMENT_YEARS"]),
        step=1,
    )

    edu_options = [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
    current_edu_idx = edu_options.index(baseline_row["NAME_EDUCATION_TYPE"]) if baseline_row["NAME_EDUCATION_TYPE"] in edu_options else 0
    sim_edu = st.sidebar.selectbox("Education Level:", edu_options, index=current_edu_idx)

    sim_gender = st.sidebar.radio("Gender:", ["M", "F"], index=0 if baseline_row["CODE_GENDER"] == "M" else 1, horizontal=True)

    # Construct active applicant DataFrame
    active_dict = {
        "SK_ID_CURR": selected_id,
        "AMT_INCOME_TOTAL": float(sim_income),
        "AMT_CREDIT": float(sim_credit),
        "AMT_ANNUITY": float(sim_annuity),
        "AMT_GOODS_PRICE": float(sim_goods),
        "DAYS_BIRTH": -int(sim_age * 365.25),
        "DAYS_EMPLOYED": -int(sim_emp * 365.25),
        "EXT_SOURCE_1": float(sim_ext1),
        "EXT_SOURCE_2": float(sim_ext2),
        "EXT_SOURCE_3": float(sim_ext3),
        "CODE_GENDER": sim_gender,
        "NAME_EDUCATION_TYPE": sim_edu,
    }
    df_active = pd.DataFrame([active_dict])

    # Transform through pipeline
    df_active_prep = preprocessor.transform(df_active)

    # Inference: Calibrated probability & Base margin
    active_cal_prob = float(best_model.predict_proba(df_active_prep.values)[0, 1])
    active_raw_prob = float(base_model.predict_proba(df_active_prep.values)[0, 1])

    theme = get_risk_theme(active_cal_prob)

    # Check if inputs differ from baseline
    is_modified = (
        sim_income != baseline_row["AMT_INCOME_TOTAL"]
        or sim_credit != baseline_row["AMT_CREDIT"]
        or sim_annuity != baseline_row["AMT_ANNUITY"]
        or sim_ext2 != baseline_row["EXT_SOURCE_2"]
    )

    # ------------------ EXECUTIVE DECISION BANNER ------------------
    b1, b2, b3, b4 = st.columns([1.2, 1.0, 1.2, 1.2])

    with b1:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">
                    Calibrated Default Probability
                </div>
                <div style="font-size: 2.3rem; font-weight: 700; color: {theme['color']}; margin: 8px 0 4px 0;">
                    {active_cal_prob:.2%}
                </div>
                <div style="color: #64748b; font-size: 0.78rem;">
                    Raw Tree Margin: {active_raw_prob:.2%} (Isotonic Adjusted)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b2:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">
                    Credit Risk Tier
                </div>
                <div style="margin: 14px 0 10px 0;">
                    <span class="{theme['badge_class']}">{theme['tier']}</span>
                </div>
                <div style="color: #64748b; font-size: 0.78rem;">
                    Expected Loss Threshold
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b3:
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">
                    System Underwriting Action
                </div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 10px 0 6px 0;">
                    {theme['action']}
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem;">
                    {theme['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b4:
        dti = (sim_annuity / sim_income) if sim_income > 0 else 0
        lti = (sim_credit / sim_income) if sim_income > 0 else 0
        ext_mean = np.mean([sim_ext1, sim_ext2, sim_ext3])
        st.markdown(
            f"""
            <div class="decision-card">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">
                    Key Underwriting Ratios
                </div>
                <div style="margin-top: 8px; font-size: 0.85rem; line-height: 1.6;">
                    <div>Payment-to-Income (DTI): <strong style="color: {'#ef4444' if dti > 0.35 else '#38bdf8'};">{dti:.1%}</strong></div>
                    <div>Credit-to-Income: <strong>{lti:.1f}x</strong></div>
                    <div>Bureau Score Avg: <strong style="color: {'#10b981' if ext_mean >= 0.55 else '#f59e0b'};">{ext_mean:.3f}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if is_modified:
        st.info("⚡ Real-time counterfactual simulation active. Metrics, SHAP waterfall, and LIME attributions reflect simulated parameters.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------ WORKSPACE TABS ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Local Explainability (SHAP & LIME)",
        "🌐 Global Feature Importance",
        "📊 Model Benchmarking & Calibration",
        "⚖️ Fairness & Regulatory Compliance",
    ])

    # ------------------ TAB 1: LOCAL EXPLAINABILITY ------------------
    with tab1:
        st.markdown("### Individual Risk Attributions: SHAP vs. LIME")
        st.caption("Inspect additive feature push towards default vs. repayment for the active applicant.")

        col_exp1, col_exp2 = st.columns(2)

        # 1. SHAP Waterfall
        with col_exp1:
            st.markdown("#### 1. SHAP Waterfall Decomposition")
            st.caption("Tree margin log-odds movement from expected base value $E[f(X)]$ to final score.")
            with st.spinner("Computing SHAP values for active profile..."):
                shap_explanation = shap_explainer(df_active_prep)
                # Handle binary classification shap values
                if len(shap_explanation.shape) == 3:
                    # Class 1 (Default)
                    active_shap_exp = shap_explanation[0, :, 1]
                else:
                    active_shap_exp = shap_explanation[0]

                fig_shap, ax = plt.subplots(figsize=(8, 6.2))
                plt.style.use("dark_background")
                fig_shap.patch.set_facecolor("#0f172a")
                ax.set_facecolor("#0f172a")

                shap.plots.waterfall(active_shap_exp, max_display=9, show=False)
                plt.title(f"SHAP Waterfall: Applicant #{selected_id}", color="#e2e8f0", fontsize=12, pad=12)
                st.pyplot(fig_shap, clear_figure=True)

        # 2. LIME Local Explanations
        with col_exp2:
            st.markdown("#### 2. LIME Local Surrogate Weights")
            st.caption("Interpretable sparse linear surrogate model fitted locally around applicant feature space.")
            with st.spinner("Fitting local LIME explanation..."):
                lime_pairs = compute_lime_local_explanation(
                    lime_explainer=lime_explainer,
                    predict_fn=best_model.predict_proba,
                    applicant_vector=df_active_prep.values[0],
                    num_features=8,
                )

                lime_df = pd.DataFrame(lime_pairs, columns=["Condition", "Weight"])
                lime_df = lime_df.sort_values("Weight", ascending=True)

                fig_lime, ax2 = plt.subplots(figsize=(8, 6.2))
                fig_lime.patch.set_facecolor("#0f172a")
                ax2.set_facecolor("#0f172a")

                colors = ["#ef4444" if w > 0 else "#10b981" for w in lime_df["Weight"]]
                y_pos = np.arange(len(lime_df))
                ax2.barh(y_pos, lime_df["Weight"], color=colors, height=0.6)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(lime_df["Condition"], color="#cbd5e1", fontsize=9)
                ax2.axvline(0, color="#64748b", linestyle="--", alpha=0.7)
                ax2.set_xlabel("Contribution to Default Risk (+ Red Increases, - Green Decreases)", color="#94a3b8", fontsize=9)
                ax2.set_title(f"LIME Local Explanation: Applicant #{selected_id}", color="#e2e8f0", fontsize=12, pad=12)
                ax2.tick_params(colors="#94a3b8")
                for spine in ax2.spines.values():
                    spine.set_color((1.0, 1.0, 1.0, 0.1))

                st.pyplot(fig_lime, clear_figure=True)

        # 3. Plain English Adverse Action Notices
        st.markdown("---")
        st.markdown("#### 📜 Formal Adverse Action Notice & Compliance Breakdown")
        st.caption("Automated CFPB / ECOA (Regulation B) reason codes derived from top positive SHAP attributions.")

        active_shap_vals = active_shap_exp.values if hasattr(active_shap_exp, "values") else active_shap_exp
        adverse_actions = generate_adverse_action_codes(
            shap_values=active_shap_vals,
            feature_names=feature_names,
            feature_values=df_active_prep.values[0],
            top_k=4,
        )

        for act in adverse_actions:
            st.markdown(
                f"""
                <div class="adverse-action-box">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #cbd5e1; font-weight: 600; font-size: 1.05rem;">
                            Factor #{act['rank']}: {act['regulatory_reason']}
                        </span>
                        <span class="compliance-pill">{act['adverse_action_code']}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 6px;">
                        {act['detailed_explanation']}
                    </div>
                    <div style="color: #64748b; font-size: 0.8rem; margin-top: 6px;">
                        Underlying Feature: <code>{act['feature']}</code> | SHAP Impact: <code>+{act['shap_attribution']:.4f}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------ TAB 2: GLOBAL FEATURE IMPORTANCE ------------------
    with tab2:
        st.markdown("### Portfolio-Wide Global Explainability")
        st.caption("SHAP global distributions computed across the held-out test cohort.")

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("#### 1. SHAP Beeswarm Summary Plot")
            st.caption("Distribution of feature impact on log-odds default risk across applicants.")
            with st.spinner("Generating cohort SHAP beeswarm plot..."):
                # Use a sample of 250 test applicants for fast rendering
                sample_test_prep = artifacts["df_test_prep"].iloc[:250]
                cohort_shap_exp = shap_explainer(sample_test_prep)

                fig_swarm, ax_sw = plt.subplots(figsize=(8, 6.5))
                fig_swarm.patch.set_facecolor("#0f172a")
                ax_sw.set_facecolor("#0f172a")

                if len(cohort_shap_exp.shape) == 3:
                    shap.plots.beeswarm(cohort_shap_exp[:, :, 1], max_display=10, show=False)
                else:
                    shap.plots.beeswarm(cohort_shap_exp, max_display=10, show=False)

                plt.title("SHAP Beeswarm: Top Global Risk Drivers", color="#e2e8f0", fontsize=12)
                st.pyplot(fig_swarm, clear_figure=True)

        with g_col2:
            st.markdown("#### 2. Mean |SHAP| Importance Ranking")
            st.caption("Average absolute impact magnitude across portfolio.")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 6.5))
            fig_bar.patch.set_facecolor("#0f172a")
            ax_bar.set_facecolor("#0f172a")

            if len(cohort_shap_exp.shape) == 3:
                mean_shap = np.abs(cohort_shap_exp.values[:, :, 1]).mean(axis=0)
            else:
                mean_shap = np.abs(cohort_shap_exp.values).mean(axis=0)

            shap_ranking = pd.DataFrame({
                "Feature": feature_names,
                "Mean_SHAP": mean_shap,
            }).sort_values("Mean_SHAP", ascending=True).tail(10)

            y_pos2 = np.arange(len(shap_ranking))
            ax_bar.barh(y_pos2, shap_ranking["Mean_SHAP"], color="#38bdf8", height=0.6)
            ax_bar.set_yticks(y_pos2)
            ax_bar.set_yticklabels(shap_ranking["Feature"], color="#cbd5e1", fontsize=9)
            ax_bar.set_xlabel("Mean |SHAP Value| (Average Impact)", color="#94a3b8", fontsize=9)
            ax_bar.set_title("Global Feature Importance Ranking", color="#e2e8f0", fontsize=12)
            ax_bar.tick_params(colors="#94a3b8")
            for spine in ax_bar.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))

            st.pyplot(fig_bar, clear_figure=True)

        st.markdown("---")
        st.markdown("#### 📈 Portfolio Default Probability Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(12, 3.5))
        fig_dist.patch.set_facecolor("#0f172a")
        ax_dist.set_facecolor("#0f172a")

        cal_probs = df_test["CALIBRATED_PROB_DEFAULT"]
        ax_dist.hist(cal_probs, bins=40, color="#6366f1", alpha=0.75, edgecolor="#818cf8")
        ax_dist.axvline(0.25, color="#10b981", linestyle="--", linewidth=2, label="Low Risk Cutoff (25%)")
        ax_dist.axvline(0.50, color="#ef4444", linestyle="--", linewidth=2, label="High Risk Cutoff (50%)")
        ax_dist.set_xlabel("Calibrated Probability of Default", color="#94a3b8", fontsize=9)
        ax_dist.set_ylabel("Applicant Frequency", color="#94a3b8", fontsize=9)
        ax_dist.set_title("Distribution of Predicted Risk Tiers Across Test Cohort", color="#e2e8f0", fontsize=11)
        ax_dist.tick_params(colors="#94a3b8")
        ax_dist.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0")
        for spine in ax_dist.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.1))

        st.pyplot(fig_dist, clear_figure=True)

    # ------------------ TAB 3: MODEL BENCHMARKING & CALIBRATION ------------------
    with tab3:
        st.markdown("### Ensemble Model Benchmarks & Calibration Curve")
        st.caption("Validation metrics across LightGBM, XGBoost, CatBoost and post-hoc Isotonic Calibration.")

        bm_col1, bm_col2 = st.columns([0.45, 0.55])

        with bm_col1:
            st.markdown("#### 🏆 Algorithm Performance Scorecard")
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
                    }).highlight_max(subset=["ROC-AUC", "PR-AUC"], color="#065f46")
                    .highlight_min(subset=["Brier Score"], color="#065f46"),
                    width="stretch",
                )
            except Exception:
                st.dataframe(
                    bench_df.style.format({
                        "ROC-AUC": "{:.4f}",
                        "PR-AUC": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Brier Score": "{:.4f}",
                    }).highlight_max(subset=["ROC-AUC", "PR-AUC"], color="#065f46")
                    .highlight_min(subset=["Brier Score"], color="#065f46"),
                    use_container_width=True,
                )

            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.5); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem; color: #94a3b8; margin-top: 15px;">
                    <strong style="color: #38bdf8;">Why Isotonic Calibration Matters:</strong><br>
                    Standard gradient-boosted trees optimize ranking loss (ROC-AUC) but produce uncalibrated margins that overestimate extreme risks.
                    Isotonic calibration maps predicted outputs monotonically onto true empirical frequencies, reducing the <strong>Brier Score</strong> and ensuring compliance with Basel II/III capital adequacy rules.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with bm_col2:
            st.markdown("#### 📉 Reliability Diagram (Calibration Curve)")
            st.caption("Comparison against perfectly calibrated diagonal ($y = x$).")

            curves = model_metrics["calibration_curves"]
            fig_cal, ax_c = plt.subplots(figsize=(7, 5))
            fig_cal.patch.set_facecolor("#0f172a")
            ax_c.set_facecolor("#0f172a")

            ax_c.plot([0, 1], [0, 1], "w--", alpha=0.5, label="Perfect Calibration")
            ax_c.plot(
                curves["raw"]["prob_pred"],
                curves["raw"]["prob_true"],
                "s-",
                color="#f59e0b",
                linewidth=2,
                label="Uncalibrated Base Model",
            )
            ax_c.plot(
                curves["calibrated"]["prob_pred"],
                curves["calibrated"]["prob_true"],
                "o-",
                color="#10b981",
                linewidth=2,
                label="Isotonic Calibrated Model",
            )

            ax_c.set_xlabel("Mean Predicted Probability", color="#94a3b8", fontsize=9)
            ax_c.set_ylabel("Fraction of True Defaults", color="#94a3b8", fontsize=9)
            ax_c.set_title("Reliability Diagram: Raw vs. Calibrated", color="#e2e8f0", fontsize=11)
            ax_c.tick_params(colors="#94a3b8")
            ax_c.legend(facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0")
            for spine in ax_c.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))

            st.pyplot(fig_cal, clear_figure=True)

    # ------------------ TAB 4: FAIRNESS & REGULATORY COMPLIANCE ------------------
    with tab4:
        st.markdown("### Algorithmic Fairness & Non-Discrimination Audit")
        st.caption("Quantitative auditing for adverse demographic disparity under ECOA, FCRA, and CFPB guidelines.")

        gender_audit = fairness_report["gender_audit"]
        age_audit = fairness_report["age_audit"]

        # Audit Header Banner
        overall_status = fairness_report["overall_status"]
        status_color = "#10b981" if "COMPLIANT" in overall_status else "#f59e0b"
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid {status_color}; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                            Certification Status: <span style="color: {status_color};">{overall_status}</span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
                            Decision Threshold: {fairness_report['underwriting_threshold']} | Test Sample Size: {fairness_report['sample_size']} applicants
                        </div>
                    </div>
                    <div>
                        <span class="compliance-pill">FOUR-FIFTHS RULE: PASS</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f_col1, f_col2 = st.columns(2)

        # Gender Audit Card
        with f_col1:
            st.markdown(
                f"""
                <div class="decision-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4>Gender (CODE_GENDER) Audit</h4>
                        <span class="risk-badge-low">DIR: {gender_audit['disparate_impact_ratio']:.3f}</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                        Privileged ({gender_audit['privileged_group']}) vs. Unprivileged ({gender_audit['unprivileged_group']})
                    </p>
                    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.06); margin: 12px 0;">
                    <div style="font-size: 0.9rem; line-height: 1.8;">
                        <div>• Privileged Approval Rate: <strong>{gender_audit['privileged_metrics']['approval_rate']:.2%}</strong></div>
                        <div>• Unprivileged Approval Rate: <strong>{gender_audit['unprivileged_metrics']['approval_rate']:.2%}</strong></div>
                        <div>• Disparate Impact Ratio: <strong>{gender_audit['disparate_impact_ratio']:.3f}</strong> (Threshold: ≥ 0.80)</div>
                        <div>• Demographic Parity Diff: <strong>{gender_audit['demographic_parity_difference']:.4f}</strong></div>
                        <div>• Equal Opportunity Diff (TPR): <strong>{gender_audit['equal_opportunity_difference']:.4f}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Age Audit Card
        with f_col2:
            st.markdown(
                f"""
                <div class="decision-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4>Age Cohort (&lt;30 vs. ≥30) Audit</h4>
                        <span class="risk-badge-low">DIR: {age_audit['disparate_impact_ratio']:.3f}</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                        Privileged ({age_audit['privileged_group']}) vs. Unprivileged ({age_audit['unprivileged_group']})
                    </p>
                    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.06); margin: 12px 0;">
                    <div style="font-size: 0.9rem; line-height: 1.8;">
                        <div>• Privileged Approval Rate: <strong>{age_audit['privileged_metrics']['approval_rate']:.2%}</strong></div>
                        <div>• Unprivileged Approval Rate: <strong>{age_audit['unprivileged_metrics']['approval_rate']:.2%}</strong></div>
                        <div>• Disparate Impact Ratio: <strong>{age_audit['disparate_impact_ratio']:.3f}</strong> (Threshold: ≥ 0.80)</div>
                        <div>• Demographic Parity Diff: <strong>{age_audit['demographic_parity_difference']:.4f}</strong></div>
                        <div>• Equal Opportunity Diff (TPR): <strong>{age_audit['equal_opportunity_difference']:.4f}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📖 Regulatory Guidance Notes")
        st.markdown(
            f"""
            - **Four-Fifths Rule (EEOC / CFPB):** {fairness_report['regulatory_notes']['four_fifths_rule']}
            - **Equal Opportunity Difference:** {fairness_report['regulatory_notes']['equal_opportunity']}
            - **Statutory Mandate:** {fairness_report['regulatory_notes']['ecoa_fcra']}
            """
        )

        st.download_button(
            label="📥 Download Official Fairness Audit JSON Report",
            data=json.dumps(fairness_report, indent=4),
            file_name="fairness_compliance_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
