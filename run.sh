#!/usr/bin/env bash
# One-click script to install dependencies, train models, audit fairness, and launch CreditIQ Cockpit

set -e

echo "=========================================================================="
echo "🏦 CreditIQ: Smart Financial Risk & Loan Default Predictor with Explainability"
echo "=========================================================================="

echo "[1/3] Verifying and installing required Python packages..."
pip install -r requirements.txt

echo "[2/3] Checking model artifacts..."
if [ ! -f "models/best_model.pkl" ] || [ ! -f "models/shap_explainer.pkl" ]; then
    echo "Model artifacts not found. Executing end-to-end training and evaluation pipeline..."
    python src/train.py
else
    echo "Found existing trained models and explainers in models/ directory."
fi

echo "[3/3] Launching Streamlit Loan Officer Decision Cockpit..."
echo "Access the cockpit locally at: http://localhost:8501"
streamlit run app/streamlit_app.py --server.port 8501 --server.headless false


