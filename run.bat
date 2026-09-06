@echo off
REM One-click script for Windows to install dependencies, train models, audit fairness, and launch CreditIQ Cockpit

echo ==========================================================================
echo 🏦 CreditIQ: Smart Financial Risk ^& Loan Default Predictor with Explainability
echo ==========================================================================

echo [1/3] Verifying and installing required Python packages...
pip install -r requirements.txt

echo [2/3] Checking model artifacts...
if not exist "models\best_model.pkl" (
    echo Model artifacts not found. Executing end-to-end training and evaluation pipeline...
    python src\train.py
) else (
    echo Found existing trained models and explainers in models\ directory.
)

echo [3/3] Launching Streamlit Loan Officer Decision Cockpit...
echo Access the cockpit locally at: http://localhost:8501
streamlit run app\streamlit_app.py --server.port 8501 --server.headless false
pause
