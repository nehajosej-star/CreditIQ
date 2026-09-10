import sys
sys.path.insert(0, ".")
import joblib
import numpy as np
import pandas as pd
from src.explainers import compute_lime_local_explanation

models_dir = "models"
preprocessor = joblib.load(f"{models_dir}/preprocessor.pkl")
base_model = joblib.load(f"{models_dir}/base_model.pkl")
best_model = joblib.load(f"{models_dir}/best_model.pkl")
with open(f"{models_dir}/lime_explainer.pkl", "rb") as f:
    import cloudpickle
    lime_explainer = cloudpickle.load(f)

feature_names = preprocessor.feature_names_
c2i_idx = feature_names.index("CREDIT_TO_INCOME_RATIO")
a2i_idx = feature_names.index("ANNUITY_TO_INCOME_RATIO")
ext_idx = feature_names.index("EXT_SOURCES_MEAN")
term_idx = feature_names.index("CREDIT_TERM")
inc_idx = feature_names.index("AMT_INCOME_TOTAL")
cred_idx = feature_names.index("AMT_CREDIT")
ann_idx = feature_names.index("AMT_ANNUITY")
ext1_idx = feature_names.index("EXT_SOURCE_1")
ext2_idx = feature_names.index("EXT_SOURCE_2")
ext3_idx = feature_names.index("EXT_SOURCE_3")

def calibrate_prob_vectorized(X: np.ndarray) -> np.ndarray:
    # X is (N, n_features)
    base_probs = base_model.predict_proba(X)[:, 1]
    
    # Extract domain features
    c2i = X[:, c2i_idx]
    a2i = X[:, a2i_idx]
    ext = X[:, ext_idx]
    
    # Also check raw columns if ratios were not updated in perturbed rows
    inc = np.maximum(X[:, inc_idx], 1.0)
    cred = np.maximum(X[:, cred_idx], 1.0)
    ann = np.maximum(X[:, ann_idx], 0.0)
    
    eff_c2i = np.maximum(c2i, cred / inc)
    eff_a2i = np.maximum(a2i, ann / inc)
    
    ext1 = X[:, ext1_idx]
    ext2 = X[:, ext2_idx]
    ext3 = X[:, ext3_idx]
    eff_ext = np.where(ext > 0, ext, np.mean([ext1, ext2, ext3], axis=0))
    
    p = np.clip(base_probs, 0.005, 0.995)
    base_logit = np.log(p / (1.0 - p))
    
    # External bureau health: benchmark 0.50, severe penalty below 0.20
    bureau_penalty = 3.2 * (0.50 - eff_ext)
    low_bureau_mask = eff_ext < 0.20
    bureau_penalty += np.where(low_bureau_mask, 1.8 * (0.20 - eff_ext) / 0.20, 0.0)
    
    # Debt-to-income penalty
    dti_penalty = 2.8 * np.maximum(0.0, eff_a2i - 0.28)
    dti_penalty += np.where(eff_a2i > 0.45, 1.8 * (eff_a2i - 0.45), 0.0)
    
    # Loan-to-income penalty
    lti_penalty = 0.40 * np.maximum(0.0, eff_c2i - 3.5)
    lti_penalty += np.where(eff_c2i > 6.0, 0.40 * (eff_c2i - 6.0), 0.0)
    
    # Bureau bonus
    bureau_bonus = 0.8 * np.maximum(0.0, eff_ext - 0.65)
    
    adj_logit = base_logit + bureau_penalty + dti_penalty + lti_penalty - bureau_bonus
    cal_p = 1.0 / (1.0 + np.exp(-adj_logit))
    cal_p = np.clip(cal_p, 0.005, 0.985)
    
    # Return (N, 2)
    return np.column_stack([1.0 - cal_p, cal_p])

print("Vectorized calibrator defined. Testing on standard applicant...")
df_test = pd.read_parquet(f"{models_dir}/test_data.parquet")
sample_row = df_test.iloc[0].to_dict()

# Baseline
x_base = preprocessor.transform(pd.DataFrame([sample_row])).values[0]
p_base = calibrate_prob_vectorized(x_base.reshape(1, -1))[0, 1]
print(f"Applicant {sample_row['SK_ID_CURR']} base risk: {p_base*100:.1f}%")

# Now simulate distressed applicant: credit 1.8M, income 120k, debts 0.12
distressed_row = dict(sample_row)
distressed_row["AMT_INCOME_TOTAL"] = 120000.0
distressed_row["AMT_CREDIT"] = 1800000.0
distressed_row["AMT_ANNUITY"] = 90000.0
distressed_row["EXT_SOURCE_1"] = 0.12
distressed_row["EXT_SOURCE_2"] = 0.12
distressed_row["EXT_SOURCE_3"] = 0.12
x_distressed = preprocessor.transform(pd.DataFrame([distressed_row])).values[0]
p_distressed = calibrate_prob_vectorized(x_distressed.reshape(1, -1))[0, 1]
print(f"Distressed applicant risk: {p_distressed*100:.1f}%")

print("Testing LIME on distressed applicant...")
lime_rules = compute_lime_local_explanation(
    lime_explainer=lime_explainer,
    predict_fn=calibrate_prob_vectorized,
    applicant_vector=x_distressed,
    num_features=5,
)
print("Distressed LIME rules:")
for cond, weight in lime_rules:
    print(f"  {cond:50} -> {weight:+.4f}")
