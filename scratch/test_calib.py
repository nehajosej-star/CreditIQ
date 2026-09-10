import numpy as np

def compute_calibrated_underwrite_prob(
    base_model_prob: float,
    credit_to_income: float,
    annuity_to_income: float,
    ext_sources_mean: float,
    credit_term: float,
) -> float:
    # Start from logit of base model probability
    p = np.clip(base_model_prob, 0.005, 0.995)
    base_logit = np.log(p / (1.0 - p))
    
    # Domain underwriting risk adjustments:
    # 1. External Bureau Health (target benchmark is 0.50; <0.20 heavily penalized)
    ext_delta = 0.50 - ext_sources_mean
    bureau_penalty = 3.2 * ext_delta
    if ext_sources_mean < 0.20:
        bureau_penalty += 1.8 * (0.20 - ext_sources_mean) / 0.20
        
    # 2. Debt-to-Income / Monthly Repayment Burden (safe threshold <= 0.28)
    dti_penalty = 2.8 * max(0.0, annuity_to_income - 0.28)
    if annuity_to_income > 0.45:
        dti_penalty += 1.8 * (annuity_to_income - 0.45)
        
    # 3. Loan-to-Income Multiple (safe threshold <= 3.5x)
    lti_penalty = 0.40 * max(0.0, credit_to_income - 3.5)
    if credit_to_income > 6.0:
        lti_penalty += 0.40 * (credit_to_income - 6.0)
        
    # 4. Bureau premium for excellent scores (> 0.65)
    bureau_discount = 0.8 * max(0.0, ext_sources_mean - 0.65)
    
    # Combined calibrated logit
    adjusted_logit = base_logit + bureau_penalty + dti_penalty + lti_penalty - bureau_discount
    
    # Sigmoidal mapping to calibrated probability
    calibrated_p = 1.0 / (1.0 + np.exp(-adjusted_logit))
    return float(np.clip(calibrated_p, 0.005, 0.985))

test_cases = [
    ("Prime Borrower (LTI 2.0x, DTI 15%, Bureau 0.80)", 0.08, 2.0, 0.15, 0.80, 0.07),
    ("Healthy Borrower (LTI 3.2x, DTI 25%, Bureau 0.55)", 0.15, 3.2, 0.25, 0.55, 0.07),
    ("Borderline / Elevated Debt (LTI 4.5x, DTI 35%, Bureau 0.45)", 0.22, 4.5, 0.35, 0.45, 0.07),
    ("Moderate Risk (LTI 5.5x, DTI 38%, Bureau 0.35)", 0.25, 5.5, 0.38, 0.35, 0.07),
    ("High Risk: Low Bureau < 0.20 (LTI 4.0x, DTI 28%, Bureau 0.15)", 0.20, 4.0, 0.28, 0.15, 0.07),
    ("High Risk: High Loan multiple 8.5x (LTI 8.5x, DTI 50%, Bureau 0.40)", 0.28, 8.5, 0.50, 0.40, 0.07),
    ("Severe Distress (LTI 10x, DTI 60%, Bureau 0.10)", 0.35, 10.0, 0.60, 0.10, 0.07),
]

for name, bp, lti, dti, ext, ct in test_cases:
    cp = compute_calibrated_underwrite_prob(bp, lti, dti, ext, ct)
    tier = "Low Risk (<0.25)" if cp < 0.25 else "Medium Risk (0.25-0.50)" if cp < 0.50 else "High Risk (>=0.50)"
    print(f"{name:60} -> prob: {cp*100:5.1f}% | {tier}")
