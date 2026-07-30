# CreditSight
"An explainable credit risk pipeline that pairs XGBoost predictions with Gemini-generated underwriting explanations."
# Credit Risk Pipeline 🏦

An end-to-end credit risk scoring pipeline that combines a traditional machine learning model with a large language model to produce **accurate, explainable** loan default predictions.

## What it does

1. **Downloads** a real-world credit risk dataset (~32K historical loan applicants) from Kaggle
2. **Cleans** the data, removing unrealistic outliers
3. **Trains** an XGBoost classifier with 3-fold stratified cross-validation
4. **Calibrates** the model's probability outputs so they're statistically trustworthy, not just relative rankings
5. **Scores** a new applicant, producing a Probability of Default (PD)
6. **Explains** the score using Google's Gemini Flash model, generating a plain-English risk summary and an underwriting recommendation (Approve / Decline / Manual Review)

## How it works

```
Applicant data ──► XGBoost ──► PD (e.g. 93.87%)
                                    │
Applicant data + PD ──► Gemini Flash ──► Human-readable explanation
```

- **XGBoost** is the predictive engine — it finds statistical patterns in historical data and outputs a single number.
- **Gemini Flash** is the explanation layer — it takes that number and the applicant's profile and writes up the reasoning like a human credit analyst would. It does no math of its own; it only interprets what XGBoost already calculated.

This separation of concerns — a specialized ML model for prediction, an LLM for explanation — is a practical pattern for building explainable AI in regulated industries like finance.

## Tech stack

| Component | Tool |
|---|---|
| Data source | [Kaggle: Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) via `kagglehub` |
| Preprocessing | scikit-learn (`ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`) |
| Model | XGBoost (`XGBClassifier`), calibrated with `CalibratedClassifierCV` |
| Explanation | Google Gemini API (`gemini-flash-latest`) |
| Environment | Google Colab (also runs as a standalone Python script) |

## Getting started

### Requirements
```bash
pip install kagglehub google-genai xgboost scikit-learn pandas numpy
```

### Credentials needed
- **Kaggle API token** — from [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token"
- **Gemini API key** (optional, only needed for the explanation step) — from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

> ⚠️ Never hardcode API keys in your code or notebook. Set them via environment variables or an interactive prompt (`getpass`) so they aren't saved or shared accidentally.

### Run it
A ready-to-run Google Colab notebook is included (`credit_risk_pipeline_colab.ipynb`), broken into clearly labeled cells:
1. Install dependencies
2. Download the dataset
3. Set your Gemini key (via a secure prompt)
4. Load & clean data
5. Build preprocessing pipeline
6. Train & calibrate the model
7. Score a sample applicant
8. Generate an AI explanation of the result

## Example output

```
Calculated PD for Applicant: 93.87%

Key Risk Drivers:
- Loan-to-income ratio of 74% — unsustainable debt burden
- Prior default + thin (2-year) credit history
- No collateral, short employment tenure, renter

Recommendation: DECLINE
```

## Model performance

Reported via 3-fold stratified cross-validation ROC-AUC during training — see console output when running the pipeline.

## Disclaimer

This is a demonstration/educational project. It is **not** a production-grade credit decisioning system and should not be used for real lending decisions without proper validation, fairness auditing, and regulatory review.

## License

MIT
