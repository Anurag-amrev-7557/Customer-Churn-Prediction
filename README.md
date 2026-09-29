# Customer Churn Analysis & Retention Strategy

A portfolio-grade, end-to-end data science and machine learning project analyzing customer attrition on the **IBM Telco Customer Churn** dataset (7,043 customer accounts).

This project goes beyond raw model scores to translate attrition risk into actionable business strategies, answering three core commercial questions with empirical evidence and financial simulation.

---

## 🎯 Executive Answers to Core Business Questions

### 1. Which customer segments are most likely to churn?
* **Month-to-Month Contract Customers**: Exhibit a **42.7% churn rate**, compared to **11.3%** for 1-year contracts and just **2.8%** for 2-year contracts. Month-to-month subscribers represent **88.6% of all churned customers**.
* **Fiber Optic Internet Users**: Experience **41.9% churn** (vs. **18.9%** for DSL and **7.4%** for customers without internet service). Despite higher speeds, fiber customers experience higher attrition due to higher monthly fees ($70–$110/mo) and competitive market alternatives.
* **Electronic Check Payers**: Suffer a **45.3% churn rate**, more than double that of credit card autopay (**15.2%**), bank transfer autopay (**16.7%**), or mailed checks (**19.1%**).
* **Early-Tenure Customers (0–12 Months)**: Suffer a **47.4% churn rate**, dropping down to **6.6%** once customers surpass 48 months of tenure.
* **Unprotected Accounts**: Customers without **TechSupport** or **OnlineSecurity** add-ons churn at **41.6%** and **41.8%** respectively, compared to **15.2%** and **14.6%** for customers with these services enabled.

### 2. Which customer characteristics are most strongly associated with churn?
* **Tenure ($r = -0.352, p < 10^{-200}$, Odds Ratio = $0.285$)**: The single strongest protective attribute against attrition. Each standard-deviation increase in customer tenure slashes churn odds by **71.5%**.
* **Contract Structure (Cramér's $V = 0.410$, Odds Ratio = $1.797$)**: Month-to-month contracts elevate churn odds by **+79.7%** compared to long-term commitments.
* **Fiber Optic Internet (Cramér's $V = 0.323$, Odds Ratio = $1.912$)**: Fiber optic subscription nearly doubles churn odds (**+91.2%** increase in risk).
* **Payment Method (Cramér's $V = 0.303$)**: Electronic check payment represents an active friction point and lack of automated billing commitment.
* **Absence of Tech Support / Online Security (Cramér's $V \approx 0.345$)**: Services that build platform stickiness and reduce customer frustration when technical issues occur.

### 3. How can a telecom company use churn predictions to prioritize customer-retention efforts?
* **Concentrated Risk Decile Targeting**: Rather than blast-contacting all customers or contacting customers at random, sorting customers by predicted churn probability into risk deciles allows the company to capture **50.5% of all churners by contacting only the top 20% highest-risk accounts** (a **2.52x Cumulative Lift**).
* **High-ROI Economic Framework**: In our 1,409-customer held-out test cohort, targeting the top 20% with a $40 retention incentive and an achievable 30% save rate generates:
  * **$52,251.66** in gross preserved annual recurring revenue.
  * **$11,240.00** in campaign incentive costs.
  * **$41,011.66 in net profit** with a **364.9% campaign ROI**.
  * **+$35,042.81 in profit lift** over non-targeted outreach.
* **Tiered Intervention Playbooks**:
  1. *Tier 1 (Critical Risk, Prob > 60%)*: Outbound intervention by dedicated retention agents offering 1-year contract lock discounts and free tech support add-ons.
  2. *Tier 2 (Moderate Risk, Prob 40–60%)*: Automated in-app/email nudge offering a $5/month discount for switching from electronic check to automatic credit card billing.
  3. *Tier 3 (Low Risk, Prob < 40%)*: Zero retention discounting to prevent unnecessary margin cannibalization on naturally loyal customers.

---

## 🏗️ Project Architecture & Workflow

```
customer-churn/
├── app.py                      # Interactive Streamlit executive dashboard & live scoring
├── requirements.txt            # Python dependencies
├── .cursorrules                # Cursor IDE execution rules & safeguards
├── data/
│   ├── raw/                    # Raw IBM Telco dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   └── processed/              # Cleaned dataset (telco_churn_cleaned.csv)
├── models/
│   ├── best_model.joblib       # Serialized champion model pipeline (ColumnTransformer + RF)
│   ├── model_comparison.json   # Benchmark metrics, CV scores, and ROC/PR curve points
│   └── test_predictions.csv    # Test set scored probabilities and actual outcomes
├── reports/
│   ├── decile_analysis.csv     # Decile table with lift and cumulative gains
│   ├── eda_summary.json        # Segment profiles and Cramér's V associations
│   ├── explainability_summary.json # Odds ratios and permutation importances
│   ├── retention_strategy_summary.json # ROI simulation and playbooks
│   └── figures/                # Publication-quality charts (.png)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Validation, missing value handling, typing
│   ├── eda.py                  # Statistical associations, segment profiling, charts
│   ├── preprocessing.py        # Leakage-free stratified split & ColumnTransformer
│   ├── models.py               # Benchmarking, 5-Fold CV, imbalance comparison
│   ├── explainability.py       # Odds ratios, Gini importance, permutation tests
│   └── business_impact.py      # Decile lift, gains curves, ROI simulation
└── tests/
    ├── test_data_loader.py     # Schema validation and TotalCharges test
    ├── test_preprocessing.py   # Stratification and pipeline shape test
    ├── test_models.py          # Model inference and ROC-AUC test (>= 0.80)
    └── test_business_impact.py # Decile lift and ROI simulation tests
```

---

## 🔬 Preprocessing & Data Validation

The IBM Telco dataset contains 7,043 rows and 21 columns. Data validation identified two crucial preprocessing details:
1. **Handling `TotalCharges` Blanks**:
   * Exactly 11 customer records contained blank strings (`" "`) in `TotalCharges`.
   * Inspection confirmed that all 11 records correspond to brand new customers with `tenure == 0` months.
   * Rather than dropping these valid customers or introducing bias with global mean imputation, their `TotalCharges` was imputed as `0.0` (reflecting zero accumulated charges to date).
2. **Leakage-Free Feature Pipeline**:
   * Train/Test separation (80/20 stratified split: 5,634 train / 1,409 test) is applied **before** fitting any transformers.
   * Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled using `StandardScaler`.
   * Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` yielding 46 encoded features.

---

## 📊 Model Benchmarking & Class Imbalance Experiments

Because churn represents 26.54% of the population, we benchmarked multiple model families alongside class balancing strategies (cost-sensitive weighting and SMOTE oversampling), evaluated using 5-Fold Stratified Cross-Validation and a held-out test set:

| Model Pipeline | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1-Score | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.8445** | **0.8426** | **0.6758** | **0.5294** | **0.5937** | **0.6384** |
| Logistic Regression (Baseline) | 0.8461 | 0.8420 | 0.6572 | 0.5588 | 0.6040 | 0.6401 |
| Logistic Regression (Balanced) | 0.8460 | 0.8415 | 0.5043 | 0.7834 | 0.6136 | 0.6375 |
| Random Forest (Balanced) | 0.8445 | 0.8402 | 0.5248 | 0.7647 | 0.6224 | 0.6340 |
| Logistic Regression (SMOTE) | 0.8448 | 0.8399 | 0.5034 | 0.7968 | 0.6170 | 0.6364 |
| HistGradientBoosting | 0.8378 | 0.8350 | 0.6355 | 0.5267 | 0.5760 | 0.6318 |

### Key Modeling Insights:
1. **Calibrated Ranking**: Random Forest and Logistic Regression achieve comparable test ROC-AUC (~0.842–0.843), demonstrating strong discriminatory capability across customer risk tiers.
2. **Precision-Recall Tradeoff**: Setting `class_weight='balanced'` or using SMOTE shifts the decision boundary, boosting test recall from ~53% to ~79% while lowering precision from ~67% to ~50%. In customer retention, ranking customers by continuous probability into deciles provides superior business flexibility over a rigid 0.5 threshold.

---

## 📈 Decile Lift & Cumulative Gains

Sorting the 1,409 held-out test customers by predicted churn probability demonstrates exceptional prioritization power:

| Decile | Customers | Actual Churners | Decile Churn Rate | Cumulative Churners | Decile Lift | Cumulative Lift | Cumulative Gain (%) | Avg Monthly Charge |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decile 1** | 141 | 109 | **77.30%** | 109 | **2.91x** | **2.91x** | **29.14%** | $80.28 |
| **Decile 2** | 141 | 80 | **56.74%** | 189 | **2.14x** | **2.52x** | **50.53%** | $74.60 |
| **Decile 3** | 141 | 54 | 38.30% | 243 | 1.44x | 2.16x | 64.97% | $74.17 |
| **Decile 4** | 141 | 53 | 37.59% | 296 | 1.42x | 1.98x | 79.14% | $65.13 |
| **Decile 5** | 141 | 27 | 19.15% | 323 | 0.72x | 1.73x | 86.36% | $74.98 |
| **Decile 6** | 140 | 23 | 16.43% | 346 | 0.62x | 1.54x | 92.51% | $71.69 |
| **Decile 7** | 141 | 15 | 10.64% | 361 | 0.40x | 1.38x | 96.52% | $57.72 |
| **Decile 8** | 141 | 8 | 5.67% | 369 | 0.21x | 1.23x | 98.66% | $59.27 |
| **Decile 9** | 141 | 3 | 2.13% | 372 | 0.08x | 1.11x | 99.47% | $50.24 |
| **Decile 10** | 141 | 2 | **1.42%** | 374 | **0.05x** | **1.00x** | **100.0%** | $32.85 |

* **Decile 1**: A customer in Decile 1 is **nearly 3x more likely to churn** than an average customer.
* **Decile 10**: Churn rate is only **1.42%**. Reaching out to Decile 10 accounts would waste budget and risk discount cannibalization.

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute End-to-End Pipeline
```bash
# 1. Load and validate data
python -m src.data_loader

# 2. Run EDA and generate segment statistics
python -m src.eda

# 3. Train benchmark models and cross-validate
python -m src.models

# 4. Generate model explainability (odds ratios & feature importances)
python -m src.explainability

# 5. Compute deciles and simulate retention campaign ROI
python -m src.business_impact
```

### 3. Run Automated Tests
```bash
pytest -v tests/
```
All 11 automated unit and integration tests validate dataset integrity, zero-tenure null handling, stratified split bounds, preprocessor shapes, inference probability bounds, model ROC-AUC benchmark ($\ge 0.80$), and decile lift properties.

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501` to:
- Review executive answers and interactive KPI cards.
- Explore segment-level churn rates and Cramér's V associations.
- Inspect ROC curves, PR curves, confusion matrices, and odds ratios.
- Run the **Retention ROI Simulator** and **Live Individual Customer Scorer**.

---

## 📜 Technical Notes
- **Language**: Python 3.11+
- **Core Frameworks**: `pandas`, `scikit-learn`, `imbalanced-learn`, `scipy`, `streamlit`, `plotly`, `pytest`
- **Reproducibility**: All random seeds (`random_state=42`) and data processing pipelines are pinned for 100% deterministic reproducibility.
