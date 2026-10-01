<p align="center">
  <h1 align="center">💰 Profit-Aware Customer Retention</h1>
  <p align="center">
    <strong>Calibrated Churn Prediction with Asymmetric Cost Optimization</strong>
  </p>
  <p align="center">
    <a href="#abstract">Abstract</a> •
    <a href="#key-findings">Key Findings</a> •
    <a href="#methodology">Methodology</a> •
    <a href="#results">Results</a> •
    <a href="#usage">Usage</a> •
    <a href="#references">References</a>
  </p>
</p>

---

## Abstract

Standard churn prediction frameworks optimize **symmetric metrics** (accuracy, F1-score) that treat false positives and false negatives equally. In real-world customer retention, these errors carry **vastly different costs**: missing a churner loses their lifetime value, while unnecessarily contacting a loyal customer wastes campaign budget.

This project demonstrates a complete pipeline that bridges the gap between **statistical performance** and **business profitability** by:

1. Training and tuning **4 classifiers** (Logistic Regression, Random Forest, XGBoost, LightGBM) with proper cross-validation
2. Applying **post-hoc probability calibration** (Platt Scaling & Isotonic Regression) on a dedicated held-out calibration set
3. Defining an **explicit profit function** that encodes the asymmetric cost structure
4. Deriving and empirically validating the **theoretically optimal decision threshold**
5. Proving that **calibration quality — not discrimination (ROC-AUC) — governs profitability**

## Economic Framework

The retention decision for each customer is modeled as a cost-benefit analysis:

| Symbol | Meaning | Default Value |
|--------|---------|---------------|
| V | Customer Lifetime Value | $500 |
| C | Cost of intervention (offer, call, etc.) | $50 |
| s | Probability of successful rescue given contact | 0.30 |
| p* | Theoretical optimal threshold = C / (s·V) | **0.333** |

**Profit function:**

```
Π(t) = s · V · TP(t) − C · (TP(t) + FP(t))
```

For a **perfectly calibrated** model, the expected profit of contacting a customer with predicted churn probability `p` is:

```
E[profit] = p · s · V − C
```

Setting `E[profit] ≥ 0` yields the closed-form optimal threshold:

```
p* = C / (s · V) = 50 / (0.30 × 500) = 0.333
```

---

## Key Findings

### 1. Calibration Dramatically Improves Probability Quality
Tree-based models (Random Forest, XGBoost, LightGBM) exhibit **sigmoid-shaped distortions** in their uncalibrated outputs — probabilities are systematically pushed toward 0 and 1. Isotonic Regression and Platt Scaling significantly reduce Brier Loss and Expected Calibration Error (ECE).

### 2. Calibration Aligns the Optimal Threshold with Theory
Uncalibrated models require **radically different, unintuitive thresholds** (e.g., 0.15 or 0.60) to maximize profit. After calibration, the empirical optimal threshold converges to the **theoretically derived** `p* ≈ 0.333`, making deployment decisions interpretable and robust.

### 3. ROC-AUC ≠ Profitability
Models with **identical ROC-AUC** can yield **entirely different profits**. ROC-AUC measures ranking quality (discrimination) but is **invariant to calibration**. Profit depends on calibrated probabilities matching reality — captured by **Brier Loss**, not ROC-AUC.

### 4. F1-Optimal ≠ Profit-Optimal
F1-score treats false positives and false negatives symmetrically (harmonic mean of precision and recall). The business cost structure is asymmetric (3:1 ratio of lost value to wasted cost), making the F1-optimal threshold suboptimal for profit.

---

## Methodology

### Pipeline Architecture

```
Raw Data
  │
  ├── Preprocessing (ColumnTransformer)
  │     ├── StandardScaler (numeric features)
  │     └── OneHotEncoder (categorical features)
  │
  ├── 60/20/20 Stratified Split
  │     ├── Train Set (60%) ──────► Model Training + Hyperparameter Tuning
  │     ├── Calibration Set (20%) ► Post-Hoc Calibration (Platt / Isotonic)
  │     └── Test Set (20%) ───────► Final Evaluation (never seen before)
  │
  ├── Models
  │     ├── Logistic Regression
  │     ├── Random Forest
  │     ├── XGBoost
  │     └── LightGBM
  │
  ├── Calibration (per model)
  │     ├── Uncalibrated
  │     ├── Platt Scaling (Sigmoid)
  │     └── Isotonic Regression
  │
  └── Evaluation
        ├── ROC-AUC, PR-AUC, Brier Loss
        ├── Profit-Optimal vs F1-Optimal Threshold
        ├── Sensitivity Analysis (Cost sweep)
        ├── Bootstrap Confidence Intervals
        └── SHAP Explainability
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **3-way split** (Train/Calib/Test) | Prevents calibration overfitting — the calibrator never sees training data |
| **Pipeline-based preprocessing** | `ColumnTransformer` inside `Pipeline` prevents data leakage |
| **RandomizedSearchCV** | Efficient hyperparameter search with 5-fold stratified CV |
| **Profit function as primary metric** | Directly optimizes business value instead of statistical surrogates |
| **Bootstrap CIs** | Quantifies uncertainty in profit estimates for stakeholder communication |

---

## Results

### Calibration Curves

The reliability diagrams show how calibration transforms the probability distributions:
- **Before calibration**: Tree-based models show characteristic sigmoid distortions
- **After calibration**: Probabilities align with the diagonal (perfect calibration)

### Sensitivity Analysis

The empirical optimal threshold tracks the theoretical curve `p* = C/(s·V)` closely after calibration, confirming that the model's probabilities are well-calibrated across different cost scenarios.

### SHAP Feature Importance

Top predictive features for churn risk include:
- **Contract type** (month-to-month contracts have highest churn risk)
- **Tenure** (shorter tenure → higher risk)
- **Monthly charges** (higher charges → higher risk)
- **Internet service type** (fiber optic shows higher churn)

> ⚠️ **Caution**: SHAP values explain *predictive risk*, not *causal effect*. High-risk features may not be actionable intervention targets.

---

## Dataset

**IBM Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Features | 20 (demographic, account, service) |
| Target | `Churn` (Yes/No) |
| Churn Rate | ~26.5% |
| Source | IBM Sample Datasets |

The dataset is included in this repository as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

---

## Usage

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap
```

### Running the Notebook

```bash
# Clone the repository
git clone https://github.com/ISHAN12369/Profit-Aware-Customer-Retention.git
cd Profit-Aware-Customer-Retention

# Launch Jupyter
jupyter notebook churn_prediction.ipynb
```

Or open directly in **Google Colab**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ISHAN12369/Profit-Aware-Customer-Retention/blob/main/churn_prediction.ipynb)

---

## Project Structure

```
Profit-Aware-Customer-Retention/
├── churn_prediction.ipynb                    # Complete analysis notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Dataset
├── README.md                                  # This file
├── .gitignore                                 # Git ignore rules
└── LICENSE                                    # MIT License
```

---

## Limitations

| Limitation | Impact |
|-----------|--------|
| **Static Economics** | Assumes fixed `s=0.30`, `C=$50`, `V=$500`. Real-world values vary by customer segment and treatment type |
| **Prediction ≠ Causation** | Predicts churn *risk*, not rescue *probability*. High-risk customers may be "lost causes" |
| **Single Dataset** | Results specific to IBM Telco data; generalizability not established |
| **No RCT** | Cannot isolate causal effect of intervention from observational correlations |

## Future Directions

- **Uplift Modeling**: Estimate the Conditional Average Treatment Effect (CATE) using RCT data to directly model `P(Churn|Treatment) - P(Churn|Control)`
- **Dynamic CLV**: Replace fixed `V` with survival-model-based lifetime value estimates
- **Multi-Action Optimization**: Optimize across intervention types (discount, loyalty program, personal call)
- **Online Calibration**: Monitor and recalibrate with production data drift detection

---

## References

1. Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
2. Verbeke, W., et al. (2012). *New insights into churn prediction in the telecommunication sector*. European Journal of Operational Research.
3. Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
4. Radcliffe, N. J., & Surry, P. D. (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees*.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
