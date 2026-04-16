# ICU Mortality Prediction using Machine Learning

This project predicts **in-hospital mortality for ICU patients** using structured clinical data extracted from the PhysioNet dataset. Multiple machine learning models are trained and compared, including Logistic Regression and Gradient Boosting (XGBoost), with interpretability provided via SHAP.

---

## 📊 Project Overview

Early prediction of patient mortality in Intensive Care Units (ICU) can support clinical decision-making and improve patient outcomes.

This project:
- Builds a structured dataset from raw ICU time-series records
- Trains multiple ML models for binary classification (survived vs died)
- Evaluates models using accuracy and ROC-AUC
- Handles missing data and feature scaling properly
- Provides model interpretability using SHAP

---

## 📁 Project Structure


```bash
└── icu_mortality/
    ├── __init__.py
    └── main.py

pyproject.toml
 ```

---

## 🧠 Models Used

- Logistic Regression (baseline, interpretable)
- XGBoost (gradient boosting, high performance)

---

## 📈 Evaluation Metrics

- Accuracy
- ROC-AUC
- Classification Report (precision, recall, F1-score)
- Confusion Matrix
- ROC Curves

---

## 🔍 Explainability

The best-performing model is analyzed using **SHAP (SHapley Additive Explanations)** to identify which clinical features most influence mortality prediction.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AlkisChatz/icu-mortality-prediction.git
cd icu-mortality-prediction
 ```

## Create environment and install dependencies:

```bash
pip install -e .
 ```

## Run the full pipeline:

```bash
python -m icu_mortality.main
 ```
