import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

import kagglehub

 
def load_patient_data(patient_data_path: str) -> pd.DataFrame:
    records = []
 
    for file in os.listdir(patient_data_path):
        if not file.endswith(".txt"):
            continue
 
        file_path = os.path.join(patient_data_path, file)
        df = pd.read_csv(file_path)
        df.columns = ["Time", "Parameter", "Value"]
 
        df["Value"] = df["Value"].replace(-1.0, float("nan"))
 
        record_id = int(file.split(".")[0])
 
        features = df.groupby("Parameter")["Value"].mean()
        features["RecordID"] = record_id
        records.append(features)
 
    return pd.DataFrame(records)
 
 
def load_outcomes(outcomes_file: str) -> pd.DataFrame:
    outcomes_df = pd.read_csv(outcomes_file)
    return outcomes_df[["RecordID", "In-hospital_death"]]
 
 
def build_dataset(patient_data_path: str, outcomes_file: str) -> pd.DataFrame:
    data = load_patient_data(patient_data_path)
    outcomes = load_outcomes(outcomes_file)
    dataset = pd.merge(data, outcomes, on="RecordID")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Class distribution:\n{dataset['In-hospital_death'].value_counts()}\n")
    return dataset
 
 
def preprocess(X_train, X_test):
    """Fit only with train data to avoid data leakage. """
 
    # Impute — fit on train, transform both
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
 
    # Scale — fit on train, transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
 
    return X_train, X_test
 
def build_models(class_ratio: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=class_ratio,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }
 
 
def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> dict:
    results = {}
 
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
 
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
 
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Survived", "Died"]))
 
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": acc,
            "roc_auc": auc,
        }
 
    return results
 
 
# ============================================================
# 4. EVALUATION PLOTS
# ============================================================
 
def plot_roc_curves(results: dict, y_test):
    """Plot ROC curves for all models on one chart."""
    plt.figure(figsize=(8, 6))
 
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
 
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — ICU Mortality Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
 
 
def plot_confusion_matrices(results: dict, X_test, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
 
    for ax, (name, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_estimator(
            res["model"],
            X_test,
            y_test,
            display_labels=["Survived", "Died"],
            ax=ax,
            colorbar=False,
        )
        ax.set_title(name)
 
    plt.suptitle("Confusion Matrices — ICU Mortality Prediction", y=1.02)
    plt.tight_layout()
    plt.show()
 
 
def plot_shap(best_model, X_train, X_test, feature_names):
    """Plot SHAP summary for the best model."""
    print("\nGenerating SHAP explainability plot...")
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)
 
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=True,
    )
 
 
def main():

    path = kagglehub.dataset_download("msafi04/predict-mortality-of-icu-patients-physionet")

    outcomes_file = os.path.join(path,"Outcomes-a.txt")

    patient_data_path = os.path.join(path, "set-a","set-a") 
 
    dataset = build_dataset(patient_data_path, outcomes_file)
 
    X = dataset.drop(columns=["RecordID", "In-hospital_death"])
    y = dataset["In-hospital_death"]
    feature_names = X.columns.tolist()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    X_train, X_test = preprocess(X_train, X_test)
 
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class ratio (survived/died): {class_ratio:.2f}")
 
    models = build_models(class_ratio)
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
 
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_name} ({results[best_name]['roc_auc']:.4f})")
 
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, X_test, y_test)
 
    # --- SHAP on best model ---
    plot_shap(results[best_name]["model"], X_train, X_test, feature_names)
    # --- we expect it to be xgboost as the ensemble models can handle non-linear relations better than a liner model like Logistic Regression
 
 
if __name__ == "__main__":
    main()
