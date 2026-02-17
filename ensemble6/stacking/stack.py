import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score
)

# =============================
# READ DATA
# =============================
def read_csv(file_path):
    return pd.read_csv(file_path)

# =============================
# PREPROCESS
# =============================
def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({'H': 0, 'P': 1})
    return X, y

# =============================
# METRICS
# =============================
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    return acc, sensitivity, specificity, precision, f1, mcc, kappa, auc_roc

# =============================
# MAIN FUNCTION
# =============================
def main():

    DATA_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
    MODEL_DIR = "C:/capstone/alzheimers/proposed/ensemble6/stacking/"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    df = read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================
    # FEATURE SELECTION (RFE)
    # =============================
    k = 30
    rfe = RFE(estimator=ExtraTreesClassifier(random_state=42), n_features_to_select=k)

    X_train_sel = rfe.fit_transform(X_train, y_train)
    X_test_sel = rfe.transform(X_test)

    # =============================
    # BASE MODELS
    # =============================
    base_models = [
        ('et', ExtraTreesClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]

    # Meta model
    meta_model = LogisticRegression(random_state=42)

    # =============================
    # STACKING MODEL
    # =============================
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    stacking_model.fit(X_train_sel, y_train)

    # =============================
    # TESTING
    # =============================
    y_pred = stacking_model.predict(X_test_sel)

    acc, sen, spe, pre, f1, mcc, kappa, auc = calculate_metrics(y_test, y_pred)

    print("\n===== Stacking Results =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Sensitivity  : {sen:.4f}")
    print(f"Specificity  : {spe:.4f}")
    print(f"Precision    : {pre:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"MCC          : {mcc:.4f}")
    print(f"Kappa        : {kappa:.4f}")
    print(f"AUC          : {auc:.4f}")

    # =============================
    # SAVE MODEL
    # =============================
    joblib.dump(stacking_model, MODEL_DIR + "stacking_model.joblib")
    joblib.dump(rfe, MODEL_DIR + "stacking_selector.joblib")
    joblib.dump(X.columns[rfe.support_], MODEL_DIR + "stacking_features.joblib")

    print("\n✅ Stacking model saved successfully")

if __name__ == "__main__":
    main()
