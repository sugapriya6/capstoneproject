########### Ensemble model using majority voting ###########

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score
)

# ===================== DATA FUNCTIONS =====================

def read_csv(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({'H': 0, 'P': 1})
    return X, y

# ===================== FEATURE SELECTION =====================

def rfe_and_train(X_train, y_train, k, classifier):
    model = classifier(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=k)
    X_sel = rfe.fit_transform(X_train, y_train)
    model.fit(X_sel, y_train)
    return model, rfe

def selectkbest_and_train(X_train, y_train, k, classifier):
    skb = SelectKBest(f_classif, k=k)
    X_sel = skb.fit_transform(X_train, y_train)
    model = classifier()
    model.fit(X_sel, y_train)
    return model, skb

# ===================== ENSEMBLE =====================

def combine_predictions(predictions):
    return np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
    )

def evaluate_model(model, selector, X_test):
    return model.predict(selector.transform(X_test))

# ===================== METRICS =====================

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

# ===================== MAIN =====================

def main():

    # -------- DATA --------
    file_path = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
    df = read_csv(file_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- TRAIN MODELS --------
    model_et, rfe_et   = rfe_and_train(X_train, y_train, 30, ExtraTreesClassifier)
    model_rf, rfe_rf   = rfe_and_train(X_train, y_train, 10, RandomForestClassifier)
    model_lr, rfe_lr   = rfe_and_train(X_train, y_train, 10, LogisticRegression)
    model_xgb, rfe_xgb = rfe_and_train(X_train, y_train, 20, XGBClassifier)

    model_gnb, skb_gnb = selectkbest_and_train(X_train, y_train, 20, GaussianNB)
    model_svm, skb_svm = selectkbest_and_train(X_train, y_train, 40, SVC)
    model_mlp, skb_mlp = selectkbest_and_train(X_train, y_train, 60, MLPClassifier)

    # -------- ENSEMBLE PREDICTION --------
    preds = [
        evaluate_model(model_et, rfe_et, X_test),
        evaluate_model(model_rf, rfe_rf, X_test),
        evaluate_model(model_lr, rfe_lr, X_test),
        evaluate_model(model_xgb, rfe_xgb, X_test),
        evaluate_model(model_gnb, skb_gnb, X_test),
        evaluate_model(model_svm, skb_svm, X_test),
        evaluate_model(model_mlp, skb_mlp, X_test)
    ]

    combined_preds = combine_predictions(preds)

    # -------- METRICS --------
    acc, sen, spe, pre, f1, mcc, kappa, auc = calculate_metrics(y_test, combined_preds)

    print(f"Accuracy      : {acc:.4f}")
    print(f"Sensitivity   : {sen:.4f}")
    print(f"Specificity   : {spe:.4f}")
    print(f"Precision     : {pre:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"MCC           : {mcc:.4f}")
    print(f"Kappa         : {kappa:.4f}")
    print(f"AUC-ROC       : {auc:.4f}")

    # -------- SAVE MODELS --------
    MODEL_DIR = "C:/capstone/alzheimers/proposed/ensemble6/voting/"
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model_et, MODEL_DIR + "et_model.joblib")
    joblib.dump(rfe_et, MODEL_DIR + "et_selector.joblib")
    joblib.dump(X.columns[rfe_et.support_], MODEL_DIR + "et_features.joblib")

    joblib.dump(model_rf, MODEL_DIR + "rf_model.joblib")
    joblib.dump(rfe_rf, MODEL_DIR + "rf_selector.joblib")
    joblib.dump(X.columns[rfe_rf.support_], MODEL_DIR + "rf_features.joblib")

    joblib.dump(model_lr, MODEL_DIR + "lr_model.joblib")
    joblib.dump(rfe_lr, MODEL_DIR + "lr_selector.joblib")
    joblib.dump(X.columns[rfe_lr.support_], MODEL_DIR + "lr_features.joblib")

    joblib.dump(model_xgb, MODEL_DIR + "xgb_model.joblib")
    joblib.dump(rfe_xgb, MODEL_DIR + "xgb_selector.joblib")
    joblib.dump(X.columns[rfe_xgb.support_], MODEL_DIR + "xgb_features.joblib")

    joblib.dump(model_gnb, MODEL_DIR + "gnb_model.joblib")
    joblib.dump(skb_gnb, MODEL_DIR + "gnb_selector.joblib")
    joblib.dump(X.columns[skb_gnb.get_support()], MODEL_DIR + "gnb_features.joblib")

    joblib.dump(model_svm, MODEL_DIR + "svm_model.joblib")
    joblib.dump(skb_svm, MODEL_DIR + "svm_selector.joblib")
    joblib.dump(X.columns[skb_svm.get_support()], MODEL_DIR + "svm_features.joblib")

    joblib.dump(model_mlp, MODEL_DIR + "mlp_model.joblib")
    joblib.dump(skb_mlp, MODEL_DIR + "mlp_selector.joblib")
    joblib.dump(X.columns[skb_mlp.get_support()], MODEL_DIR + "mlp_features.joblib")

    print("✅ All ensemble models saved successfully")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
