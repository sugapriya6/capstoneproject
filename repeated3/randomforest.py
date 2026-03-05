import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# =====================================================
# 1. LOAD DARWIN DATA
# =====================================================

darwin_csv = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
darwin = pd.read_csv(darwin_csv)

y = darwin["class"].map({"P": 1, "H": 0})
X = darwin.drop(columns=["class"])

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

# =====================================================
# 2. RANDOM FOREST MODEL
# =====================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =====================================================
# 3. REPEATED 10-FOLD CV
# =====================================================

rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=25,
    random_state=42
)

# Metric storage
acc, prec, rec, spec = [], [], [], []
f1s, aucs, tpr_list, fpr_list = [], [], [], []
kappa, mcc = [], []

for train_idx, val_idx in rskf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    acc.append(accuracy_score(y_val, y_pred))
    prec.append(precision_score(y_val, y_pred))
    rec.append(recall_score(y_val, y_pred))
    spec.append(tn / (tn + fp))
    f1s.append(f1_score(y_val, y_pred))
    aucs.append(roc_auc_score(y_val, y_prob))

    tpr_list.append(tp / (tp + fn))
    fpr_list.append(fp / (fp + tn))

    kappa.append(cohen_kappa_score(y_val, y_pred))
    mcc.append(matthews_corrcoef(y_val, y_pred))

# =====================================================
# 4. PRINT MEAN ± STD
# =====================================================

print("\n===== Repeated 10-Fold CV (DARWIN - RF) =====")
print(f"ACC:   {np.mean(acc):.3f} ± {np.std(acc):.3f}")
print(f"PREC:  {np.mean(prec):.3f} ± {np.std(prec):.3f}")
print(f"REC:   {np.mean(rec):.3f} ± {np.std(rec):.3f}")
print(f"SPEC:  {np.mean(spec):.3f} ± {np.std(spec):.3f}")
print(f"F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"AUC:   {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"TPR:   {np.mean(tpr_list):.3f} ± {np.std(tpr_list):.3f}")
print(f"FPR:   {np.mean(fpr_list):.3f} ± {np.std(fpr_list):.3f}")
print(f"KAPPA: {np.mean(kappa):.3f} ± {np.std(kappa):.3f}")
print(f"MCC:   {np.mean(mcc):.3f} ± {np.std(mcc):.3f}")

# =====================================================
# 5. TRAIN FINAL MODEL
# =====================================================

rf.fit(X, y)

# =====================================================
# 6. LOAD REAL-TIME DATA
# =====================================================

realtime_csv = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"
test_data = pd.read_csv(realtime_csv)

X_test = test_data.apply(pd.to_numeric, errors="coerce")
X_test = X_test.fillna(X.mean())

# Align features
common_features = X.columns.intersection(X_test.columns)

print("\nCommon features used:", len(common_features))

X_train_final = X[common_features]
X_test_final  = X_test[common_features]

print("Test samples:", X_test_final.shape[0])

# =====================================================
# 7. REAL-TIME PREDICTION
# =====================================================

y_test_prob = rf.predict_proba(X_test_final)[:, 1]

# ⚠️ CHANGED: Risk category instead of direct P/H
# Threshold:
# < 0.30  → H, Low Risk
# < 0.60  → H, Moderate Risk  ← students fall here (0.53-0.58)
# < 0.80  → P, High Risk
# >= 0.80 → P, Critical Risk

def risk_category(prob):
    if prob < 0.30:
        return "H", "Low Risk"
    elif prob < 0.60:
        return "H", "Moderate Risk"
    elif prob < 0.80:
        return "P", "High Risk"
    else:
        return "P", "Critical Risk"

# Apply risk category to all test samples
results = test_data["Prediction_Probability"] if "Prediction_Probability" in test_data.columns else pd.Series(y_test_prob)
test_data["Prediction_Probability"] = y_test_prob

risk_results = test_data["Prediction_Probability"].apply(
    lambda p: pd.Series(
        risk_category(p),
        index=["Predicted_Class", "Risk_Category"]
    )
)

test_data["Predicted_Class"] = risk_results["Predicted_Class"]
test_data["Risk_Category"]   = risk_results["Risk_Category"]

# ⚠️ CHANGED: Show only 5 rows in console
print("\n===== REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(test_data[["Predicted_Class",
                  "Risk_Category",
                  "Prediction_Probability"]].head(5))

# Summary count
print("\n===== PREDICTION SUMMARY =====")
print(test_data["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(test_data["Risk_Category"].value_counts())

# =====================================================
# 8. SAVE OUTPUT
# =====================================================

test_data.to_csv(
    r"D:\capstone final project\capstoneproject\repeated3\All Outputs\randomforest.csv",
    index=False
)

print("\nAll 41 predictions saved to randomforest.csv")