import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# =============================
# 1. LOAD DATASETS
# =============================
darwin_path   = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
realtime_path = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"

darwin   = pd.read_csv(darwin_path)
realtime = pd.read_csv(realtime_path)

# =============================
# 2. SPLIT FEATURES & LABEL
# =============================
X = darwin.drop(columns=["class"])
y = darwin["class"].map({"H": 0, "P": 1})

# =============================
# 3. FEATURE ALIGNMENT
# =============================
common_features = X.columns.intersection(realtime.columns)
X       = X[common_features]
X_test  = realtime[common_features]

# =============================
# 4. SVM MODEL
# =============================
svm = SVC(
    kernel="linear",
    probability=True,   # required for predict_proba
    random_state=42
)

# =============================
# 5. REPEATED STRATIFIED CV
# =============================
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=25,       # ⚠️ CHANGED: 10 → 25 to match paper
    random_state=42
)

acc, prec, rec, spec, f1s, aucs = [], [], [], [], [], []
tpr, fpr = [], []
kappas, mccs = [], []

for train_idx, test_idx in rskf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_val)
    y_prob = svm.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    acc.append(accuracy_score(y_val, y_pred))
    prec.append(precision_score(y_val, y_pred))
    rec.append(recall_score(y_val, y_pred))
    spec.append(tn / (tn + fp))
    f1s.append(f1_score(y_val, y_pred))
    aucs.append(roc_auc_score(y_val, y_prob))
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))
    kappas.append(cohen_kappa_score(y_val, y_pred))
    mccs.append(matthews_corrcoef(y_val, y_pred))

# =============================
# 6. PRINT CV RESULTS
# =============================
print("\n===== Repeated 10-Fold CV (DARWIN - SVM) =====")
print(f"ACC:   {np.mean(acc):.3f} ± {np.std(acc):.3f}")
print(f"PREC:  {np.mean(prec):.3f} ± {np.std(prec):.3f}")
print(f"REC:   {np.mean(rec):.3f} ± {np.std(rec):.3f}")
print(f"SPEC:  {np.mean(spec):.3f} ± {np.std(spec):.3f}")
print(f"F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"AUC:   {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"TPR:   {np.mean(tpr):.3f} ± {np.std(tpr):.3f}")
print(f"FPR:   {np.mean(fpr):.3f} ± {np.std(fpr):.3f}")
print(f"KAPPA: {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
print(f"MCC:   {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")

# =============================
# 7. TRAIN FINAL MODEL
# =============================
svm.fit(X, y)

print("\nCommon features used:", len(common_features))
print("Test samples:",          len(X_test))

# =============================
# 8. REAL-TIME PREDICTION WITH RISK CATEGORY
# =============================
y_test_prob = svm.predict_proba(X_test)[:, 1]

def risk_category(prob):
    if prob < 0.30:
        return "H", "Low Risk"
    elif prob < 0.60:
        return "H", "Moderate Risk"
    elif prob < 0.80:
        return "P", "High Risk"
    else:
        return "P", "Critical Risk"

risk_results = pd.Series(y_test_prob).apply(
    lambda p: pd.Series(
        risk_category(p),
        index=["Predicted_Class", "Risk_Category"]
    )
)

# Same format as randomforest.csv
realtime["Prediction_Probability"] = np.round(y_test_prob, 2)
realtime["Predicted_Class"]        = risk_results["Predicted_Class"].values
realtime["Risk_Category"]          = risk_results["Risk_Category"].values

# Show only 5 rows in console
print("\n===== REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(realtime[["Predicted_Class",
                "Risk_Category",
                "Prediction_Probability"]].head(5))

# Summary
print("\n===== PREDICTION SUMMARY =====")
print(realtime["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(realtime["Risk_Category"].value_counts())

# =============================
# 9. SAVE ALL ROWS TO CSV
# =============================
realtime.to_csv(
    r"D:\capstone final project\capstoneproject\repeated3\All Outputs\svm.csv",
    index=False
)

print("\nAll 41 predictions saved to svm.csv")