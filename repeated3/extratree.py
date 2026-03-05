import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# =========================
# 1. FILE PATHS
# =========================
darwin_file   = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
realtime_file = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"

# =========================
# 2. LOAD DARWIN DATA
# =========================
data = pd.read_csv(darwin_file)
X = data.drop(columns=["class"])
y = data["class"].map({'P': 1, 'H': 0})

# =========================
# 3. EXTRA TREES MODEL
# class_weight removed — DARWIN is balanced (51%/49%)
# =========================
model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42
)

# =========================
# 4. REPEATED 10-FOLD CV
# =========================
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=25,
    random_state=42
)

acc, prec, rec, spec, f1s = [], [], [], [], []
aucs, tpr, fpr, kappas, mccs = [], [], [], [], []

for train_idx, test_idx in rskf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    acc.append(accuracy_score(y_test, y_pred))
    prec.append(precision_score(y_test, y_pred))
    rec.append(recall_score(y_test, y_pred))
    spec.append(tn / (tn + fp))
    f1s.append(f1_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_prob))
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))
    kappas.append(cohen_kappa_score(y_test, y_pred))
    mccs.append(matthews_corrcoef(y_test, y_pred))

# =========================
# 5. PRINT CV RESULTS
# =========================
print("\n===== Repeated 10-Fold CV (DARWIN - Extra Trees) =====")
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

# =========================
# 6. TRAIN FINAL MODEL
# =========================
model.fit(X, y)

# =========================
# 7. LOAD REAL-TIME DATA
# =========================
test_data = pd.read_csv(realtime_file)
X_test    = test_data[X.columns]

print("\nCommon features used:", X_test.shape[1])
print("Test samples:",          X_test.shape[0])

# =========================
# 8. REAL-TIME PREDICTION WITH RISK CATEGORY
# Same threshold as all other classifiers
# =========================
y_test_prob = model.predict_proba(X_test)[:, 1]

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
test_data["Prediction_Probability"] = np.round(y_test_prob, 2)
test_data["Predicted_Class"]        = risk_results["Predicted_Class"].values
test_data["Risk_Category"]          = risk_results["Risk_Category"].values

# Show only 5 rows in console
print("\n===== REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(test_data[["Predicted_Class",
                  "Risk_Category",
                  "Prediction_Probability"]].head(5))

# Summary
print("\n===== PREDICTION SUMMARY =====")
print(test_data["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(test_data["Risk_Category"].value_counts())

# =========================
# 9. SAVE ALL ROWS TO CSV
# =========================
test_data.to_csv(
    r"D:\capstone final project\capstoneproject\repeated3\All Outputs\extratrees.csv",
    index=False
)

print("\nAll 41 predictions saved to extratrees.csv")