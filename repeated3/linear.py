import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# =====================================================
# 1. LOAD DATA
# =====================================================
train_path = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
test_path  = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"

data      = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X = data.drop(columns=["class"])
y = data["class"].map({"P": 1, "H": 0})

# =====================================================
# 2. CLASSIFIER
# =====================================================
model = LinearDiscriminantAnalysis()

# =====================================================
# 3. REPEATED 10-FOLD CV
# =====================================================
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=25,
    random_state=42
)

acc, prec, rec, spec, f1s, aucs = [], [], [], [], [], []
tpr, fpr, kappa, mcc = [], [], [], []

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
    kappa.append(cohen_kappa_score(y_test, y_pred))
    mcc.append(matthews_corrcoef(y_test, y_pred))

# =====================================================
# 4. PRINT CV RESULTS
# =====================================================
print("\n===== Repeated 10-Fold CV (DARWIN - LDA) =====")
print(f"ACC:   {np.mean(acc):.3f} ± {np.std(acc):.3f}")
print(f"PREC:  {np.mean(prec):.3f} ± {np.std(prec):.3f}")
print(f"REC:   {np.mean(rec):.3f} ± {np.std(rec):.3f}")
print(f"SPEC:  {np.mean(spec):.3f} ± {np.std(spec):.3f}")
print(f"F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"AUC:   {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"TPR:   {np.mean(tpr):.3f} ± {np.std(tpr):.3f}")
print(f"FPR:   {np.mean(fpr):.3f} ± {np.std(fpr):.3f}")
print(f"KAPPA: {np.mean(kappa):.3f} ± {np.std(kappa):.3f}")
print(f"MCC:   {np.mean(mcc):.3f} ± {np.std(mcc):.3f}")

# =====================================================
# 5. TRAIN FINAL MODEL ON FULL DATA
# =====================================================
model.fit(X, y)

# =====================================================
# 6. LOAD REAL-TIME DATA
# =====================================================
test_data = test_data[X.columns]   # align features

print("\nCommon features used:", test_data.shape[1])
print("Test samples:",          test_data.shape[0])

# =====================================================
# 7. REAL-TIME PREDICTION WITH RISK CATEGORY
# =====================================================
y_test_prob = model.predict_proba(test_data)[:, 1]

def risk_category(prob):
    if prob < 0.30:
        return "H", "Low Risk"
    elif prob < 0.60:
        return "H", "Moderate Risk"
    elif prob < 0.80:
        return "P", "High Risk"
    else:
        return "P", "Critical Risk"

results = pd.DataFrame()
results["Prediction_Probability"] = y_test_prob

risk_results = results["Prediction_Probability"].apply(
    lambda p: pd.Series(
        risk_category(p),
        index=["Predicted_Class", "Risk_Category"]
    )
)

results["Predicted_Class"] = risk_results["Predicted_Class"]
results["Risk_Category"]   = risk_results["Risk_Category"]

# Show only 5 rows in console
print("\n===== REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(results[["Predicted_Class",
               "Risk_Category",
               "Prediction_Probability"]].head(5))

# Summary
print("\n===== PREDICTION SUMMARY =====")
print(results["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(results["Risk_Category"].value_counts())

# =====================================================
# 8. SAVE ALL ROWS TO CSV
# =====================================================
results.to_csv(
    r"D:\capstone final project\capstoneproject\repeated3\All Outputs\lda.csv",
    index=False
)

print("\nAll 41 predictions saved to lda.csv")