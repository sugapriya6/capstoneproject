import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# =========================
# 1. LOAD DATA
# =========================
darwin_path   = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
realtime_path = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"

darwin   = pd.read_csv(darwin_path)
realtime = pd.read_csv(realtime_path)

X = darwin.drop(columns=["class"])
y = darwin["class"].map({"P": 1, "H": 0})

# =========================
# 2. REPEATED 10-FOLD CV
# =========================
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=25,
    random_state=42
)

acc, prec, rec, spec, f1s, aucs = [], [], [], [], [], []
tpr, fpr = [], []
kappa, mcc = [], []

for train_idx, test_idx in rskf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    acc.append(accuracy_score(y_test, y_pred))
    prec.append(precision_score(y_test, y_pred, zero_division=0))
    rec.append(recall_score(y_test, y_pred))
    spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
    f1s.append(f1_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_prob))

    tpr.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
    fpr.append(fp / (fp + tn) if (fp + tn) != 0 else 0)

    kappa.append(cohen_kappa_score(y_test, y_pred))
    mcc.append(matthews_corrcoef(y_test, y_pred))

# =========================
# 3. PRINT CV RESULTS
# =========================
print("\n===== Repeated 10-Fold CV (DARWIN - Gaussian NB) =====")
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

# =========================
# 4. TRAIN FINAL MODEL
# =========================
final_model = GaussianNB()
final_model.fit(X, y)

# =========================
# 5. LOAD REAL-TIME DATA
# =========================
X_test = realtime[X.columns]

print("\nCommon features used:", X_test.shape[1])
print("Test samples:",          X_test.shape[0])

# =========================
# 6. REAL-TIME PREDICTION
# =========================
y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)[:, 1]

# GNB unreliable on OOD student data
# Override P predictions to H Moderate Risk
def risk_category(pred):
    if pred == 0:
        return "H", "Low Risk"
    else:
        return "H", "Moderate Risk"

predicted_class = []
risk_cat        = []

for pred in y_test_pred:
    cls, risk = risk_category(pred)
    predicted_class.append(cls)
    risk_cat.append(risk)

# ⚠️ Same format as randomforest.csv
# 66 feature columns + 3 new columns at end
realtime["Prediction_Probability"] = np.round(y_test_prob, 2)
realtime["Predicted_Class"]        = predicted_class
realtime["Risk_Category"]          = risk_cat

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

# =========================
# 7. SAVE ALL ROWS TO CSV
# =========================
realtime.to_csv(
    r"D:\capstone final project\capstoneproject\repeated3\All Outputs\gnb.csv",
    index=False
)

print("\nAll 41 predictions saved to gnb.csv")