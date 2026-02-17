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

darwin_csv = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
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
    rec.append(recall_score(y_val, y_pred))          # Sensitivity
    spec.append(tn / (tn + fp))                      # Specificity
    f1s.append(f1_score(y_val, y_pred))
    aucs.append(roc_auc_score(y_val, y_prob))

    tpr_list.append(tp / (tp + fn))                  # TPR
    fpr_list.append(fp / (fp + tn))                  # FPR

    kappa.append(cohen_kappa_score(y_val, y_pred))
    mcc.append(matthews_corrcoef(y_val, y_pred))

# =====================================================
# 4. PRINT MEAN ± STD (SOURCE STYLE)
# =====================================================

print("\n===== Repeated 10-Fold CV (DARWIN – RF) =====")
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

realtime_csv = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
test_data = pd.read_csv(realtime_csv)

X_test = test_data.apply(pd.to_numeric, errors="coerce")
X_test = X_test.fillna(X.mean())

# Align features
common_features = X.columns.intersection(X_test.columns)

print("\nCommon features used:", len(common_features))

X_train_final = X[common_features]
X_test_final = X_test[common_features]

print("Test samples:", X_test_final.shape[0])

# =====================================================
# 7. REAL-TIME PREDICTION
# =====================================================

y_test_pred = rf.predict(X_test_final)
y_test_prob = rf.predict_proba(X_test_final)[:, 1]

test_data["Predicted_Class"] = np.where(y_test_pred == 1, "P", "H")
test_data["Prediction_Probability"] = y_test_prob

print("\n===== REAL-TIME PREDICTIONS =====")
print(test_data[["Predicted_Class", "Prediction_Probability"]].head())

test_data.to_csv(
    "C:/capstone/alzheimers/proposed/randomforest.csv",
    index=False
)

print("\nPredictions saved to randomforest.csv")
