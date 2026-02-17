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
# LOAD DATA
# =========================
darwin_path = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
realtime_path = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"

darwin = pd.read_csv(darwin_path)
realtime = pd.read_csv(realtime_path)

X = darwin.drop(columns=["class"])
y = darwin["class"].map({"P": 1, "H": 0})

# =========================
# REPEATED 10-FOLD CV
# =========================
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=10,
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
# PRINT CV RESULTS
# =========================
print("\n===== Repeated 10-Fold CV (DARWIN – Gaussian NB) =====")
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
# TRAIN FINAL MODEL
# =========================
final_model = GaussianNB()
final_model.fit(X, y)

# =========================
# REAL-TIME PREDICTION
# =========================
X_test = realtime[X.columns]

probs = final_model.predict_proba(X_test)[:, 1]
preds = np.where(probs >= 0.5, "P", "H")

results = pd.DataFrame({
    "Predicted_Class": preds,
    "Prediction_Probability": probs
})

print("\n===== REAL-TIME PREDICTIONS =====")
print(results.head())

results.to_csv("gnb_realtime_predictions.csv", index=False)
print("Predictions saved to gnb_realtime_predictions.csv")
