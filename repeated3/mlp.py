import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# =====================================================
# 1. LOAD DATASETS
# =====================================================
darwin_path = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
realtime_path = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"

darwin = pd.read_csv(darwin_path)
realtime = pd.read_csv(realtime_path)

# Target and features
y = darwin["class"].map({"H": 0, "P": 1})
X = darwin.drop(columns=["class"])

# =====================================================
# 2. ALIGN FEATURES (IMPORTANT)
# =====================================================
common_features = X.columns.intersection(realtime.columns)
X = X[common_features]
realtime = realtime[common_features]

# =====================================================
# 3. MLP MODEL
# =====================================================
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

# =====================================================
# 4. REPEATED 10-FOLD CV
# =====================================================
rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=10,
    random_state=42
)

acc, prec, rec, spec, f1s, aucs = [], [], [], [], [], []
tpr, fpr, kappas, mccs = [], [], [], []

for train_idx, test_idx in rskf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    acc.append(accuracy_score(y_test, y_pred))
    prec.append(precision_score(y_test, y_pred))
    rec.append(recall_score(y_test, y_pred))             # Sensitivity
    spec.append(tn / (tn + fp))                           # Specificity
    f1s.append(f1_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_prob))
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))
    kappas.append(cohen_kappa_score(y_test, y_pred))
    mccs.append(matthews_corrcoef(y_test, y_pred))

# =====================================================
# 5. PRINT CV RESULTS (BASE-LEVEL METRICS)
# =====================================================
print("\n===== Repeated 10-Fold CV (DARWIN – MLP) =====")
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

# =====================================================
# 6. TRAIN FINAL MODEL ON FULL DARWIN DATA
# =====================================================
mlp.fit(X, y)

print(f"\nCommon features used: {len(common_features)}")
print(f"Test samples: {len(realtime)}")

# =====================================================
# 7. REAL-TIME PREDICTION
# =====================================================
probs = mlp.predict_proba(realtime)[:, 1]
preds = np.where(probs >= 0.5, "P", "H")

results = pd.DataFrame({
    "Predicted_Class": preds,
    "Prediction_Probability": probs
})

print("\n===== REAL-TIME PREDICTIONS =====")
print(results.head())

results.to_csv("mlp_realtime_predictions.csv", index=False)
print("Predictions saved to mlp_realtime_predictions.csv")
