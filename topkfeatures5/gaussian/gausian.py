import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, f1_score,
    matthews_corrcoef, cohen_kappa_score
)

# ==============================
# 1. PATHS
# ==============================
TRAIN_PATH = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
MODEL_DIR  = r"D:\capstone final project\capstoneproject\topkfeatures\gaussian"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv(TRAIN_PATH)
X  = df.drop(columns=["class"])
y  = df["class"].map({"H": 0, "P": 1})

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ==============================
# 3. SPLIT FIRST — THEN ANOVA
# GNB uses SelectKBest (ANOVA)
# NOT RFE — GNB has no feature_importances_
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ==============================
# 4. ANOVA ACROSS K VALUES
# ==============================
min_k = 10
max_k = X.shape[1]   # 66
step  = 10

k_values           = []
accuracy_values    = []
sensitivity_values = []
specificity_values = []
precision_values   = []
tpr_values         = []
fpr_values         = []
f1_values          = []
mcc_values         = []
cohen_kappa_values = []
auc_roc_values     = []

print(f"\nRunning ANOVA SelectKBest for k = {min_k} to {max_k}, step {step}...")

for k in range(min_k, max_k + 1, step):

    # ANOVA fit on TRAIN only
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)

    # Train GNB on TRAIN
    gnb = GaussianNB()
    gnb.fit(X_train_sel, y_train)

    # Evaluate on unseen TEST
    X_test_sel = selector.transform(X_test)
    y_pred = gnb.predict(X_test_sel)
    y_prob = gnb.predict_proba(X_test_sel)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    k_values.append(k)
    accuracy_values.append(accuracy_score(y_test, y_pred))
    sensitivity_values.append(recall_score(y_test, y_pred))
    specificity_values.append(tn / (tn + fp))
    precision_values.append(precision_score(y_test, y_pred, zero_division=0))
    tpr_values.append(tp / (tp + fn))
    fpr_values.append(fp / (fp + tn))
    f1_values.append(f1_score(y_test, y_pred))
    mcc_values.append(matthews_corrcoef(y_test, y_pred))
    cohen_kappa_values.append(cohen_kappa_score(y_test, y_pred))
    auc_roc_values.append(roc_auc_score(y_test, y_prob))

    print(f"  k={k:3d} | ACC={accuracy_values[-1]:.3f} | MCC={mcc_values[-1]:.3f} | AUC={auc_roc_values[-1]:.3f}")

# ==============================
# 5. FIND BEST K
# ==============================
best_acc_idx   = np.argmax(accuracy_values)
best_sens_idx  = np.argmax(sensitivity_values)
best_spec_idx  = np.argmax(specificity_values)
best_prec_idx  = np.argmax(precision_values)
best_f1_idx    = np.argmax(f1_values)
best_mcc_idx   = np.argmax(mcc_values)
best_kappa_idx = np.argmax(cohen_kappa_values)
best_auc_idx   = np.argmax(auc_roc_values)
best_fpr_idx   = np.argmin(fpr_values)

print("\n===== BEST K VALUES =====")
print(f"Best Accuracy    k={k_values[best_acc_idx]:3d} → {accuracy_values[best_acc_idx]:.4f}")
print(f"Best Sensitivity k={k_values[best_sens_idx]:3d} → {sensitivity_values[best_sens_idx]:.4f}")
print(f"Best Specificity k={k_values[best_spec_idx]:3d} → {specificity_values[best_spec_idx]:.4f}")
print(f"Best Precision   k={k_values[best_prec_idx]:3d} → {precision_values[best_prec_idx]:.4f}")
print(f"Best F1          k={k_values[best_f1_idx]:3d} → {f1_values[best_f1_idx]:.4f}")
print(f"Best MCC         k={k_values[best_mcc_idx]:3d} → {mcc_values[best_mcc_idx]:.4f}")
print(f"Best AUC         k={k_values[best_auc_idx]:3d} → {auc_roc_values[best_auc_idx]:.4f}")

# ==============================
# 6. PLOT METRICS VS K
# ==============================
plt.figure(figsize=(14, 8))
plt.plot(k_values, accuracy_values,    label="Accuracy",    linewidth=2)
plt.plot(k_values, sensitivity_values, label="Sensitivity", linewidth=2)
plt.plot(k_values, specificity_values, label="Specificity", linewidth=2)
plt.plot(k_values, precision_values,   label="Precision",   linewidth=2)
plt.plot(k_values, f1_values,          label="F1 Score",    linewidth=2)
plt.plot(k_values, mcc_values,         label="MCC",         linewidth=2)
plt.plot(k_values, auc_roc_values,     label="AUC-ROC",     linewidth=2)
plt.plot(k_values, fpr_values,         label="FPR",         linewidth=2, linestyle="--")

plt.scatter(k_values[best_acc_idx], accuracy_values[best_acc_idx],
            marker="o", s=100, color="orange", zorder=5,
            label=f"Best ACC k={k_values[best_acc_idx]}")
plt.scatter(k_values[best_mcc_idx], mcc_values[best_mcc_idx],
            marker="o", s=100, color="pink", zorder=5,
            label=f"Best MCC k={k_values[best_mcc_idx]}")
plt.scatter(k_values[best_auc_idx], auc_roc_values[best_auc_idx],
            marker="o", s=100, color="gray", zorder=5,
            label=f"Best AUC k={k_values[best_auc_idx]}")

plt.title("GNB — Performance Metrics vs Top-K Features (ANOVA)")
plt.xlabel("Number of Features (k)")
plt.ylabel("Metric Value")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(MODEL_DIR, "gnb_topk_metrics.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nPlot saved: {plot_path}")

# ==============================
# 7. SAVE FINAL MODEL
#    Best k by MCC
#    Retrained on FULL DARWIN
# ==============================
best_k = k_values[best_mcc_idx]
print(f"\nSaving final GNB model with best k = {best_k}")

# Refit ANOVA on FULL data
final_selector = SelectKBest(score_func=f_classif, k=best_k)
X_final = final_selector.fit_transform(X, y)

# Train final GNB on FULL data
final_gnb = GaussianNB()
final_gnb.fit(X_final, y)

joblib.dump(final_gnb,                             os.path.join(MODEL_DIR, "gnb_model.joblib"))
joblib.dump(final_selector,                        os.path.join(MODEL_DIR, "gnb_selector.joblib"))
joblib.dump(X.columns[final_selector.get_support()], os.path.join(MODEL_DIR, "gnb_features.joblib"))

print(f"gnb_model.joblib    saved")
print(f"gnb_selector.joblib saved")
print(f"gnb_features.joblib saved")
print(f"\nTop {best_k} features selected from {X.shape[1]} total")
print("Selected features:")
print(list(X.columns[final_selector.get_support()]))