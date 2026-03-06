import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)

# ===============================
# 1. PATHS
# ===============================
darwin_path = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
output_dir  = r"D:\capstone final project\capstoneproject\mccv"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 2. LOAD DATA
# ===============================
darwin = pd.read_csv(darwin_path)
X = darwin.drop(columns=["class"])
y = darwin["class"].map({"H": 0, "P": 1})

print("Dataset loaded:")
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {y.value_counts().to_dict()}")

# ===============================
# 3. DEFINE ALL 10 CLASSIFIERS
# ===============================
classifiers = {
    "RF":  RandomForestClassifier(n_estimators=200, random_state=42),
    "LR":  LogisticRegression(max_iter=1000, random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "GNB": GaussianNB(),
    "ET":  ExtraTreesClassifier(n_estimators=200, random_state=42),
    "XGB": XGBClassifier(
               n_estimators=200, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8,
               eval_metric="logloss", random_state=42
           ),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "MLP": MLPClassifier(
               hidden_layer_sizes=(100, 50), activation="relu",
               solver="adam", max_iter=1000, random_state=42
           ),
    "DT":  DecisionTreeClassifier(random_state=42),
}

# ===============================
# 4. MCCV SETTINGS
# ===============================
N_ITERATIONS = 100    # Monte Carlo repetitions
TEST_SIZE    = 0.20   # 80% train / 20% test

# ===============================
# 5. RUN MCCV FOR ALL CLASSIFIERS
# ===============================
all_results = []

for clf_name, clf in classifiers.items():
    print(f"\n{'='*50}")
    print(f"Running MCCV: {clf_name} ({N_ITERATIONS} iterations)")
    print(f"{'='*50}")

    acc, prec, rec, spec = [], [], [], []
    f1s, aucs, tpr, fpr  = [], [], [], []
    kappas, mccs         = [], []

    for i in range(N_ITERATIONS):
        # Random 80/20 split each iteration
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=i,       # different seed each iteration
            stratify=y
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred, zero_division=0))
        rec.append(recall_score(y_test, y_pred))
        spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        tpr.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) != 0 else 0)
        kappas.append(cohen_kappa_score(y_test, y_pred))
        mccs.append(matthews_corrcoef(y_test, y_pred))

    # Print results
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

    # Store results for CSV
    metric_names = ["Accuracy","Precision","Sensitivity","Specificity",
                    "F1 Score","ROC AUC","TPR","FPR","Cohens Kappa","MCC"]
    metric_data  = [acc, prec, rec, spec, f1s, aucs, tpr, fpr, kappas, mccs]

    for metric, values in zip(metric_names, metric_data):
        all_results.append({
            "Classifier": clf_name,
            "Metric":     metric,
            "Mean":       round(np.mean(values), 4),
            "Std":        round(np.std(values),  4),
            "CV_Ratio":   round(np.std(values) / np.mean(values), 4)
                          if np.mean(values) != 0 else 0
        })

# ===============================
# 6. SAVE FULL RESULTS TO CSV
# ===============================
df_results = pd.DataFrame(all_results)

csv_path = os.path.join(output_dir, "All_Classifiers_Mean_Std_MCCV.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n✅ MCCV results saved to: {csv_path}")

# ===============================
# 7. PRINT SUMMARY RANKING TABLE
# ===============================
print("\n" + "="*65)
print("   MCCV SUMMARY RANKING TABLE (by Accuracy)")
print("="*65)
print(f"{'Rank':<6}{'Classifier':<12}{'ACC':<10}{'AUC':<10}"
      f"{'MCC':<10}{'CV Ratio':<12}{'Selected'}")
print("-"*65)

# Get accuracy row for each classifier
summary = df_results[df_results["Metric"] == "Accuracy"].copy()
auc_df  = df_results[df_results["Metric"] == "ROC AUC"][["Classifier","Mean"]].rename(
              columns={"Mean": "AUC"})
mcc_df  = df_results[df_results["Metric"] == "MCC"][["Classifier","Mean"]].rename(
              columns={"Mean": "MCC_val"})

summary = summary.merge(auc_df, on="Classifier")
summary = summary.merge(mcc_df, on="Classifier")
summary = summary.sort_values("Mean", ascending=False).reset_index(drop=True)

top7 = ["ET", "RF", "GNB", "SVM", "MLP", "LR", "XGB"]

for i, row in summary.iterrows():
    selected = "✅ YES" if row["Classifier"] in top7 else "❌ NO"
    print(f"{i+1:<6}{row['Classifier']:<12}{row['Mean']:<10}"
          f"{row['AUC']:<10}{row['MCC_val']:<10}"
          f"{row['CV_Ratio']:<12}{selected}")

print("\n✅ MCCV complete — CSV ready for base_level_ranking.py")