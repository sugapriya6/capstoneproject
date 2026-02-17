import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix
)

# =========================
# CLASSIFIERS
# =========================
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# =========================
# 1. LOAD DATA
# =========================
data_path = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
data = pd.read_csv(data_path)

X = data.drop(columns=["class"])
y = data["class"].map({"P": 1, "H": 0})

# =========================
# 2. DEFINE MODELS
# =========================
models = {
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "LR": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "GNB": GaussianNB(),
    "ET": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "XGB": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True),
    "MLP": MLPClassifier(max_iter=500, random_state=42),
    "DT": DecisionTreeClassifier(random_state=42)
}

# =========================
# 3. CV SETUP
# =========================
# 🔹 RCV
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# 🔹 MCCV (Monte Carlo CV – fewer repeats)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

# =========================
# 4. METRIC COLLECTION
# =========================
final_results = []

for model_name, model in models.items():
    print(f"Running {model_name}...")

    scores = {
        "ACC": [], "PREC": [], "REC": [], "SPEC": [],
        "TPR": [], "FPR": [],
        "F1": [], "AUC": [], "KAPPA": [], "MCC": []
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)           # Sensitivity
        spec = tn / (tn + fp)
        tpr = rec                                    # Explicit TPR
        fpr = fp / (fp + tn)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        scores["ACC"].append(acc)
        scores["PREC"].append(prec)
        scores["REC"].append(rec)
        scores["SPEC"].append(spec)
        scores["TPR"].append(tpr)
        scores["FPR"].append(fpr)
        scores["F1"].append(f1)
        scores["AUC"].append(auc)
        scores["KAPPA"].append(kappa)
        scores["MCC"].append(mcc)

    # Store mean & std
    for metric in scores:
        final_results.append({
            "Classifier": model_name,
            "Metric": metric,
            "Mean": np.mean(scores[metric]),
            "Std": np.std(scores[metric])
        })

# =========================
# 5. SAVE RESULTS
# =========================
results_df = pd.DataFrame(final_results)
results_df.to_csv("All_Classifiers_Mean_Std_MCCV.csv", index=False)

print("\nSaved file: All_Classifiers_Mean_Std_MCCV.csv")
print(results_df.head())
