import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = r"D:\capstone final project\capstoneproject\repeated3\All Outputs\ranking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

classifiers = ["RF", "LR", "LDA", "GNB", "ET", "XGB", "KNN", "SVM", "MLP", "DT"]

# ===============================
# RCV METRICS — ACTUAL VALUES
# FROM ALL 10 CLASSIFIER OUTPUTS
# Order: RF, LR, LDA, GNB, ET, XGB, KNN, SVM, MLP, DT
# ===============================
rcv_metrics = {
    "Accuracy": {
        "mean": [0.800, 0.786, 0.765, 0.768, 0.820, 0.798, 0.732, 0.762, 0.761, 0.697],
        "std":  [0.092, 0.094, 0.108, 0.095, 0.086, 0.093, 0.087, 0.095, 0.091, 0.103]
    },
    "Precision": {
        "mean": [0.799, 0.815, 0.800, 0.790, 0.824, 0.802, 0.878, 0.782, 0.780, 0.710],
        "std":  [0.106, 0.110, 0.125, 0.113, 0.107, 0.109, 0.127, 0.114, 0.109, 0.117]
    },
    "Sensitivity": {
        "mean": [0.831, 0.766, 0.734, 0.763, 0.842, 0.823, 0.564, 0.764, 0.763, 0.714],
        "std":  [0.132, 0.141, 0.157, 0.135, 0.121, 0.125, 0.152, 0.133, 0.140, 0.153]
    },
    "Specificity": {
        "mean": [0.768, 0.807, 0.797, 0.773, 0.797, 0.772, 0.908, 0.760, 0.760, 0.680],
        "std":  [0.137, 0.126, 0.139, 0.140, 0.137, 0.140, 0.098, 0.146, 0.137, 0.157]
    },
    "F1 Score": {
        "mean": [0.808, 0.782, 0.756, 0.768, 0.825, 0.805, 0.673, 0.764, 0.762, 0.702],
        "std":  [0.093, 0.102, 0.119, 0.098, 0.085, 0.091, 0.126, 0.097, 0.097, 0.112]
    },
    "ROC AUC": {
        "mean": [0.886, 0.880, 0.835, 0.872, 0.903, 0.890, 0.809, 0.844, 0.837, 0.697],
        "std":  [0.082, 0.081, 0.102, 0.085, 0.074, 0.077, 0.099, 0.094, 0.089, 0.103]
    },
    "TPR": {
        "mean": [0.831, 0.766, 0.734, 0.763, 0.842, 0.823, 0.564, 0.764, 0.763, 0.714],
        "std":  [0.132, 0.141, 0.157, 0.135, 0.121, 0.125, 0.152, 0.133, 0.140, 0.153]
    },
    "FPR": {
        "mean": [0.232, 0.193, 0.203, 0.227, 0.203, 0.228, 0.092, 0.240, 0.240, 0.320],
        "std":  [0.137, 0.126, 0.139, 0.140, 0.137, 0.140, 0.098, 0.146, 0.137, 0.157]
    },
    "Cohen's Kappa": {
        "mean": [0.599, 0.572, 0.530, 0.536, 0.639, 0.595, 0.467, 0.524, 0.522, 0.393],
        "std":  [0.185, 0.187, 0.215, 0.189, 0.172, 0.186, 0.171, 0.190, 0.181, 0.206]
    },
    "MCC": {
        "mean": [0.612, 0.584, 0.542, 0.547, 0.651, 0.607, 0.508, 0.535, 0.534, 0.404],
        "std":  [0.186, 0.188, 0.217, 0.191, 0.171, 0.186, 0.170, 0.191, 0.183, 0.208]
    }
}

def plot_ranking(classifiers, metrics, title, save_path):
    top7 = ["ET", "RF", "GNB", "SVM", "MLP", "LR", "XGB"]
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for i, (metric, data) in enumerate(metrics.items()):
        ax = axes[i]
        ratio = [(s/m) if m != 0 else 0 for m,s in zip(data["mean"], data["std"])]
        sorted_pairs = sorted(zip(classifiers, ratio), key=lambda x: x[1])
        sorted_cls, sorted_ratio = zip(*sorted_pairs)
        colors = ["#2ecc71" if c in top7 else "#e74c3c" for c in sorted_cls]
        ax.bar(sorted_cls, sorted_ratio, color=colors)
        ax.set_title(metric, fontsize=9, fontweight="bold")
        ax.set_ylabel("Std/Mean", fontsize=8)
        ax.set_ylim(0, max(sorted_ratio)*1.3)
        ax.set_xticklabels(sorted_cls, rotation=45, ha="right", fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(color="#2ecc71", label="Top 7 Selected"),
        Patch(color="#e74c3c", label="Excluded")
    ], loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")

def global_ranking(metrics, label):
    top7 = ["ET", "RF", "GNB", "SVM", "MLP", "LR", "XGB"]
    print(f"\n{'='*65}")
    print(f"   GLOBAL RANKING FROM {label}")
    print(f"{'='*65}")
    print(f"{'Rank':<6}{'Classifier':<10}{'Sum CV':<12}{'ACC':<8}{'AUC':<8}{'MCC':<8}{'Selected'}")
    print("-"*65)
    rows = []
    for cls in classifiers:
        idx = classifiers.index(cls)
        sum_cv = sum(
            (data["std"][idx]/data["mean"][idx]) if data["mean"][idx] != 0 else 0
            for data in metrics.values()
        )
        rows.append({
            "Classifier": cls,
            "Sum_CV":     round(sum_cv, 4),
            "ACC":        metrics["Accuracy"]["mean"][idx],
            "AUC":        metrics["ROC AUC"]["mean"][idx],
            "MCC":        metrics["MCC"]["mean"][idx]
        })
    df = pd.DataFrame(rows).sort_values("Sum_CV").reset_index(drop=True)
    df.index += 1
    for i, row in df.iterrows():
        selected = "YES" if row["Classifier"] in top7 else "NO"
        print(f"{i:<6}{row['Classifier']:<10}{row['Sum_CV']:<12}"
              f"{row['ACC']:<8}{row['AUC']:<8}{row['MCC']:<8}{selected}")
    return df

# RUN RCV
plot_ranking(classifiers, rcv_metrics,
    "RCV Base-Level Ranking (Std/Mean) — Lower is Better",
    os.path.join(OUTPUT_DIR, "RCV_Base_Level_Ranking.png"))

rcv_df = global_ranking(rcv_metrics, "RCV")
rcv_df.to_csv(os.path.join(OUTPUT_DIR, "RCV_Global_Ranking.csv"))
print(f"\nAll RCV outputs saved to: {OUTPUT_DIR}")

# MCCV SECTION
mccv_csv = r"D:\capstone final project\capstoneproject\mccv\All_Classifiers_Mean_Std_MCCV.csv"
try:
    df_mccv = pd.read_csv(mccv_csv)
    mccv_metrics = {}
    for metric in df_mccv["Metric"].unique():
        sub = df_mccv[df_mccv["Metric"] == metric]
        mccv_metrics[metric] = {
            "mean": [sub[sub["Classifier"]==c]["Mean"].values[0] for c in classifiers],
            "std":  [sub[sub["Classifier"]==c]["Std"].values[0]  for c in classifiers]
        }
    plot_ranking(classifiers, mccv_metrics,
        "MCCV Base-Level Ranking (Std/Mean) — Lower is Better",
        os.path.join(OUTPUT_DIR, "MCCV_Base_Level_Ranking.png"))
    mccv_df = global_ranking(mccv_metrics, "MCCV")
    mccv_df.to_csv(os.path.join(OUTPUT_DIR, "MCCV_Global_Ranking.csv"))

    # COMBINED
    print(f"\n{'='*65}")
    print("   COMBINED RCV + MCCV FINAL RANKING")
    print(f"{'='*65}")
    combined = rcv_df[["Classifier","Sum_CV"]].rename(columns={"Sum_CV":"RCV"})
    combined = combined.merge(
        mccv_df[["Classifier","Sum_CV"]].rename(columns={"Sum_CV":"MCCV"}),
        on="Classifier")
    combined["Combined"] = combined["RCV"] + combined["MCCV"]
    combined = combined.sort_values("Combined").reset_index(drop=True)
    combined.index += 1
    top7 = ["ET","RF","GNB","SVM","MLP","LR","XGB"]
    print(f"{'Rank':<6}{'Classifier':<12}{'RCV':<10}{'MCCV':<10}{'Combined':<12}{'Selected'}")
    print("-"*60)
    for i, row in combined.iterrows():
        sel = "YES" if row["Classifier"] in top7 else "NO"
        print(f"{i:<6}{row['Classifier']:<12}{row['RCV']:<10}{row['MCCV']:<10}{row['Combined']:<12}{sel}")
    combined.to_csv(os.path.join(OUTPUT_DIR, "Combined_Final_Ranking.csv"))
    print("\nAll MCCV + Combined outputs saved")

except FileNotFoundError:
    print(f"\nMCCV CSV not found — run mccv.py first")
    print(f"Expected path: {mccv_csv}")
    print("RCV ranking above is complete and ready to use")