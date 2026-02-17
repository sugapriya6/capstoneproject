import numpy as np
import matplotlib.pyplot as plt

# Data for each classifier
classifiers = ["RF", "LR", "LDA", "GNB", "Extra Tree", "XGB", "KNN", "SVM", "MLP", "DT"]

# Metrics data for each classifier
metrics = {

    "Accuracy": {
        "mean": [0.8027, 0.7976, 0.7597, 0.7627, 0.8190, 0.7966, 0.7376, 0.8144, 0.7806, 0.7108],
        "std":  [0.0931, 0.0901, 0.0983, 0.0973, 0.0840, 0.0895, 0.0922, 0.0888, 0.0902, 0.0930]
    },

    "Specificity": {
        "mean": [0.7719, 0.8131, 0.7853, 0.7697, 0.7983, 0.7683, 0.9097, 0.7719, 0.7811, 0.6794],
        "std":  [0.1426, 0.1235, 0.1390, 0.1468, 0.1442, 0.1441, 0.0991, 0.1408, 0.1353, 0.1235]
    },

    "Sensitivity (Recall)": {
        "mean": [0.8336, 0.7839, 0.7358, 0.7581, 0.8406, 0.8244, 0.5733, 0.8564, 0.7817, 0.7414],
        "std":  [0.1318, 0.1421, 0.1469, 0.1342, 0.1244, 0.1099, 0.1525, 0.1318, 0.1387, 0.1409]
    },

    "Precision": {
        "mean": [0.8028, 0.8248, 0.7924, 0.7872, 0.8260, 0.7987, 0.8813, 0.8089, 0.8005, 0.7110],
        "std":  [0.1130, 0.1103, 0.1197, 0.1208, 0.1141, 0.1077, 0.1233, 0.1096, 0.1165, 0.0991]
    },

    "TPR (True Positive Rate)": {   # same as Recall
        "mean": [0.8336, 0.7839, 0.7358, 0.7581, 0.8406, 0.8244, 0.5733, 0.8564, 0.7817, 0.7414],
        "std":  [0.1318, 0.1421, 0.1469, 0.1342, 0.1244, 0.1099, 0.1525, 0.1318, 0.1387, 0.1409]
    },

    "FPR (False Positive Rate)": {
        "mean": [0.2281, 0.1869, 0.2147, 0.2303, 0.2017, 0.2317, 0.0903, 0.2281, 0.2189, 0.3206],
        "std":  [0.1426, 0.1235, 0.1390, 0.1468, 0.1442, 0.1441, 0.0991, 0.1408, 0.1353, 0.1235]
    },

    "Cohen's Kappa": {
        "mean": [0.6053, 0.5957, 0.5198, 0.5262, 0.6380, 0.5927, 0.4795, 0.6285, 0.5615, 0.4208],
        "std":  [0.1856, 0.1793, 0.1961, 0.1942, 0.1678, 0.1786, 0.1812, 0.1769, 0.1804, 0.1855]
    },

    "F1 Score": {
        "mean": [0.8097, 0.7941, 0.7536, 0.7631, 0.8244, 0.8053, 0.6821, 0.8228, 0.7812, 0.7197],
        "std":  [0.0950, 0.1012, 0.1065, 0.0994, 0.0840, 0.0858, 0.1254, 0.0880, 0.0957, 0.1022]
    },

    "MCC (Matthews Correlation Coefficient)": {
        "mean": [0.6187, 0.6080, 0.5326, 0.5375, 0.6531, 0.6027, 0.5186, 0.6449, 0.5748, 0.4294],
        "std":  [0.1849, 0.1786, 0.1983, 0.1960, 0.1670, 0.1768, 0.1787, 0.1781, 0.1826, 0.1872]
    },

    "ROC AUC": {
        "mean": [0.8821, 0.8815, 0.8387, 0.8716, 0.8986, 0.8912, 0.7948, 0.8740, 0.8526, 0.7104],
        "std":  [0.0805, 0.0791, 0.0888, 0.0890, 0.0784, 0.0716, 0.0875, 0.0847, 0.0853, 0.0927]
    },
}

# Create a figure and subplots for each metric in a 2x5 grid
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 8))
fig.suptitle("Standard Deviation to Mean Ratio for Classifier Performance Metrics")

# Flatten the axes for easy iteration
axes = axes.flatten()

for i, (metric, data) in enumerate(metrics.items()):
    ax = axes[i]
    mean_values = data["mean"]
    std_values = data["std"]
    ratio = [(std / mean) if mean != 0 and std != 0 else mean for mean, std in zip(mean_values, std_values)]

    # Sort classifiers based on the ratio values
    sorted_classifiers, sorted_ratio = zip(*sorted(zip(classifiers, ratio), key=lambda x: x[1]))

    ax.bar(sorted_classifiers, sorted_ratio, capsize=5)
    ax.set_title(f"{metric} (Std/Mean or Mean)")
    ax.set_ylim([0, max(sorted_ratio) * 1.2])
    ax.set_xticklabels(sorted_classifiers, rotation=45)  # Rotate y-axis labels by 45 degrees

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

     