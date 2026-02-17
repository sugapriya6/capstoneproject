############# Gaussian Naive Bayes with SelectKBest #############
############# ANOVA ###################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score
)

# Load the dataset (replace 'your_dataset.csv' with your actual file name)
file_path = "C:/capstone/alzheimers/proposed/preprocess1/data_train_processed.csv"
df = pd.read_csv(file_path)

# Assuming the last column is the class label
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Convert class labels ('H' and 'P') to numerical values if necessary
# Example: If 'H' is 0 and 'P' is 1
y = y.map({'H': 0, 'P': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()

# Initialize lists to store metric values and corresponding indices
k_values = []
accuracy_values = []
sensitivity_values = []
specificity_values = []
precision_values = []
tpr_values = []
fpr_values = []
f1_values = []
mcc_values = []
cohen_kappa_values = []
auc_roc_values = []

# Iterate over different values of k
for k in range(10, 339, 10):
    # Apply SelectKBest feature selection
    k_best_selector = SelectKBest(f_classif, k=k)
    X_train_selected = k_best_selector.fit_transform(X_train, y_train)
    X_test_selected = k_best_selector.transform(X_test)

    # Train the Gaussian Naive Bayes classifier using the selected features
    gnb_classifier.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred = gnb_classifier.predict(X_test_selected)
    y_prob = gnb_classifier.predict_proba(X_test_selected)[:, 1]

    # Calculate and append metric values for each value of k
    k_values.append(k)
    accuracy_values.append(accuracy_score(y_test, y_pred))
    sensitivity_values.append(recall_score(y_test, y_pred))

    # Calculate specificity using confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity_values.append(tn / (tn + fp))

    precision_values.append(precision_score(y_test, y_pred))
    tpr_values.append(recall_score(y_test, y_pred))
    fpr_values.append(fp / (tn + fp))
    f1_values.append(f1_score(y_test, y_pred))
    mcc_values.append(matthews_corrcoef(y_test, y_pred))
    cohen_kappa_values.append(cohen_kappa_score(y_test, y_pred))
    auc_roc_values.append(roc_auc_score(y_test, y_prob))

# Find the indices for best values
best_accuracy_index = np.argmax(accuracy_values)
best_sensitivity_index = np.argmax(sensitivity_values)
best_specificity_index = np.argmax(specificity_values)
best_precision_index = np.argmax(precision_values)
best_tpr_index = np.argmax(tpr_values)
best_fpr_index = np.argmin(fpr_values)
best_f1_index = np.argmax(f1_values)
best_mcc_index = np.argmax(mcc_values)
best_cohen_kappa_index = np.argmax(cohen_kappa_values)
best_auc_roc_index = np.argmax(auc_roc_values)

# Plot the metrics across different values of k
plt.figure(figsize=(12, 8))

plt.plot(k_values, accuracy_values, label='Accuracy')
plt.plot(k_values, sensitivity_values, label='Sensitivity (TPR)')
plt.plot(k_values, specificity_values, label='Specificity')
plt.plot(k_values, precision_values, label='Precision')
plt.plot(k_values, tpr_values, label='True Positive Rate (TPR)')
plt.plot(k_values, fpr_values, label='False Positive Rate (FPR)')
plt.plot(k_values, f1_values, label='F1 Score')
plt.plot(k_values, mcc_values, label='MCC (Matthews Correlation Coefficient)')
plt.plot(k_values, cohen_kappa_values, label="Cohen's Kappa")
plt.plot(k_values, auc_roc_values, label='AUC-ROC')

# Mark the points on the plot
plt.scatter(k_values[best_accuracy_index], accuracy_values[best_accuracy_index], marker='o', color='orange', label='Best Accuracy')
plt.scatter(k_values[best_sensitivity_index], sensitivity_values[best_sensitivity_index], marker='o', color='blue', label='Best Sensitivity')
plt.scatter(k_values[best_specificity_index], specificity_values[best_specificity_index], marker='o', color='green', label='Best Specificity')
plt.scatter(k_values[best_precision_index], precision_values[best_precision_index], marker='o', color='red', label='Best Precision')
plt.scatter(k_values[best_tpr_index], tpr_values[best_tpr_index], marker='o', color='purple', label='Best TPR')
plt.scatter(k_values[best_fpr_index], fpr_values[best_fpr_index], marker='o', color='yellow', label='Best FPR')
plt.scatter(k_values[best_f1_index], f1_values[best_f1_index], marker='o', color='brown', label='Best F1 Score')
plt.scatter(k_values[best_mcc_index], mcc_values[best_mcc_index], marker='o', color='pink', label='Best MCC')
plt.scatter(k_values[best_cohen_kappa_index], cohen_kappa_values[best_cohen_kappa_index], marker='o', color='cyan', label="Best Cohen's Kappa")
plt.scatter(k_values[best_auc_roc_index], auc_roc_values[best_auc_roc_index], marker='o', color='gray', label='Best AUC-ROC')

plt.title('Performance Metrics across Different Values of k')
plt.xlabel('Number of Features (k)')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)

plt.savefig("C:/capstone/alzheimers/proposed/topkfeatures5/gaussian/gausian_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

# Print the best k values for each metric
print(f"Best Accuracy (k={k_values[best_accuracy_index]}): {accuracy_values[best_accuracy_index]}")
print(f"Best Sensitivity (TPR) (k={k_values[best_sensitivity_index]}): {sensitivity_values[best_sensitivity_index]}")
print(f"Best Specificity (k={k_values[best_specificity_index]}): {specificity_values[best_specificity_index]}")
print(f"Best Precision (k={k_values[best_precision_index]}): {precision_values[best_precision_index]}")
print(f"Best TPR (k={k_values[best_tpr_index]}): {tpr_values[best_tpr_index]}")
print(f"Best FPR (k={k_values[best_fpr_index]}): {fpr_values[best_fpr_index]}")
print(f"Best F1 Score (k={k_values[best_f1_index]}): {f1_values[best_f1_index]}")
print(f"Best MCC (k={k_values[best_mcc_index]}): {mcc_values[best_mcc_index]}")
print(f"Best Cohen's Kappa (k={k_values[best_cohen_kappa_index]}): {cohen_kappa_values[best_cohen_kappa_index]}")
print(f"Best AUC-ROC (k={k_values[best_auc_roc_index]}): {auc_roc_values[best_auc_roc_index]}")

     
# ==============================
# SAVE FINAL GNB MODEL (BEST k)
# ==============================

import joblib

# Select best k based on MCC
best_k = k_values[best_mcc_index]
print("Saving GaussianNB model with best k =", best_k)

# Refit SelectKBest with best k on FULL training data
final_selector = SelectKBest(score_func=f_classif, k=best_k)
X_train_final = final_selector.fit_transform(X_train, y_train)

# Train final Gaussian Naive Bayes model
final_gnb_model = GaussianNB()
final_gnb_model.fit(X_train_final, y_train)

# Save directory
MODEL_DIR = "C:/capstone/alzheimers/proposed/topkfeatures5/gaussian/"

# Save model, selector, and selected feature names
joblib.dump(final_gnb_model, MODEL_DIR + "gnb_model.joblib")
joblib.dump(final_selector, MODEL_DIR + "gnb_selector.joblib")
joblib.dump(X.columns[final_selector.get_support()], MODEL_DIR + "gnb_features.joblib")

print("✅ Gaussian Naive Bayes model saved successfully")
