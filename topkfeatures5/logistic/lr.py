############# Logistic Regression #############
#############  REF  ##################
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
from sklearn.feature_selection import RFE
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

# Create a Logistic Regression classifier
lr_classifier = LogisticRegression()

# Perform grid search for k values
min_k = 10
max_k = X.shape[1]  # Use the total number of features in your dataset
step = 10

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

for k in range(min_k, max_k + 1, step):
    # Apply Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=lr_classifier, n_features_to_select=k)
    X_train_rfe = rfe.fit_transform(X_train, y_train)

    # Train the Logistic Regression classifier using the selected features
    lr_classifier.fit(X_train_rfe, y_train)

    # Apply the same feature selection to the test set
    X_test_rfe = rfe.transform(X_test)

    # Make predictions on the test set
    y_pred = lr_classifier.predict(X_test_rfe)
    y_prob = lr_classifier.predict_proba(X_test_rfe)[:, 1]

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
plt.savefig("C:/capstone/alzheimers/proposed/topkfeatures5/logistic/logistic_metrics.png", dpi=300, bbox_inches="tight")
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
# SAVE FINAL MODEL
# ==============================
MODEL_DIR = "C:/capstone/alzheimers/proposed/topkfeatures5/logistic/"

joblib.dump(lr_classifier, MODEL_DIR + "lr_model.joblib")
joblib.dump(X.columns[rfe.support_], MODEL_DIR + "lr_features.joblib")

print("✅ Logistic Regression model and selected features saved successfully")
