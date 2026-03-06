import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

# ==============================
# 1. PATHS
# ==============================
MODEL_DIR     = r"D:\capstone final project\capstoneproject\topkfeatures\extratree"
REALTIME_PATH = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"
OUTPUT_PATH   = r"D:\capstone final project\capstoneproject\repeated3\All Outputs\et_topk_predictions.csv"

# ==============================
# 2. LOAD MODEL AND FEATURES
# ==============================
et_model          = joblib.load(MODEL_DIR + r"\et_model.joblib")
selected_features = joblib.load(MODEL_DIR + r"\et_features.joblib")

print(f"Model loaded. Selected features: {len(selected_features)}")

# ==============================
# 3. LOAD REAL-TIME DATA
# ==============================
df_real = pd.read_csv(REALTIME_PATH)
print(f"Real-time samples: {len(df_real)}")

# ==============================
# 4. FEATURE ALIGNMENT
# ==============================
X_real = df_real[selected_features]

# ==============================
# 5. PREDICT
# ==============================
y_prob = et_model.predict_proba(X_real)[:, 1]

# ==============================
# 6. RISK CATEGORY
# Same thresholds as all other classifiers
# ==============================
def risk_category(prob):
    if prob < 0.30:
        return "H", "Low Risk"
    elif prob < 0.60:
        return "H", "Moderate Risk"
    elif prob < 0.80:
        return "P", "High Risk"
    else:
        return "P", "Critical Risk"

risk_results = pd.Series(y_prob).apply(
    lambda p: pd.Series(
        risk_category(p),
        index=["Predicted_Class", "Risk_Category"]
    )
)

# ==============================
# 7. BUILD OUTPUT
# Same format as extratrees.csv
# All original features + 3 columns
# ==============================
df_real["Prediction_Probability"] = np.round(y_prob, 4)
df_real["Predicted_Class"]        = risk_results["Predicted_Class"].values
df_real["Risk_Category"]          = risk_results["Risk_Category"].values

# ==============================
# 8. PRINT RESULTS
# ==============================
print("\n===== ET TOP-K REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(df_real[["Predicted_Class",
               "Risk_Category",
               "Prediction_Probability"]].head(5))

print("\n===== PREDICTION SUMMARY =====")
print(df_real["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(df_real["Risk_Category"].value_counts())

# ==============================
# 9. SAVE
# ==============================
df_real.to_csv(OUTPUT_PATH, index=False)
print(f"\nAll {len(df_real)} predictions saved to et_topk_predictions.csv")