import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

# ==============================
# 1. PATHS
# ==============================
MODEL_DIR     = r"D:\capstone final project\capstoneproject\topkfeatures\gaussian"
TRAIN_PATH    = r"D:\capstone final project\capstoneproject\preprocess1\data_train_processed.csv"
REALTIME_PATH = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"
OUTPUT_PATH   = r"D:\capstone final project\capstoneproject\repeated3\All Outputs\gnb_topk_predictions.csv"

# ==============================
# 2. LOAD MODEL AND SELECTOR
# ==============================
gnb_model         = joblib.load(MODEL_DIR + r"\gnb_model.joblib")
selector          = joblib.load(MODEL_DIR + r"\gnb_selector.joblib")
selected_features = joblib.load(MODEL_DIR + r"\gnb_features.joblib")

print(f"Model loaded. Selected features: {len(selected_features)}")

# ==============================
# 3. LOAD DARWIN TRAINING DATA
#    To get mean and std for normalization
# ==============================
df_train  = pd.read_csv(TRAIN_PATH)
X_train   = df_train.drop(columns=["class"])

# Compute DARWIN mean and std
darwin_mean = X_train.mean()
darwin_std  = X_train.std()

print(f"DARWIN stats loaded for normalization")

# ==============================
# 4. LOAD REAL-TIME STUDENT DATA
#    Raw (before normalization)
# ==============================
df_real = pd.read_csv(REALTIME_PATH)
print(f"Real-time samples: {len(df_real)}")

# ==============================
# 5. NORMALIZE STUDENT DATA
#    Using DARWIN mean and std
#    NOT student's own mean/std
# ==============================
X_real_raw = df_real[X_train.columns].copy()

# Apply DARWIN Z-score to student data
X_real_normalized = (X_real_raw - darwin_mean) / darwin_std

# Fill any NaN (if std=0 for some feature)
X_real_normalized = X_real_normalized.fillna(0)

print(f"Student data normalized using DARWIN statistics")

# ==============================
# 6. APPLY ANOVA SELECTOR
# ==============================
full_features  = selector.feature_names_in_
X_real_full    = X_real_normalized.reindex(
    columns=full_features, fill_value=0)
X_real_sel     = selector.transform(X_real_full)

# ==============================
# 7. PREDICT
# ==============================
y_prob = gnb_model.predict_proba(X_real_sel)[:, 1]

print(f"\nProbability range: {y_prob.min():.3f} to {y_prob.max():.3f}")

# ==============================
# 8. RISK CATEGORY
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
# 9. BUILD OUTPUT
# ==============================
df_real["Prediction_Probability"] = np.round(y_prob, 4)
df_real["Predicted_Class"]        = risk_results["Predicted_Class"].values
df_real["Risk_Category"]          = risk_results["Risk_Category"].values

# ==============================
# 10. PRINT RESULTS
# ==============================
print("\n===== GNB TOP-K REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(df_real[["Predicted_Class",
               "Risk_Category",
               "Prediction_Probability"]].head(5))

print("\n===== PREDICTION SUMMARY =====")
print(df_real["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(df_real["Risk_Category"].value_counts())

# ==============================
# 11. SAVE
# ==============================
df_real.to_csv(OUTPUT_PATH, index=False)
print(f"\nAll {len(df_real)} predictions saved to gnb_topk_predictions.csv")