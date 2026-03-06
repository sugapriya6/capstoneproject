import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

# ==============================
# 1. PATHS
# ==============================
BASE          = r"D:\capstone final project\capstoneproject\topkfeatures"
ENSEMBLE_DIR  = r"D:\capstone final project\capstoneproject\topkfeatures\ensemble"
REALTIME_PATH = r"D:\capstone final project\capstoneproject\preprocess1\data_test_processed.csv"
OUTPUT_PATH   = r"D:\capstone final project\capstoneproject\repeated3\All Outputs\ensemble_predictions.csv"

# ==============================
# 2. LOAD PROPOSED MODEL
#    RF + ET with LR meta-learner
# ==============================
meta_lr   = joblib.load(ENSEMBLE_DIR + r"\meta_lr_rf_et.joblib")

rf_model  = joblib.load(BASE + r"\random\rf_model.joblib")
rf_feats  = joblib.load(BASE + r"\random\rf_features.joblib")

et_model  = joblib.load(BASE + r"\extratree\et_model.joblib")
et_feats  = joblib.load(BASE + r"\extratree\et_features.joblib")

print("Proposed model (RF+ET stacking) loaded ✅")

# ==============================
# 3. LOAD REAL-TIME DATA
# ==============================
df_real = pd.read_csv(REALTIME_PATH)
print(f"Real-time samples: {len(df_real)}")

# ==============================
# 4. GET BASE PREDICTIONS
#    RF and ET on their own top-k features
# ==============================
H_test = np.zeros((len(df_real), 2))

H_test[:, 0] = rf_model.predict_proba(df_real[rf_feats])[:, 1]
H_test[:, 1] = et_model.predict_proba(df_real[et_feats])[:, 1]

print(f"RF prob range : {H_test[:,0].min():.3f} – {H_test[:,0].max():.3f}")
print(f"ET prob range : {H_test[:,1].min():.3f} – {H_test[:,1].max():.3f}")

# ==============================
# 5. META-LEARNER FINAL PREDICTION
# ==============================
y_prob = meta_lr.predict_proba(H_test)[:, 1]

# ==============================
# 6. RISK CATEGORY
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
# All original features + base probs
# + final prediction + risk
# ==============================
df_real["RF_Probability"]         = np.round(H_test[:, 0], 4)
df_real["ET_Probability"]         = np.round(H_test[:, 1], 4)
df_real["Prediction_Probability"] = np.round(y_prob, 4)
df_real["Predicted_Class"]        = risk_results["Predicted_Class"].values
df_real["Risk_Category"]          = risk_results["Risk_Category"].values

# ==============================
# 8. PRINT RESULTS
# ==============================
print("\n===== ENSEMBLE (RF+ET) REAL-TIME PREDICTIONS (Sample 5 Rows) =====")
print(df_real[["RF_Probability",
               "ET_Probability",
               "Prediction_Probability",
               "Predicted_Class",
               "Risk_Category"]].head(5).to_string())

print("\n===== PREDICTION SUMMARY =====")
print(df_real["Predicted_Class"].value_counts())
print("\n--- Risk Category Breakdown ---")
print(df_real["Risk_Category"].value_counts())

# ==============================
# 9. SAVE
# ==============================
df_real.to_csv(OUTPUT_PATH, index=False)
print(f"\nAll {len(df_real)} predictions saved to ensemble_predictions.csv")