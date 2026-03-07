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
OUTPUT_PATH   = r"D:\capstone final project\capstoneproject\ensemble\ensemble_predictions.csv"

# ==============================
# 2. LOAD PROPOSED MODEL
#    Load from ENSEMBLE_DIR
#    These were saved after retraining on full DARWIN
# ==============================
meta_lr  = joblib.load(ENSEMBLE_DIR + r"\meta_lr_rf_et.joblib")
rf_model = joblib.load(ENSEMBLE_DIR + r"\base_RF.joblib")   # ← from ensemble folder
et_model = joblib.load(ENSEMBLE_DIR + r"\base_ET.joblib")   # ← from ensemble folder

# Load feature lists (still from topkfeatures folder)
rf_feats = joblib.load(BASE + r"\random\rf_features.joblib")
et_feats = joblib.load(BASE + r"\extratree\et_features.joblib")

print("Proposed model (RF+ET stacking) loaded ✅")
print(f"RF uses {len(rf_feats)} features")
print(f"ET uses {len(et_feats)} features")

# ==============================
# 3. LOAD REAL-TIME DATA
# ==============================
df_real = pd.read_csv(REALTIME_PATH)
print(f"\nReal-time samples loaded: {len(df_real)}")
print(f"Features available: {df_real.shape[1]}")

# ==============================
# 4. CHECK ALL REQUIRED FEATURES EXIST
# ==============================
missing_rf = [f for f in rf_feats if f not in df_real.columns]
missing_et = [f for f in et_feats if f not in df_real.columns]

if missing_rf:
    print(f"⚠️ Missing RF features: {missing_rf}")
if missing_et:
    print(f"⚠️ Missing ET features: {missing_et}")
if not missing_rf and not missing_et:
    print("All required features present ✅")

# ==============================
# 5. GET BASE PREDICTIONS
#    RF and ET on their own top-k features
# ==============================
H_test = np.zeros((len(df_real), 2))

H_test[:, 0] = rf_model.predict_proba(df_real[rf_feats])[:, 1]
H_test[:, 1] = et_model.predict_proba(df_real[et_feats])[:, 1]

print(f"\nRF probability range : {H_test[:,0].min():.3f} – {H_test[:,0].max():.3f}")
print(f"ET probability range : {H_test[:,1].min():.3f} – {H_test[:,1].max():.3f}")

# ==============================
# 6. META-LEARNER FINAL PREDICTION
# ==============================
y_prob = meta_lr.predict_proba(H_test)[:, 1]
y_pred = meta_lr.predict(H_test)

print(f"\nFinal probability range : {y_prob.min():.3f} – {y_prob.max():.3f}")

# ==============================
# 7. RISK CATEGORY
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
# 8. BUILD OUTPUT
# ==============================
df_out = df_real.copy()
df_out["RF_Probability"]         = np.round(H_test[:, 0], 4)
df_out["ET_Probability"]         = np.round(H_test[:, 1], 4)
df_out["Prediction_Probability"] = np.round(y_prob, 4)
df_out["Predicted_Class"]        = risk_results["Predicted_Class"].values
df_out["Risk_Category"]          = risk_results["Risk_Category"].values

# ==============================
# 9. PRINT RESULTS
# ==============================
print("\n" + "="*65)
print("   ENSEMBLE (RF+ET) REAL-TIME PREDICTIONS — FIRST 5 ROWS")
print("="*65)
print(df_out[["RF_Probability",
              "ET_Probability",
              "Prediction_Probability",
              "Predicted_Class",
              "Risk_Category"]].head(5).to_string())

print("\n" + "="*40)
print("   PREDICTION SUMMARY")
print("="*40)
counts = df_out["Predicted_Class"].value_counts()
total  = len(df_out)
for cls, cnt in counts.items():
    label = "Alzheimer's Patient" if cls == "P" else "Healthy"
    print(f"  {cls} ({label}) : {cnt} / {total} "
          f"({cnt/total*100:.1f}%)")

print("\n--- Risk Category Breakdown ---")
risk_counts = df_out["Risk_Category"].value_counts()
for risk, cnt in risk_counts.items():
    print(f"  {risk:<20}: {cnt} samples ({cnt/total*100:.1f}%)")

# ==============================
# 10. SAVE
# ==============================
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ All {len(df_out)} predictions saved to:")
print(f"   {OUTPUT_PATH}")