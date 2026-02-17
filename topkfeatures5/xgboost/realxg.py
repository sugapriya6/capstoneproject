import pandas as pd
import joblib

# ==============================
# LOAD SAVED XGBOOST OBJECTS
# ==============================
MODEL_DIR = "C:/capstone/alzheimers/proposed/topkfeatures5/xgboost/"

xgb_classifier = joblib.load(MODEL_DIR + "xgb_model.joblib")
selected_features = joblib.load(MODEL_DIR + "xgb_features.joblib")

# ==============================
# LOAD REAL-TIME DATA
# ==============================
REALTIME_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REALTIME_PATH)

# ==============================
# FEATURE ALIGNMENT
# (NO FEATURE MODIFICATION)
# ==============================
X_real_final = df_real[selected_features]

# ==============================
# PREDICTION
# ==============================
y_real_pred = xgb_classifier.predict(X_real_final)
y_real_prob = xgb_classifier.predict_proba(X_real_final)[:, 1]

# ==============================
# CREATE OUTPUT DATAFRAME
# ==============================
output_df = pd.DataFrame({
    "User_ID": range(1, len(X_real_final) + 1),
    "Predicted_Label": ["Alzheimer Risk" if y == 1 else "Healthy" for y in y_real_pred],
    "Risk_Score_Percent": (y_real_prob * 100).round(2)
})

# ==============================
# SAVE TO CSV
# ==============================
OUTPUT_PATH = (
    "C:/capstone/alzheimers/proposed/topkfeatures5/xgboost/"
    "realtime_predictions.csv"
)
output_df.to_csv(OUTPUT_PATH, index=False)

# ==============================
# DISPLAY OUTPUT
# ==============================
print("\n========= REAL-TIME ALZHEIMER RISK PREDICTION (XGBOOST) =========")
print(output_df)
print(f"\nResults saved to: {OUTPUT_PATH}")
