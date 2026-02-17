import pandas as pd
import joblib

# ==============================
# LOAD MODEL & FEATURES
# ==============================
MODEL_DIR = "C:/capstone/alzheimers/proposed/topkfeatures5/random/"

rf_model = joblib.load(MODEL_DIR + "rf_model.joblib")
selected_features = joblib.load(MODEL_DIR + "rf_features.joblib")

# ==============================
# LOAD REAL-TIME DATA
# ==============================
REALTIME_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REALTIME_PATH)

# ==============================
# FEATURE ALIGNMENT
# ==============================
X_real = df_real[selected_features]

# ==============================
# PREDICTION
# ==============================
y_pred = rf_model.predict(X_real)
y_prob = rf_model.predict_proba(X_real)[:, 1]

# ==============================
# SAVE OUTPUT
# ==============================
output_df = pd.DataFrame({
    "User_ID": range(1, len(X_real) + 1),
    "Predicted_Label": ["Alzheimer Risk" if y == 1 else "Healthy" for y in y_pred],
    "Risk_Score_Percent": (y_prob * 100).round(2)
})

OUTPUT_PATH = MODEL_DIR + "rf_realtime_predictions.csv"
output_df.to_csv(OUTPUT_PATH, index=False)

print("\n========= RANDOM FOREST REAL-TIME RESULTS =========")
print(output_df)
print(f"\nSaved at: {OUTPUT_PATH}")
