import pandas as pd
import joblib

# ==============================
# LOAD SAVED OBJECTS
# ==============================
MODEL_DIR = "C:/capstone/alzheimers/proposed/topkfeatures5/mlp/"

mlp_model = joblib.load(MODEL_DIR + "mlp_model.joblib")
selector = joblib.load(MODEL_DIR + "mlp_selector.joblib")

# VERY IMPORTANT
full_features = selector.feature_names_in_

# ==============================
# LOAD REAL-TIME DATA
# ==============================
REALTIME_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REALTIME_PATH)

# ==============================
# FEATURE ALIGNMENT (CRITICAL)
# ==============================
X_real_full = df_real.reindex(columns=full_features, fill_value=0)

# ==============================
# APPLY SELECTKBEST
# ==============================
X_real_selected = selector.transform(X_real_full)

# ==============================
# PREDICTION
# ==============================
y_real_pred = mlp_model.predict(X_real_selected)
y_real_prob = mlp_model.predict_proba(X_real_selected)[:, 1]

# ==============================
# SAVE OUTPUT
# ==============================
output_df = pd.DataFrame({
    "User_ID": range(1, len(X_real_selected) + 1),
    "Predicted_Label": ["Alzheimer Risk" if y == 1 else "Healthy" for y in y_real_pred],
    "Risk_Score_Percent": (y_real_prob * 100).round(2)
})

OUTPUT_PATH = MODEL_DIR + "realtime_predictions.csv"
output_df.to_csv(OUTPUT_PATH, index=False)

print("\n========= REAL-TIME ALZHEIMER RISK PREDICTION =========")
print(output_df)
print(f"\nResults saved to: {OUTPUT_PATH}")
