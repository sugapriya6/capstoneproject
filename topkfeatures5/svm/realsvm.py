import pandas as pd
import joblib

# ==============================
# LOAD SAVED OBJECTS
# ==============================
MODEL_DIR = "C:/capstone/alzheimers/proposed/topkfeatures5/svm/"

svm_model = joblib.load(MODEL_DIR + "svm_model.joblib")
selector = joblib.load(MODEL_DIR + "svm_selector.joblib")

# ==============================
# LOAD REAL-TIME DATA
# ==============================
REALTIME_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REALTIME_PATH)

# ==============================
# IMPORTANT: USE ALL FEATURES
# (same as training, except class)
# ==============================
X_real_full = df_real.copy()   # no column removal

# ==============================
# APPLY SAME ANOVA SELECTION
# ==============================
X_real_selected = selector.transform(X_real_full)

# ==============================
# PREDICTION
# ==============================
y_real_pred = svm_model.predict(X_real_selected)
y_real_prob = svm_model.predict_proba(X_real_selected)[:, 1]

# ==============================
# OUTPUT DATAFRAME
# ==============================
output_df = pd.DataFrame({
    "User_ID": range(1, len(X_real_selected) + 1),
    "Predicted_Label": ["Alzheimer Risk" if y == 1 else "Healthy" for y in y_real_pred],
    "Risk_Score_Percent": (y_real_prob * 100).round(2)
})

# ==============================
# SAVE RESULTS
# ==============================
OUTPUT_PATH = (
    "C:/capstone/alzheimers/proposed/topkfeatures5/svm/"
    "realtime_predictions.csv"
)
output_df.to_csv(OUTPUT_PATH, index=False)

# ==============================
# DISPLAY
# ==============================
print("\n========= REAL-TIME ALZHEIMER RISK PREDICTION (SVM) =========")
print(output_df)
print(f"\nResults saved to: {OUTPUT_PATH}")
