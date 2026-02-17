import pandas as pd
import joblib

MODEL_DIR = "C:/capstone/alzheimers/proposed/ensemble6/boosting/"

# Load model & selector
model = joblib.load(MODEL_DIR + "adaboost_model.joblib")
selector = joblib.load(MODEL_DIR + "adaboost_selector.joblib")

# Load new unseen data
REAL_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REAL_PATH)

# Apply same feature selection
X_real_sel = selector.transform(df_real)

# Predict
predictions = model.predict(X_real_sel)

df_real["Prediction"] = ["Alzheimer Risk" if p == 1 else "Healthy" for p in predictions]

df_real.to_csv(MODEL_DIR + "boosting_realtime_predictions.csv", index=False)

print("✅ Real-time Boosting prediction completed")
