import pandas as pd
import joblib

MODEL_DIR = "C:/capstone/alzheimers/proposed/ensemble6/bagging/"

# Load model
model = joblib.load(MODEL_DIR + "bagging_model.joblib")
selector = joblib.load(MODEL_DIR + "bagging_selector.joblib")

# Load unseen data
REAL_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REAL_PATH)

# Apply same feature selection
X_real_sel = selector.transform(df_real)

# Predict
predictions = model.predict(X_real_sel)

df_real["Prediction"] = ["Alzheimer Risk" if p == 1 else "Healthy" for p in predictions]

df_real.to_csv(MODEL_DIR + "bagging_realtime_predictions.csv", index=False)

print("✅ Real-time Bagging prediction completed")
