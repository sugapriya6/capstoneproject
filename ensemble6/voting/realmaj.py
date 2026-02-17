import pandas as pd
import numpy as np
import joblib

MODEL_DIR = "C:/capstone/alzheimers/proposed/ensemble6/voting/"

# ===== LOAD MODELS =====
models = {
    "et": joblib.load(MODEL_DIR + "et_model.joblib"),
    "rf": joblib.load(MODEL_DIR + "rf_model.joblib"),
    "lr": joblib.load(MODEL_DIR + "lr_model.joblib"),
    "xgb": joblib.load(MODEL_DIR + "xgb_model.joblib"),
    "gnb": joblib.load(MODEL_DIR + "gnb_model.joblib"),
    "svm": joblib.load(MODEL_DIR + "svm_model.joblib"),
    "mlp": joblib.load(MODEL_DIR + "mlp_model.joblib"),
}

selectors = {
    "et": joblib.load(MODEL_DIR + "et_selector.joblib"),
    "rf": joblib.load(MODEL_DIR + "rf_selector.joblib"),
    "lr": joblib.load(MODEL_DIR + "lr_selector.joblib"),
    "xgb": joblib.load(MODEL_DIR + "xgb_selector.joblib"),
    "gnb": joblib.load(MODEL_DIR + "gnb_selector.joblib"),
    "svm": joblib.load(MODEL_DIR + "svm_selector.joblib"),
    "mlp": joblib.load(MODEL_DIR + "mlp_selector.joblib"),
}

# ===== LOAD REAL‑TIME DATA =====
REALTIME_PATH = "C:/capstone/alzheimers/proposed/preprocess1/data_test_processed.csv"
df_real = pd.read_csv(REALTIME_PATH)

# ===== ENSEMBLE PREDICTIONS =====
all_predictions = []

for key in models:
    X_real_sel = selectors[key].transform(df_real)
    preds = models[key].predict(X_real_sel)
    all_predictions.append(preds)

# ===== MAJORITY VOTING =====
all_predictions = np.array(all_predictions)

final_predictions = np.apply_along_axis(
    lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_predictions
)

# ===== OUTPUT =====
output = pd.DataFrame({
    "Sample_ID": range(1, len(final_predictions) + 1),
    "Prediction": ["Alzheimer Risk" if p == 1 else "Healthy" for p in final_predictions]
})

output.to_csv(MODEL_DIR + "ensemble_realtime_predictions.csv", index=False)

print("✅ Real‑time ensemble prediction completed")
print(output)
