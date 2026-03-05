import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --------------------------
# CONFIG — FIXED PATHS
# --------------------------
BASE = r"D:\capstone final project\capstoneproject\preprocess1"
MODE_FILE    = BASE + r"\mode_values.pkl"
OUTLIER_FILE = BASE + r"\outlier_stats.pkl"
SCALER_FILE  = BASE + r"\scaler.pkl"
CORR_FILE    = BASE + r"\high_corr.pkl"
CORR_THRESHOLD = 0.9

# --------------------------
# 1️⃣ Handle missing values
# --------------------------
def handle_nulls(df, phase):
    if phase == "train":
        mode_values = {}
        for col in df.columns:
            if df[col].isnull().any():
                mode = df[col].mode()[0]
                df[col].fillna(mode, inplace=True)
                mode_values[col] = mode
        with open(MODE_FILE, "wb") as f:
            pickle.dump(mode_values, f)
    else:
        with open(MODE_FILE, "rb") as f:
            mode_values = pickle.load(f)
        for col, mode in mode_values.items():
            if col in df.columns:
                df[col].fillna(mode, inplace=True)
    return df

# --------------------------
# 2️⃣ Handle outliers (IQR)
# --------------------------
def handle_outliers(df, phase):
    numeric_cols = df.select_dtypes(include=np.number).columns

    if phase == "train":
        stats = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)
            stats[col] = (lower, upper)
        with open(OUTLIER_FILE, "wb") as f:
            pickle.dump(stats, f)
    else:
        with open(OUTLIER_FILE, "rb") as f:
            stats = pickle.load(f)
        for col, (lower, upper) in stats.items():
            if col in df.columns:
                df[col] = np.clip(df[col], lower, upper)
    return df

# --------------------------
# 3️⃣ Feature scaling
# --------------------------
def scale_features(df, phase, has_class):
    if has_class:
        X = df.drop(columns=["class"])
    else:
        X = df.copy()

    if phase == "train":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    if has_class:
        X_scaled["class"] = df["class"].values

    return X_scaled

# --------------------------
# 4️⃣ Remove highly correlated features
# --------------------------
def remove_correlated(df, phase, has_class):
    if has_class:
        X = df.drop(columns=["class"])
    else:
        X = df.copy()

    if phase == "train":
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
        X = X.drop(columns=to_drop)
        with open(CORR_FILE, "wb") as f:
            pickle.dump(to_drop, f)
    else:
        with open(CORR_FILE, "rb") as f:
            to_drop = pickle.load(f)
        X = X.drop(columns=[c for c in to_drop if c in X.columns])

    if has_class:
        X["class"] = df["class"].values

    return X

# --------------------------
# 5️⃣ Main pipeline
# --------------------------
def preprocess_data(input_file, output_file, phase):
    df = pd.read_csv(input_file)
    df = df.iloc[:, 1:]          # remove id column
    has_class = "class" in df.columns

    df = handle_nulls(df, phase)
    df = handle_outliers(df, phase)
    df = scale_features(df, phase, has_class)
    df = remove_correlated(df, phase, has_class)

    df.to_csv(output_file, index=False)

    print(f"✅ {phase.upper()} preprocessing done")
    print(f"📁 Saved as: {output_file}")
    print(f"📐 Shape: {df.shape}")

# --------------------------
# RUN
# --------------------------
BASE_DATA = r"D:\capstone final project\capstoneproject\preprocess1"

# CHANGE TO WHEREVER YOUR data.csv ACTUALLY IS:
preprocess_data(
    r"D:\capstone final project\capstoneproject\preprocess1\content\data.csv",
    BASE_DATA + r"\data_train_processed.csv",
    phase="train"
)

preprocess_data(
    r"D:\capstone final project\capstoneproject\preprocess1\content\data_test.csv",
    BASE_DATA + r"\data_test_processed.csv",
    phase="test"
)