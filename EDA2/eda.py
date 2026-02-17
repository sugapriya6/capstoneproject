import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1️⃣ Null value handling
# --------------------------
def handle_nulls(df, phase="train", mode_file="mode_values.pkl"):
    null_columns = df.columns[df.isnull().any()]
    if phase == "train":
        mode_values = {}
        for column in null_columns:
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)
            mode_values[column] = mode_val
        with open(mode_file, 'wb') as f:
            pickle.dump(mode_values, f)
    else:
        with open(mode_file, 'rb') as f:
            mode_values = pickle.load(f)
        for column in null_columns:
            if column in df.columns:
                df[column].fillna(mode_values[column], inplace=True)
    return df

# --------------------------
# 2️⃣ Outlier handling
# --------------------------
def handle_outliers(df, phase="train", stats_file="outlier_stats.pkl"):
    numeric_columns = df.select_dtypes(include=np.number).columns
    if phase == "train":
        outlier_stats = {}
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[column] = np.clip(df[column], lower, upper)
            outlier_stats[column] = (lower, upper)
        with open(stats_file, 'wb') as f:
            pickle.dump(outlier_stats, f)
    else:
        with open(stats_file, 'rb') as f:
            outlier_stats = pickle.load(f)
        for column in numeric_columns:
            if column in outlier_stats:
                lower, upper = outlier_stats[column]
                df[column] = np.clip(df[column], lower, upper)
    return df

# --------------------------
# 3️⃣ Normalization (✅ FIXED HERE ONLY)
# --------------------------
def normalize_data(df, phase="train",
                   scaler_file="scaler.pkl",
                   feature_file="scaler_features.pkl"):

    numeric_columns = df.select_dtypes(include=np.number).columns

    if phase == "train":
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        # ✅ Save scaler AND column names
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

        with open(feature_file, 'wb') as f:
            pickle.dump(list(numeric_columns), f)

    else:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        with open(feature_file, 'rb') as f:
            train_features = pickle.load(f)

        # ✅ Add missing columns
        for col in train_features:
            if col not in df.columns:
                df[col] = 0

        # ✅ Keep only train columns & same order
        df = df[train_features]

        df[train_features] = scaler.transform(df[train_features])

    return df

# --------------------------
# 4️⃣ Visualization (unchanged logic)
# --------------------------
def visualize(df, title_prefix=""):
    # Remove constant columns ONLY for visualization
    df_vis = df.loc[:, df.nunique() > 1]

    # BIGGER FIGURE
    plt.figure(figsize=(20, 8))

    # ------------------
    # Boxplot
    # ------------------
    plt.subplot(1, 2, 1)
    plt.title(f"{title_prefix} - Boxplot (First 18 Columns)")
    df_vis.iloc[:, :18].boxplot()
    plt.xticks(rotation=90)
    plt.grid(True)

    # ------------------
    # Heatmap
    # ------------------
    plt.subplot(1, 2, 2)
    plt.title(f"{title_prefix} - Correlation Heatmap")

    sns.heatmap(
        df_vis.iloc[:, :18].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=1,          # clear cell borders
        square=True,           # equal box size
        annot_kws={"size": 8}  # smaller numbers
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # ------------------
    # SAVE IMAGE
    # ------------------
    filename = title_prefix.replace(" ", "_").replace("-", "").lower() + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()




# --------------------------
# 5️⃣ Full pipeline
# --------------------------
def preprocess_outliers(input_file, output_file, phase="train"):
    df = pd.read_csv(input_file)
    last_col = df.iloc[:, -1]
    df = df.iloc[:, 1:-1]

    df = handle_nulls(df, phase=phase)
    df = handle_outliers(df, phase=phase)

    visualize(df, title_prefix=f"{phase.capitalize()} - After Outlier Handling")

    df['LastColumn'] = last_col
    df.to_csv(output_file, index=False)

    print(f"{phase.capitalize()} outlier preprocessing done. Saved to {output_file}")

def preprocess_normalization(input_file, output_file, phase="train"):
    df = pd.read_csv(input_file)
    last_col = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    df = normalize_data(df, phase=phase)

    visualize(df, title_prefix=f"{phase.capitalize()} - After Normalization")

    df['LastColumn'] = last_col
    df.to_csv(output_file, index=False)

    print(f"{phase.capitalize()} normalization done. Saved to {output_file}")

# --------------------------
# RUN
# --------------------------
preprocess_outliers("content/data.csv", "cleaned_train.csv", phase="train")
preprocess_normalization("cleaned_train.csv", "normalized_train.csv", phase="train")

preprocess_outliers("content/data_test.csv", "cleaned_test.csv", phase="test")
preprocess_normalization("cleaned_test.csv", "normalized_test.csv", phase="test")
