import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


def preprocess_user_data(df):
    # EXACT steps from notebook
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    # Categorical handling
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 🔥 ALIGN COLUMNS
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric columns
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df
