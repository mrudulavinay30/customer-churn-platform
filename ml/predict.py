import pandas as pd
import joblib

from ml.preprocessing import preprocess_user_data

# -----------------------------
# 1️⃣ Load artifacts
# -----------------------------
log_model = joblib.load("log_model.pkl")
rf_model = joblib.load("rf_model.pkl")
best_threshold = joblib.load("best_threshold.pkl")

# -----------------------------
# 2️⃣ Load user data
# -----------------------------
df = pd.read_csv("sample_input.csv")

# -----------------------------
# 3️⃣ Preprocess (YOUR FUNCTION)
# -----------------------------
X_processed = preprocess_user_data(df)

# -----------------------------
# 4️⃣ Logistic Regression Predictions
# -----------------------------
df["Logistic_Prob"] = log_model.predict_proba(X_processed)[:, 1]
df["Logistic_Pred"] = (df["Logistic_Prob"] >= best_threshold).astype(int)

# -----------------------------
# 5️⃣ Random Forest Predictions
# -----------------------------
df["RF_Prob"] = rf_model.predict_proba(X_processed)[:, 1]
df["RF_Pred"] = (df["RF_Prob"] >= best_threshold).astype(int)

# -----------------------------
# 6️⃣ Save output
# -----------------------------
df.to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv")
print(f"✅ Business threshold used: {best_threshold}")
