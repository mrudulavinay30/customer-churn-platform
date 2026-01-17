import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("C:/customer-churn-platform/data/raw_churn.csv")

# -----------------------------
# 2️⃣ Preprocessing (SAME AS YOUR NOTEBOOK)
# -----------------------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode target
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Categorical → One-hot
cat_cols = X.select_dtypes(include="object").columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# -----------------------------
# 3️⃣ Train–Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# 4️⃣ Scaling (ONLY numeric)
# -----------------------------
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -----------------------------
# 5️⃣ Train Models
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Save artifacts
# -----------------------------
joblib.dump(log_model, "log_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# 🔥 VERY IMPORTANT
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

# Business-optimized threshold (from your notebook)
best_threshold = 0.37
joblib.dump(best_threshold, "best_threshold.pkl")

print("✅ Training complete. All artifacts saved.")
