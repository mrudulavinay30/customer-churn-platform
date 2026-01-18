from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from ml.cost_analysis import threshold_cost_curve
from ml.preprocessing import preprocess_user_data
from ml.eda import generate_eda
from ml.feature_importance import get_feature_importance

feature_names = joblib.load("ml/feature_columns.pkl")


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOTS_FOLDER = "static/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

log_model = joblib.load("ml/log_model.pkl")
rf_model = joblib.load("ml/rf_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/risk")
def risk():
    df = pd.read_csv("static/results.csv")
    table_data = df.to_dict(orient="records")
    return render_template("risk.html", table_data=table_data)
 
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    threshold = float(request.form["threshold"])

    # Save CSV
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path)

    # -----------------------------
    # 1️⃣ FEATURE IMPORTANCE FIRST
    # -----------------------------
    rf_importance = get_feature_importance(
        rf_model,
        feature_names,
        top_n=10
    )

    importance_data = rf_importance.to_dict(orient="records")

 # 🔥 MAP ENCODED FEATURES → RAW FEATURES
    raw_important_features = set()

    for feature in rf_importance["feature"]:
     raw_feature = feature.split("_")[0]  # e.g. Contract_Two year → Contract
     if raw_feature in df.columns:
        raw_important_features.add(raw_feature)

    important_features = list(raw_important_features)


    # -----------------------------
    # 2️⃣ GENERATE EDA (IMPORTANT ONLY)
    # -----------------------------
    generate_eda(
        df,
        important_features=important_features,
        out_dir=PLOTS_FOLDER
    )

    # -----------------------------
    # 3️⃣ COLLECT PLOTS FOR JINJA
    # -----------------------------
    plot_files = [
        f"plots/{fname}"
        for fname in os.listdir(PLOTS_FOLDER)
        if fname.endswith(".png")
    ]

    # -----------------------------
    # 4️⃣ PREDICTIONS
    # -----------------------------
    X_processed = preprocess_user_data(df)

    df["Logistic_Prob"] = log_model.predict_proba(X_processed)[:, 1]
    df["RF_Prob"] = rf_model.predict_proba(X_processed)[:, 1]

    df["Logistic_Pred"] = (df["Logistic_Prob"] >= threshold).astype(int)
    df["RF_Pred"] = (df["RF_Prob"] >= threshold).astype(int)


    df["Risk_Level"] = pd.cut(
    df["RF_Prob"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
    )
    table_data = df.to_dict(orient="records")

    # -----------------------------
    # 5️⃣ COST-OPTIMAL THRESHOLD
    # -----------------------------
    if "Churn" in df.columns:
        y_true = df["Churn"].map({"Yes": 1, "No": 0}).values
        y_prob = df["Logistic_Prob"].values

        best_threshold, _, _ = threshold_cost_curve(
            y_true=y_true,
            y_prob=y_prob,
            cost_fn=500,
            cost_fp=50
        )
    else:
        best_threshold = threshold

    df.to_csv("static/results.csv", index=False)

    return render_template(
        "results.html",
        table_data=table_data,
        threshold=threshold,
        best_threshold=best_threshold,
        plot_files=plot_files,
        importance_data=importance_data
    )




if __name__ == "__main__":
    app.run(debug=True, port=5003,use_reloader=True)
