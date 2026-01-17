from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

from ml.cost_analysis import threshold_cost_curve
from ml.preprocessing import preprocess_user_data
from ml.eda import generate_eda

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


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    threshold = float(request.form["threshold"])

    # Save uploaded CSV
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path)

    # ✅ Generate EDA plots
    generate_eda(df, out_dir=PLOTS_FOLDER)

    # ✅ Collect plot paths for Jinja
    plot_files = [
        f"plots/{fname}"
        for fname in os.listdir(PLOTS_FOLDER)
        if fname.endswith(".png")
    ]

    # Preprocess
    X_processed = preprocess_user_data(df)

    # Probabilities
    df["Logistic_Prob"] = log_model.predict_proba(X_processed)[:, 1]
    df["RF_Prob"] = rf_model.predict_proba(X_processed)[:, 1]

    # Apply user threshold
    df["Logistic_Pred"] = (df["Logistic_Prob"] >= threshold).astype(int)
    df["RF_Pred"] = (df["RF_Prob"] >= threshold).astype(int)

    # Auto threshold (only if churn labels exist)
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
        threshold=threshold,
        best_threshold=best_threshold,
        plot_files=plot_files
    )


if __name__ == "__main__":
    app.run(debug=True, port=5003,use_reloader=True)
