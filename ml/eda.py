import matplotlib
matplotlib.use("Agg")  # Non-GUI backend (server safe)

import matplotlib.pyplot as plt
import pandas as pd
import os

import pandas as pd
import os

def generate_eda(df, important_features, out_dir="static/plots"):

    
    os.makedirs(out_dir, exist_ok=True)
    # Clear old plots
    for file in os.listdir(out_dir):
     if file.endswith(".png"):
        os.remove(os.path.join(out_dir, file))

    # ---------------------------
    # 1️⃣ BASIC DISTRIBUTIONS
    # ---------------------------
    
    
    numeric_cols = [
    f for f in important_features
    if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]
    print(numeric_cols)

    for col in numeric_cols:
      if col in df.columns:
        # Convert to numeric, coerce errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in numeric_cols:
        if col in df.columns:
            plt.figure()
            df[col].hist(bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.savefig(f"{out_dir}/{col}_distribution.png")
            plt.close()

    # ---------------------------
    # 2️⃣ TARGET DISTRIBUTION
    # ---------------------------
    if "Churn" in df.columns:
        plt.figure()
        df["Churn"].value_counts().plot(kind="bar")
        plt.title("Churn Distribution")
        plt.xlabel("Churn")
        plt.ylabel("Count")
        plt.savefig(f"{out_dir}/churn_distribution.png")
        plt.close()

    # ---------------------------
    # 3️⃣ NUMERIC vs CHURN
    # ---------------------------
    if "Churn" in df.columns:
        df_plot = df.copy()
        df_plot["Churn_num"] = df_plot["Churn"].map({"No": 0, "Yes": 1})
        for col in numeric_cols:
            if col in df_plot.columns:
                plt.figure()
                df_plot.boxplot(column=col, by="Churn_num")
                plt.title(f"{col} vs Churn")
                plt.suptitle("")
                plt.xlabel("Churn (0 = No, 1 = Yes)")
                plt.ylabel(col)
                plt.savefig(f"{out_dir}/{col}_vs_churn.png")
                plt.close()
    #Pandas automatically adds a default “super title”:
    #“Boxplot grouped by Churn”
    #So you end up with:
    #A big ugly top title
    #Your own title overlapping below it
    #plt.suptitle("") means:
    #“Remove the automatic super-title”
    # ---------------------------
    # 4️⃣ CATEGORICAL vs CHURN
    # ---------------------------
    if "Churn" in df.columns:
        cat_cols = [
            f for f in important_features
            if f in df.columns and df[f].dtype == "object"
        ]


        for col in cat_cols:
            plt.figure()
            pd.crosstab(df[col], df["Churn"]).plot(kind="bar", stacked=True)#crosstab for cat clmns to cal freq
            plt.title(f"{col} vs Churn")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()#no overlapping
            plt.savefig(f"{out_dir}/{col}_vs_churn.png")
            plt.close()
   #eda can handle cat clmns