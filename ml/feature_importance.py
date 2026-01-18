import pandas as pd

def get_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    fi_df = fi_df.sort_values(by="importance", ascending=False)

    

    return fi_df.head(top_n)
