def get_retention_actions(row):
    actions = []

    # ✅ force scalar string
    risk = str(row["Risk_Level"])

    if risk == "High":
        actions.append("Immediate retention call from support team")
        actions.append("Offer personalized discount")

    if row["MonthlyCharges"] > 80:
        actions.append("Suggest lower-cost plan or bundle")

    if row["tenure"] < 6:
        actions.append("Provide onboarding and welcome benefits")

    if row["TechSupport"] == "No":
        actions.append("Offer free tech support trial")

    if row["Contract"] == "Month-to-month":
        actions.append("Incentivize long-term contract")

    return actions
