import streamlit as st
import pickle
import pandas as pd
import json
import shap

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

with open("users.json") as f:
    users = json.load(f)

def login():
    st.title("🔐 Login Page")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == p:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state["logged_in"]:
    login()
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ---------------- MAIN ---------------- #
st.title("💳 Credit Default Prediction Dashboard")

dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

# ---------------- LOAD MODEL + COLUMNS ---------------- #
try:
    if dataset == "AMEX":
        model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        with open("columns/amex_columns.json") as f:
            all_columns = json.load(f)
    else:
        model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
        with open("columns/gmsc_columns.json") as f:
            all_columns = json.load(f)
except Exception as e:
    st.error(f"Error loading model/columns: {e}")
    st.stop()

# ---------------- INPUT UI ---------------- #
st.sidebar.header("Input Features")

if dataset == "AMEX":
    payment_score = st.sidebar.slider("Payment Behavior Score", 300, 900, 700)
    balance = st.sidebar.number_input("Account Balance (₹)", 0, 1000000, 40000)
    days_due = st.sidebar.number_input("Days Past Due", 0, 120, 5)
    risk_score = st.sidebar.slider("Risk Indicator Score", 0, 10, 3)
    spending = st.sidebar.number_input("Monthly Spending (₹)", 0, 100000, 20000)
    delay_count = st.sidebar.number_input("Recent Delay Count", 0, 50, 2)

    user_features = ["P_2", "B_1", "D_39", "R_1", "S_3", "D_41"]

elif dataset == "GMSC":
    utilization = st.sidebar.slider("Credit Utilization (%)", 0.0, 1.0, 0.3)
    age = st.sidebar.slider("Age", 18, 80, 30)
    past_due = st.sidebar.number_input("30-59 Days Past Due", 0, 10, 1)
    debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.5)
    income = st.sidebar.number_input("Monthly Income (₹)", 0, 1000000, 50000)
    open_credit = st.sidebar.number_input("Open Credit Lines", 0, 20, 5)

    user_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans"
    ]

# ---------------- BUILD FULL INPUT ---------------- #
full_input = {col: 0 for col in all_columns}

if dataset == "AMEX":
    full_input["P_2"] = payment_score / 1000
    full_input["B_1"] = balance / 100000
    full_input["D_39"] = days_due / 100
    full_input["R_1"] = risk_score / 10
    full_input["S_3"] = spending / 100000
    full_input["D_41"] = delay_count / 100

elif dataset == "GMSC":
    full_input["RevolvingUtilizationOfUnsecuredLines"] = utilization
    full_input["age"] = age
    full_input["NumberOfTime30-59DaysPastDueNotWorse"] = past_due
    full_input["DebtRatio"] = debt_ratio
    full_input["MonthlyIncome"] = income
    full_input["NumberOfOpenCreditLinesAndLoans"] = open_credit

df = pd.DataFrame([full_input])

# ---------------- PREDICTION ---------------- #
if st.button("Predict Risk"):

    prob = model.predict_proba(df)[0][1]

    st.markdown("## 📊 Prediction Result")
    st.write(f"Default Probability: {prob:.2f}")
    st.progress(int(prob * 100))

    # MODEL RISK
    if prob < 0.3:
        model_risk = "Low Risk"
    elif prob < 0.7:
        model_risk = "Medium Risk"
    else:
        model_risk = "High Risk"

    # ---------------- BUSINESS RULES ---------------- #
    rule_triggered = False
    rule_reasons = []

    if dataset == "AMEX":
        if days_due > 60:
            rule_triggered = True
            rule_reasons.append("Very high overdue days")
        if delay_count > 10:
            rule_triggered = True
            rule_reasons.append("Frequent payment delays")
        if payment_score < 400:
            rule_triggered = True
            rule_reasons.append("Poor payment behavior")

    elif dataset == "GMSC":
        if utilization > 0.8:
            rule_triggered = True
            rule_reasons.append("High credit utilization")
        if past_due > 3:
            rule_triggered = True
            rule_reasons.append("Frequent past dues")
        if debt_ratio > 1.5:
            rule_triggered = True
            rule_reasons.append("High debt ratio")

    # FINAL DECISION
    if rule_triggered:
        final_risk = "High Risk"
        decision = "❌ Reject Loan"
    else:
        final_risk = model_risk
        decision = (
            "✅ Approve Loan" if model_risk == "Low Risk"
            else "⚠️ Review Manually" if model_risk == "Medium Risk"
            else "❌ Reject Loan"
        )

    # DISPLAY
    if final_risk == "High Risk":
        st.error("🔴 High Risk")
    elif final_risk == "Medium Risk":
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # ---------------- SHAP ---------------- #
    st.markdown("### 🤖 Model Explanation")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        values = shap_values[0] if isinstance(shap_values, list) else shap_values

        shap_df = pd.DataFrame({
            "feature": df.columns,
            "impact": values[0]
        })

        shap_df["abs"] = shap_df["impact"].abs()

        # 👉 ONLY show user features (fix D_42 issue)
        shap_df = shap_df[shap_df["feature"].isin(user_features)]

        top = shap_df.sort_values("abs", ascending=False).head(3)

        model_reasons = []

        for _, row in top.iterrows():
            if row["impact"] > 0:
                st.write(f"🔺 {row['feature']} increased risk")
                model_reasons.append(f"{row['feature']} increased risk")
            else:
                st.write(f"🔻 {row['feature']} reduced risk")
                model_reasons.append(f"{row['feature']} reduced risk")

    except:
        st.info("Model explanation unavailable")
        model_reasons = []

    # ---------------- BUSINESS EXPLANATION ---------------- #
    if rule_triggered:
        st.markdown("### ⚠️ Business Rule Explanation")
        for r in rule_reasons:
            st.write(f"🔴 {r}")

    # ---------------- FINAL INTERPRETATION ---------------- #
    st.markdown("### 🧠 Final Interpretation")

    for r in model_reasons:
        st.write(f"• {r}")
    for r in rule_reasons:
        st.write(f"• {r}")

    # ---------------- DECISION ---------------- #
    st.markdown("### 📌 Suggested Decision")
    st.write(decision)

    # ---------------- FINAL NOTE ---------------- #
    st.markdown("---")
    st.markdown(
        "💡 SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency."
    )
   
