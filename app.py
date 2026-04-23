import streamlit as st
import pickle
import pandas as pd
import json
import shap

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- LOGIN SYSTEM ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

with open("users.json") as f:
    users = json.load(f)

def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- MAIN APP ---------------- #
if not st.session_state["logged_in"]:
    login()
    st.stop()

# Logout
if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

st.title("💳 Credit Default Prediction Dashboard")

# ---------------- DATASET SELECT ---------------- #
dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

# ---------------- LOAD MODEL SAFELY ---------------- #
try:
    if dataset == "AMEX":
        model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
    else:
        model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- USER INPUT ---------------- #
st.sidebar.header("Input Features")

payment_score = st.sidebar.slider("Payment Behavior Score", 300, 900, 700)
balance = st.sidebar.number_input("Account Balance (₹)", 0, 1000000, 40000)
days_due = st.sidebar.number_input("Days Past Due", 0, 120, 5)
risk_score = st.sidebar.slider("Risk Indicator Score", 0, 10, 3)
spending = st.sidebar.number_input("Monthly Spending (₹)", 0, 100000, 20000)
delay_count = st.sidebar.number_input("Recent Delay Count", 0, 50, 2)

# ---------------- FEATURE MAP ---------------- #
input_data = {
    "P_2": payment_score / 1000,
    "B_1": balance / 100000,
    "D_39": days_due / 100,
    "R_1": risk_score / 10,
    "S_3": spending / 100000,
    "D_41": delay_count / 100
}

# ---------------- CREATE DF SAFELY ---------------- #
df = pd.DataFrame([input_data])

# ---------------- PREDICTION ---------------- #
if st.button("Predict Risk"):

    try:
        prob = model.predict_proba(df)[0][1]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    st.markdown("## 📊 Prediction Result")
    st.write(f"Default Probability: {prob:.2f}")

    # ---------------- MODEL RISK ---------------- #
    if prob < 0.3:
        model_risk = "Low Risk"
    elif prob < 0.7:
        model_risk = "Medium Risk"
    else:
        model_risk = "High Risk"

    # ---------------- BUSINESS RULES ---------------- #
    rule_triggered = False
    rule_reasons = []

    if days_due > 60:
        rule_triggered = True
        rule_reasons.append("High overdue days")

    if delay_count > 10:
        rule_triggered = True
        rule_reasons.append("Frequent payment delays")

    if payment_score < 400:
        rule_triggered = True
        rule_reasons.append("Poor payment behavior")

    # ---------------- FINAL DECISION ---------------- #
    if rule_triggered:
        final_risk = "High Risk"
        decision = "❌ Reject Loan"
    else:
        final_risk = model_risk
        if model_risk == "Low Risk":
            decision = "✅ Approve Loan"
        elif model_risk == "Medium Risk":
            decision = "⚠️ Review Manually"
        else:
            decision = "❌ Reject Loan"

    # ---------------- DISPLAY RESULT ---------------- #
    if final_risk == "High Risk":
        st.error("🔴 High Risk")
    elif final_risk == "Medium Risk":
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # ---------------- SHAP EXPLANATION ---------------- #
    st.markdown("### 🤖 Model Explanation")

    feature_map = {
        "P_2": "Payment Behavior",
        "B_1": "Balance",
        "D_39": "Days Due",
        "R_1": "Risk Score",
        "S_3": "Spending",
        "D_41": "Delay Count"
    }

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        values = shap_values[0] if isinstance(shap_values, list) else shap_values

        shap_df = pd.DataFrame({
            "feature": df.columns,
            "impact": values[0]
        })

        shap_df["abs"] = shap_df["impact"].abs()
        top = shap_df.sort_values("abs", ascending=False).head(3)

        model_reasons = []

        for _, row in top.iterrows():
            name = feature_map.get(row["feature"], row["feature"])

            if row["impact"] > 0:
                st.write(f"🔺 {name} increased risk")
                model_reasons.append(f"{name} increased risk")
            else:
                st.write(f"🔻 {name} reduced risk")
                model_reasons.append(f"{name} reduced risk")

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

    if final_risk == "High Risk":
        st.write("Customer has high risk due to:")
    elif final_risk == "Medium Risk":
        st.write("Customer has moderate risk due to:")
    else:
        st.write("Customer is low risk due to:")

    for r in model_reasons:
        st.write(f"• {r}")

    for r in rule_reasons:
        st.write(f"• {r}")

    # ---------------- FINAL DECISION ---------------- #
    st.markdown("### 📌 Suggested Decision")
    st.write(decision)

    # ---------------- TRANSPARENCY LINE ---------------- #
    st.markdown("---")
    st.markdown(
        "💡 SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency."
    )
        
