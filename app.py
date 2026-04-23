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

# Logout
if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ---------------- MAIN ---------------- #
st.title("💳 Credit Default Prediction Dashboard")

dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

# ---------------- LOAD MODEL ---------------- #
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
    st.error(f"Model/Column loading error: {e}")
    st.stop()

# ---------------- INPUT UI ---------------- #
st.sidebar.header("Input Features")

payment_score = st.sidebar.slider("Payment Behavior Score", 300, 900, 700)
balance = st.sidebar.number_input("Account Balance (₹)", 0, 1000000, 40000)
days_due = st.sidebar.number_input("Days Past Due", 0, 120, 5)
risk_score = st.sidebar.slider("Risk Indicator Score", 0, 10, 3)
spending = st.sidebar.number_input("Monthly Spending (₹)", 0, 100000, 20000)
delay_count = st.sidebar.number_input("Recent Delay Count", 0, 50, 2)

# ---------------- FEATURE BUILD ---------------- #
full_input = {col: 0 for col in all_columns}

full_input["P_2"] = payment_score / 1000
full_input["B_1"] = balance / 100000
full_input["D_39"] = days_due / 100
full_input["R_1"] = risk_score / 10
full_input["S_3"] = spending / 100000
full_input["D_41"] = delay_count / 100

df = pd.DataFrame([full_input])

# ---------------- PREDICTION ---------------- #
if st.button("Predict Risk"):

    try:
        prob = model.predict_proba(df)[0][1]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

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

    # ---------------- DISPLAY ---------------- #
    if final_risk == "High Risk":
        st.error("🔴 High Risk")
    elif final_risk == "Medium Risk":
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # ---------------- SHAP ---------------- #
    st.markdown("### 🤖 Model Explanation")

    feature_map = {
        "P_2": "Payment Behavior",
        "B_1": "Account Balance",
        "D_39": "Days Past Due",
        "R_1": "Risk Score",
        "S_3": "Monthly Spending",
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

    # ---------------- DECISION ---------------- #
    st.markdown("### 📌 Suggested Decision")
    st.write(decision)

    # ---------------- NOTE ---------------- #
    st.markdown("---")
    st.markdown(
        "💡 SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency."
    )
               
    
