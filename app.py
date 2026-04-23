import streamlit as st
import pickle
import pandas as pd
import json

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- SESSION STATE ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ---------------- LOAD USERS ---------------- #
with open("users.json") as f:
    users = json.load(f)

# ---------------- TOP 6 FEATURES ---------------- #
top_features = ["P_2", "B_1", "D_39", "R_1", "S_3", "D_41"]

# ---------------- USER-FRIENDLY NAMES ---------------- #
feature_names = {
    "P_2": "💳 Payment Behavior Score",
    "B_1": "💰 Account Balance",
    "D_39": "⏳ Days Past Due",
    "R_1": "⚠️ Risk Indicator Score",
    "S_3": "🛍️ Spending Pattern",
    "D_41": "📉 Recent Delay Count"
}

# ---------------- LOGIN FUNCTION ---------------- #
def login():
    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

# ---------------- MAIN APP ---------------- #
if not st.session_state["logged_in"]:
    login()

else:
    # -------- LOGOUT -------- #
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    # -------- TITLE -------- #
    st.title("💳 Credit Default Prediction Dashboard")
    st.markdown("### Enter Customer Financial Details")

    # -------- LOAD MODEL -------- #
    model = pickle.load(open("models/amex_model.pkl", "rb"))

    # -------- SIDEBAR INPUT -------- #
    st.sidebar.header("Input Features")
    st.sidebar.info("Enter key financial indicators to predict credit risk.")

    input_data = {}

    for col in top_features:
        label = feature_names.get(col, col)

        input_data[col] = st.sidebar.number_input(
            label,
            value=0.0,
            help=f"Model Feature: {col}"
        )

    # -------- PREDICTION -------- #
    if st.button("Predict Risk"):

        df = pd.DataFrame([input_data])

        # Ensure correct feature alignment
        try:
            model_features = model.get_booster().feature_names
            df = df.reindex(columns=model_features, fill_value=0)
        except:
            pass

        prob = model.predict_proba(df)[0][1]

        # -------- RESULT DISPLAY -------- #
        st.markdown("## 📊 Prediction Result")
        st.markdown(f"### 🔢 Default Probability: **{prob:.2f}**")

        # -------- RISK CATEGORY -------- #
        if prob < 0.3:
            risk = "🟢 Low Risk"
            explanation = "Customer is financially stable with low chances of default."
            decision = "✅ Approve Loan"
            st.success(risk)

        elif prob < 0.7:
            risk = "🟡 Medium Risk"
            explanation = "Customer shows moderate risk. Further verification is recommended."
            decision = "⚠️ Review Manually"
            st.warning(risk)

        else:
            risk = "🔴 High Risk"
            explanation = "Customer has a high probability of default. Loan approval is risky."
            decision = "❌ Reject Loan"
            st.error(risk)

        # -------- INTERPRETATION -------- #
        st.markdown("### 🧠 Interpretation")
        st.markdown(f"""
        - **Risk Level:** {risk}  
        - **Meaning:** {explanation}  
        - **Model Confidence:** **{prob:.2%}**
        """)

        # -------- FINAL DECISION -------- #
        st.markdown("### 📌 Suggested Decision")
        st.markdown(f"### {decision}")
