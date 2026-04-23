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

# ---------------- LOAD ALL MODEL FEATURES ---------------- #
with open("columns/amex_columns.json") as f:
    all_columns = json.load(f)

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

    # -------- USER-FRIENDLY INPUTS -------- #
    st.sidebar.header("Input Features")
    st.sidebar.info("Enter real-world financial details")

    input_data = {}

    input_data["P_2"] = st.sidebar.slider("💳 Payment Behavior Score", 300, 900, 700)
    input_data["B_1"] = st.sidebar.number_input("💰 Account Balance (₹)", 0, 1000000, 40000)
    input_data["D_39"] = st.sidebar.number_input("⏳ Days Past Due", 0, 100, 5)
    input_data["R_1"] = st.sidebar.slider("⚠️ Risk Indicator Score", 0, 10, 3)
    input_data["S_3"] = st.sidebar.number_input("🛍️ Monthly Spending (₹)", 0, 100000, 20000)
    input_data["D_41"] = st.sidebar.number_input("📉 Recent Delay Count", 0, 50, 2)

    # -------- PREDICTION -------- #
    if st.button("Predict Risk"):

        # -------- SCALING (VERY IMPORTANT) -------- #
        scaled_data = {}
        scaled_data["P_2"] = input_data["P_2"] / 1000
        scaled_data["B_1"] = input_data["B_1"] / 100000
        scaled_data["D_39"] = input_data["D_39"] / 100
        scaled_data["R_1"] = input_data["R_1"] / 10
        scaled_data["S_3"] = input_data["S_3"] / 100000
        scaled_data["D_41"] = input_data["D_41"] / 100

        # -------- CREATE FULL FEATURE SET -------- #
        full_input = {col: 0 for col in all_columns}

        for col in scaled_data:
            full_input[col] = scaled_data[col]

        df = pd.DataFrame([full_input])

        # -------- MODEL PREDICTION -------- #
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
