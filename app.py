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

# ---------------- FEATURE NAME MAPPING ---------------- #
feature_names = {
    "P_2": "Payment Behavior Score",
    "D_39": "Days Past Due (Delay)",
    "B_1": "Account Balance Level",
    "B_2": "Credit Utilization Ratio",
    "R_1": "Risk Indicator Score"
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

    # -------- LOAD COLUMNS -------- #
    with open("columns/amex_columns.json") as f:
        columns = json.load(f)

    # -------- SIDEBAR INPUT -------- #
    st.sidebar.header("Input Features")
    st.sidebar.info("Enter customer financial indicators to predict risk.")

    input_data = {}

    # Show only first 10 features (can increase later)
    for col in columns[:10]:
        label = feature_names.get(col, col)
        input_data[col] = st.sidebar.number_input(
            label,
            value=0.0,
            help=f"Original Feature: {col}"
        )

    # -------- PREDICTION -------- #
    if st.button("Predict Risk"):
        df = pd.DataFrame([input_data])

        prob = model.predict_proba(df)[0][1]

        st.subheader(f"📊 Default Probability: {prob:.2f}")

        # -------- RISK CATEGORY -------- #
        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.7:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")
            
    
