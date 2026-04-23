# Imports
import streamlit as st
import pickle
import pandas as pd
import json

# Page config (top)
st.set_page_config(page_title="Credit Risk AI", layout="wide")

# Login logic...

if not st.session_state["logged_in"]:
    login()
else:
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

    # Main dashboard
    st.title("💳 Credit Default Prediction Dashboard")

# ---------------- LOGIN SYSTEM ---------------- #

with open("users.json") as f:
    users = json.load(f)

def login():
    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ---------------- MAIN APP ---------------- #

if not st.session_state["logged_in"]:
    login()
else:
    st.title("💳 Credit Default Prediction Dashboard")

    # Load model
    model = pickle.load(open("models/amex_model.pkl", "rb"))

    # Load columns
    with open("columns/amex_columns.json") as f:
        columns = json.load(f)

    st.sidebar.header("Enter Customer Details")

    input_data = {}

    # Dynamic input fields
    for col in columns[:10]:   # show first 10 for simplicity
        input_data[col] = st.sidebar.number_input(col, value=0.0)

    if st.button("Predict Risk"):
        df = pd.DataFrame([input_data])

        prob = model.predict_proba(df)[0][1]

        st.subheader(f"Default Probability: {prob:.2f}")

        if prob < 0.3:
            st.success("Low Risk")
        elif prob < 0.7:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")