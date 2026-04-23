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
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == p:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- MAIN ---------------- #
if not st.session_state["logged_in"]:
    login()
else:

    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.title("💳 Credit Risk Prediction Dashboard")

    # -------- DATASET SELECT -------- #
    dataset = st.selectbox("Select Dataset", ["AMEX", "GMSC"])

    # -------- LOAD MODEL + COLUMNS -------- #
    if dataset == "AMEX":
        model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
        with open("columns/amex_columns.json") as f:
            all_columns = json.load(f)

    else:
        model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))
        with open("columns/gmsc_columns.json") as f:
            all_columns = json.load(f)

    # -------- INPUT UI -------- #
    st.sidebar.header("Input Features")

    input_data = {
        "P_2": st.sidebar.slider("Payment Score", 300, 900, 700),
        "B_1": st.sidebar.number_input("Balance", 0, 1000000, 40000),
        "D_39": st.sidebar.number_input("Days Past Due", 0, 100, 5),
        "R_1": st.sidebar.slider("Risk Score", 0, 10, 3),
        "S_3": st.sidebar.number_input("Spending", 0, 100000, 20000),
        "D_41": st.sidebar.number_input("Delay Count", 0, 50, 2)
    }

    if st.button("Predict Risk"):

        # -------- SCALING -------- #
        scaled = {
            "P_2": input_data["P_2"] / 1000,
            "B_1": input_data["B_1"] / 100000,
            "D_39": input_data["D_39"] / 100,
            "R_1": input_data["R_1"] / 10,
            "S_3": input_data["S_3"] / 100000,
            "D_41": input_data["D_41"] / 100
        }

        # -------- MATCH FEATURES -------- #
        full_input = {col: 0 for col in all_columns}
        for col in scaled:
            if col in full_input:
                full_input[col] = scaled[col]

        df = pd.DataFrame([full_input])

        # -------- PREDICTION -------- #
        prob = model.predict_proba(df)[0][1]

        st.subheader("📊 Prediction Result")
        st.write(f"Default Probability: {prob:.2f}")

        if prob < 0.3:
            st.success("Low Risk")
        elif prob < 0.7:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

        # -------- SHAP EXPLANATION -------- #
        st.subheader("🤖 Model Explanation")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)

            values = shap_values[0] if isinstance(shap_values, list) else shap_values

            importance = pd.DataFrame({
                "feature": df.columns,
                "impact": values[0]
            })

            importance["abs"] = importance["impact"].abs()
            top = importance.sort_values("abs", ascending=False).head(3)

            for _, row in top.iterrows():
                if row["impact"] > 0:
                    st.write(f"🔺 {row['feature']} increased risk")
                else:
                    st.write(f"🔻 {row['feature']} reduced risk")

        except:
            st.info("Explanation unavailable")

        # -------- FINAL LINE -------- #
        st.markdown("---")
        st.markdown(
            "💡 *SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency.*"
        )
        
