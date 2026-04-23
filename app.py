import streamlit as st
import pickle
import pandas as pd
import json
import shap
import numpy as np

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Credit Risk AI", layout="wide")

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ---------------- LOAD USERS ---------------- #
with open("users.json") as f:
    users = json.load(f)

# ---------------- LOAD FEATURE COLUMNS ---------------- #
with open("columns/amex_columns.json") as f:
    all_columns = json.load(f)

# ---------------- LOGIN ---------------- #
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
    # Logout
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.title("💳 Credit Default Prediction Dashboard")
    st.markdown("### Enter Customer Financial Details")

    # Load model
    model = pickle.load(open("models/amex_model.pkl", "rb"))

    # ---------------- INPUT ---------------- #
    st.sidebar.header("Input Features")
    st.sidebar.info("Enter real-world financial details")

    input_data = {
        "P_2": st.sidebar.slider("💳 Payment Behavior Score", 300, 900, 700),
        "B_1": st.sidebar.number_input("💰 Account Balance (₹)", 0, 1000000, 40000),
        "D_39": st.sidebar.number_input("⏳ Days Past Due", 0, 100, 5),
        "R_1": st.sidebar.slider("⚠️ Risk Indicator Score", 0, 10, 3),
        "S_3": st.sidebar.number_input("🛍️ Monthly Spending (₹)", 0, 100000, 20000),
        "D_41": st.sidebar.number_input("📉 Recent Delay Count", 0, 50, 2)
    }

    # ---------------- PREDICTION ---------------- #
    if st.button("Predict Risk"):

        # -------- SCALING -------- #
        scaled_data = {
            "P_2": input_data["P_2"] / 1000,
            "B_1": input_data["B_1"] / 100000,
            "D_39": input_data["D_39"] / 100,
            "R_1": input_data["R_1"] / 10,
            "S_3": input_data["S_3"] / 100000,
            "D_41": input_data["D_41"] / 100
        }

        # -------- FULL FEATURE VECTOR -------- #
        full_input = {col: 0 for col in all_columns}
        for col in scaled_data:
            full_input[col] = scaled_data[col]

        df = pd.DataFrame([full_input])

        # -------- MODEL PREDICTION -------- #
        prob = model.predict_proba(df)[0][1]

        st.markdown("## 📊 Prediction Result")
        st.markdown(f"### 🔢 Default Probability: **{prob:.2f}**")

        # -------- MODEL RISK -------- #
        if prob < 0.3:
            model_risk = "🟢 Low Risk"
        elif prob < 0.7:
            model_risk = "🟡 Medium Risk"
        else:
            model_risk = "🔴 High Risk"

        st.markdown(f"**Model Prediction:** {model_risk}")

        # -------- BUSINESS RULES -------- #
        rule_triggered = False
        rule_reasons = []

        if input_data["D_39"] > 60:
            rule_triggered = True
            rule_reasons.append("Days past due > 60")

        if input_data["D_41"] > 10:
            rule_triggered = True
            rule_reasons.append("Frequent delays")

        if input_data["P_2"] < 400:
            rule_triggered = True
            rule_reasons.append("Poor payment behavior")

        # -------- FINAL DECISION -------- #
        if rule_triggered:
            final_risk = "🔴 High Risk"
            decision = "❌ Reject Loan"
        else:
            final_risk = model_risk
            if "Low" in model_risk:
                decision = "✅ Approve Loan"
            elif "Medium" in model_risk:
                decision = "⚠️ Review Manually"
            else:
                decision = "❌ Reject Loan"

        # -------- DISPLAY FINAL RESULT -------- #
        st.markdown("### 📌 Final Decision (Hybrid AI)")
        if "High" in final_risk:
            st.error(final_risk)
        elif "Medium" in final_risk:
            st.warning(final_risk)
        else:
            st.success(final_risk)

        # ---------------- MODEL EXPLANATION (SHAP) ---------------- #
        st.markdown("### 🤖 Model Explanation (Data-driven)")

        try:
            background = np.zeros((1, df.shape[1]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(df, nsamples=50)

            try:
                values = shap_values[1][0]
            except:
                values = shap_values.values[0]

            shap_df = pd.DataFrame({
                "feature": df.columns,
                "impact": values
            })

            shap_df["abs"] = shap_df["impact"].abs()
            top_features = shap_df.sort_values("abs", ascending=False).head(3)

            feature_names_map = {
                "P_2": "Payment Behavior Score",
                "B_1": "Account Balance",
                "D_39": "Days Past Due",
                "R_1": "Risk Indicator Score",
                "S_3": "Spending Pattern",
                "D_41": "Recent Delay Count"
            }

            for _, row in top_features.iterrows():
                name = feature_names_map.get(row["feature"], row["feature"])
                if row["impact"] > 0:
                    st.write(f"🔺 {name} increased the risk")
                else:
                    st.write(f"🔻 {name} reduced the risk")

        except:
            st.info("Model explanation unavailable.")

        # ---------------- RULE EXPLANATION ---------------- #
        if rule_triggered:
            st.markdown("### ⚠️ Business Rule Explanation")
            for r in rule_reasons:
                st.write(f"🔴 {r} triggered high risk override")

        # ---------------- FINAL DECISION ---------------- #
        st.markdown("### 📌 Suggested Decision")
        st.markdown(f"### {decision}")

        # ---------------- TRANSPARENCY LINE ---------------- #
        st.markdown("---")
        st.markdown(
            "💡 *SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency.*"
        )
