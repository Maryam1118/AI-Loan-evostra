# ---------------- FEATURE NAME MAP ---------------- #
feature_names_map = {
    "P_2": "Payment Behavior Score",
    "B_1": "Account Balance",
    "D_39": "Days Past Due",
    "R_1": "Risk Indicator Score",
    "S_3": "Monthly Spending",
    "D_41": "Recent Delay Count"
}

# ---------------- PREDICTION ---------------- #
prob = model.predict_proba(df)[0][1]

st.markdown("## 📊 Prediction Result")
st.markdown(f"### 🔢 Default Probability: **{prob:.2f}**")

# Model risk
if prob < 0.3:
    model_risk = "🟢 Low Risk"
elif prob < 0.7:
    model_risk = "🟡 Medium Risk"
else:
    model_risk = "🔴 High Risk"

# ---------------- BUSINESS RULES ---------------- #
rule_triggered = False
rule_reasons = []

if input_data["D_39"] > 60:
    rule_triggered = True
    rule_reasons.append("Customer has very high overdue days")

if input_data["D_41"] > 10:
    rule_triggered = True
    rule_reasons.append("Customer frequently delays payments")

if input_data["P_2"] < 400:
    rule_triggered = True
    rule_reasons.append("Customer has poor payment behavior")

# ---------------- FINAL DECISION ---------------- #
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

# ---------------- DISPLAY FINAL ---------------- #
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    values = shap_values[0] if isinstance(shap_values, list) else shap_values

    shap_df = pd.DataFrame({
        "feature": df.columns,
        "impact": values[0]
    })

    shap_df["abs"] = shap_df["impact"].abs()
    top_features = shap_df.sort_values("abs", ascending=False).head(3)

    model_reasons = []

    for _, row in top_features.iterrows():
        name = feature_names_map.get(row["feature"], row["feature"])

        if row["impact"] > 0:
            st.write(f"🔺 {name} increased the risk")
            model_reasons.append(f"{name} is contributing to higher risk")
        else:
            st.write(f"🔻 {name} reduced the risk")
            model_reasons.append(f"{name} is helping reduce risk")

except:
    st.info("Model explanation unavailable")
    model_reasons = []

# ---------------- BUSINESS RULE EXPLANATION ---------------- #
if rule_triggered:
    st.markdown("### ⚠️ Business Rule Explanation")

    for r in rule_reasons:
        st.write(f"🔴 {r}")

# ---------------- FINAL INTERPRETATION ---------------- #
st.markdown("### 🧠 Final Interpretation")

if "High" in final_risk:
    st.write("Customer has a **high probability of default** due to:")
elif "Medium" in final_risk:
    st.write("Customer shows **moderate financial risk** due to:")
else:
    st.write("Customer is **financially stable** due to:")

# Combine explanations
for r in model_reasons:
    st.write(f"• {r}")

for r in rule_reasons:
    st.write(f"• {r}")

# ---------------- SUGGESTED DECISION ---------------- #
st.markdown("### 📌 Suggested Decision")
st.markdown(f"### {decision}")

# ---------------- TRANSPARENCY LINE ---------------- #
st.markdown("---")
st.markdown(
    "💡 *SHAP explains the model prediction, while business rules ensure critical risk conditions are enforced. I display both to maintain transparency.*"
)
