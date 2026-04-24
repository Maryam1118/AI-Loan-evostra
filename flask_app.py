from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json

app = Flask(__name__)
CORS(app)   

# ---------------- LOAD MODELS ---------------- #
try:
    amex_model = pickle.load(open("models/amex_xgb_model.pkl", "rb"))
    gmsc_model = pickle.load(open("models/gmsc_xgb_model.pkl", "rb"))

    with open("columns/amex_columns.json") as f:
        amex_cols = json.load(f)

    with open("columns/gmsc_columns.json") as f:
        gmsc_cols = json.load(f)

except Exception as e:
    print("Error loading models:", e)


# ---------------- HELPER FUNCTION ---------------- #
def get_risk_label(prob):
    if prob < 0.3:
        return "Low Risk", "Customer is financially stable"
    elif prob < 0.7:
        return "Medium Risk", "Customer shows moderate risk"
    else:
        return "High Risk", "Customer is likely to default"


# ---------------- ROUTE ---------------- #
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        dataset = data.get("dataset")

        # ---------------- AMEX ---------------- #
        if dataset == "AMEX":

            full = {c: 0 for c in amex_cols}

            full["P_2"] = float(data.get("payment_score", 700)) / 1000
            full["B_1"] = float(data.get("balance", 40000)) / 100000
            full["D_39"] = float(data.get("days_due", 5)) / 100
            full["R_1"] = float(data.get("risk_score", 3)) / 10
            full["S_3"] = float(data.get("spending", 20000)) / 100000
            full["D_41"] = float(data.get("delay_count", 2)) / 100

            df = pd.DataFrame([full])
            prob = float(amex_model.predict_proba(df)[0][1])

        # ---------------- GMSC ---------------- #
        elif dataset == "GMSC":

            full = {c: 0 for c in gmsc_cols}

            full["RevolvingUtilizationOfUnsecuredLines"] = float(data.get("utilization", 0.3))
            full["age"] = float(data.get("age", 30))
            full["NumberOfTime30-59DaysPastDueNotWorse"] = float(data.get("past_due", 1))
            full["DebtRatio"] = float(data.get("debt_ratio", 0.5))
            full["MonthlyIncome"] = float(data.get("income", 50000))
            full["NumberOfOpenCreditLinesAndLoans"] = float(data.get("open_credit", 5))

            df = pd.DataFrame([full])
            prob = float(gmsc_model.predict_proba(df)[0][1])

        else:
            return jsonify({"error": "Invalid dataset. Choose AMEX or GMSC"}), 400

        # ---------------- RISK LABEL ---------------- #
        risk_label, message = get_risk_label(prob)

        # ---------------- RESPONSE ---------------- #
        return jsonify({
            "probability": round(prob, 4),
            "risk_level": risk_label,
            "message": message
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ---------------- HEALTH CHECK ---------------- #
@app.route("/")
def home():
    return "✅ API is running"


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)