from flask import Flask, request, jsonify
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

model = joblib.load('logistic_regression_model.pkl')
with open('shap_explainer.pkl', 'rb') as f:
    shap_explainer = pickle.load(f)

FEATURE_NAMES = list(pd.read_csv('X_train.csv').columns)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Phishing Detection API is running", "status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        if features is None or len(features) != len(FEATURE_NAMES):
            return jsonify({"error": f"Expected {len(FEATURE_NAMES)} features"}), 400

        X = pd.DataFrame([features], columns=FEATURE_NAMES)
        prediction = int(model.predict(X)[0])
        probability = model.predict_proba(X)[0].tolist()

        return jsonify({
            "prediction": prediction,
            "label": "phishing" if prediction == 1 else "legitimate",
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.get_json()
        features = data.get('features')
        X = pd.DataFrame([features], columns=FEATURE_NAMES)

        shap_vals = shap_explainer.shap_values(X)[0]
        all_features = sorted(zip(FEATURE_NAMES, shap_vals), key=lambda x: abs(x[1]), reverse=True)

        explanation = [
            {"feature": f, "shap_value": round(float(v), 4),
             "direction": "increases_phishing_risk" if v > 0 else "decreases_phishing_risk"}
            for f, v in all_features[:5]
        ]
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
