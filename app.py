from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/fraud_model.pkl')

@app.route('/')
def home():
    return "<h1>Fraud Detection API Running</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features (expects 30)
        features = data['features']

        # Convert to numpy array and reshape for a single sample
        features_array = np.array(features).reshape(1, -1)

        # Predict probability (index [1] is usually the positive class/fraud)
        prob = model.predict_proba(features_array)[0][1]

        # Prediction label based on 0.5 threshold
        prediction = "FRAUD" if prob > 0.3 else "NOT FRAUD"

        return jsonify({
            "fraud_probability": float(prob),
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
