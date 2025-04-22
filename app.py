from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model and scaler (ensure they are saved as .pkl files)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))

# Define feature names
FEATURES = [
    "Nitrogen (N)",
    "Phosphorus (P)",
    "Potassium (K)",
    "Temperature (°C)",
    "Humidity (%)",
    "Soil Type",
    "pH Level",
    "Rainfall (mm)",
]


@app.route("/")
def home():
    return render_template("index.html")  # A simple HTML form for input\2


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from request (either JSON or form)
        data = request.json if request.is_json else request.form
        input_features = [float(data[feature]) for feature in FEATURES]

        # Scale input
        input_scaled = scaler.transform([input_features])

        # Predict crop
        predicted_crop = model.predict(input_scaled)[0]

        return jsonify({"Recommended Crop": predicted_crop})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
