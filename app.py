from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from sklearn.preprocessing import StandardScaler
from src.pipeline.Diabetic_Risk_Predict_Pipeline.predict_pipeline_diabetic import (
    CustomData,
    PredictPipeline,
)

app = Flask(__name__)
# Initialize CORS with default options
CORS(app)

# Remove the duplicate Flask app initialization
# app = Flask(__name__)  # Remove this line


@app.route("/")
def index():
    return "API is working"


@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    try:
        data_json = request.get_json()
        print("Received data from frontend:", data_json)

        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)

        data = CustomData(
            age=int(data_json["age"]),
            gender=data_json["gender"],
            height=height,
            weight=weight,
            waist_circumference=float(data_json["Waist_Circumference"]),
            diet_food_habits=int(data_json["Diet_Food_Habits"]),
            family_history=float(data_json["Family_History"]),
            high_blood_pressure=float(data_json["Blood_Pressure"]),
            cholesterol_lipid_levels=float(data_json["Cholesterol_Lipid_Levels"]),
            thirst=float(data_json["Thirst"]),
            fatigue=float(data_json["Fatigue"]),
            urination=float(data_json["Urination"]),
            vision_changes=float(data_json["Vision_Changes"]),
            bmi=bmi,
            risk_level=float(data_json["RiskLevel"]),
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return jsonify({"prediction": results[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
