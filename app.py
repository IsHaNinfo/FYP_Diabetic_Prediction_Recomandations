from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os
import io
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from sklearn.preprocessing import StandardScaler
from src.pipeline.Diabetic_Risk_Predict_Pipeline.predict_pipeline_diabetic import (
    CustomData,
    PredictPipeline,
)
from src.pipeline.Nutration_Risk_Predict_Pipeline.predict_pipeline_nutration import (
    NutritionRiskCustomData,
    NutritionRiskPredictPipeline,
)

from src.pipeline.PhysicalActivity_Risk_Prediction_Pipeline.predict_pipeline_physicalactivity import (
    PhysicalRiskCustomData,
    PhysicalRiskPredictPipeline,
)

from src.pipeline.Nutrition_Recommandations.nutrition_recommandations import (
    NutritionRecommendationsCustomData,
    NutritionRecommendationsPredictPipeline,
)

from src.pipeline.Mental_Risk_Predict_pipeline.predict_pipeline_mental import predict_mental_scenario
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


@app.route("/nutritionriskprediction", methods=["POST"])
def nutrition_risk_prediction():
    try:
        # Get JSON data from the frontend
        data_json = request.get_json()
        print("Received data for nutrition risk prediction:", data_json)
        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)
        # Create an instance of NutritionRiskCustomData
        nutrition_data = NutritionRiskCustomData(
            age=int(data_json["age"]),
            gender=data_json["gender"],
            height=height,
            weight=weight,
            carbohydrate_consumption=float(data_json["Carbohydrate_Consumption"]),
            protein_intake=float(data_json["Protein_Intake"]),
            fat_intake=float(data_json["Fat_Intake"]),
            regularity_of_meals=float(
                data_json["Regularity_of_Meals"]
            ),  # Assuming this is categorical
            portion_control=float(
                data_json["Portion_Control"]
            ),  # Assuming this is categorical
            caloric_balance=float(
                data_json["Caloric_Balance"]
            ),  # Assuming this is categorical
            sugar_consumption=float(data_json["Sugar_Consumption"]),
            DiabetesRisk=float(data_json["DiabetesRisk"]),
            # Assuming this is a float value
            bmi=bmi,
        )

        # Convert input data to DataFrame
        input_df = nutrition_data.get_data_as_data_frame()

        # Predict using NutritionRiskPredictPipeline
        predict_pipeline = NutritionRiskPredictPipeline()
        results = predict_pipeline.predict(input_df)

        return jsonify({"prediction": results.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/physicalriskprediction", methods=["POST"])
def physical_risk_prediction():
    try:
        # Get JSON data from the frontend
        data_json = request.get_json()
        print("Received data for physical risk prediction:", data_json)
        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)
        # Create an instance of NutritionRiskCustomData
        nutrition_data = PhysicalRiskCustomData(
            age=int(data_json["age"]),
            gender=data_json["gender"],
            height=height,
            weight=weight,
            energy_levels=float(data_json["EnergyLevels"]),
            physical_activity=float(data_json["Physical_Activity"]),
            sitting_time=float(data_json["Sitting_Time"]),
            cardiovascular_health=float(
                data_json["Cardiovascular_Health"]
            ),  # Assuming this is categorical
            muscle_strength=float(
                data_json["Muscle_Strength"]
            ),  # Assuming this is categorical
            flexibility=float(data_json["Flexibility"]),  # Assuming this is categorical
            balance=float(data_json["Balance"]),
            thirsty=float(data_json["Thirsty"]),
            pain_or_discomfort=float(data_json["Pain_or_Discomfort"]),
            available_time=float(data_json["Available_Time"]),
            DiabetesRisk=float(data_json["DiabetesRisk"]),
            bmi=bmi,
        )

        # Convert input data to DataFrame
        input_df = nutrition_data.get_data_as_data_frame()

        # Predict using NutritionRiskPredictPipeline
        predict_pipeline = PhysicalRiskPredictPipeline()
        results = predict_pipeline.predict(input_df)

        return jsonify({"prediction": results.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/nutritionrecommendations", methods=["POST"])
def nutrition_recommendations():
    try:
        data_json = request.get_json()
        print("Received data for nutrition recommendations:", data_json)
        # Prepare input data
        custom_data = NutritionRecommendationsCustomData(**data_json)
        input_df = custom_data.get_data_as_data_frame()
        # Predict
        pipeline = NutritionRecommendationsPredictPipeline()
        preds = pipeline.predict(input_df)
        # Return the first prediction as an example
        return jsonify({"recommendations": preds[0].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/menatalrecommendations", methods=["POST"])
def predict_mental():
    try:
        raw_input = {
            "Perceived_Control": request.form["Perceived_Control"],
            "Stress_Freq_Intensity": request.form["Stress_Freq_Intensity"],
            "Emotional_Reg": request.form["Emotional_Reg"],
            "Physical_Stress": request.form["Physical_Stress"],
            "Cognitive_Stress": request.form["Cognitive_Stress"],
            "Behavioral_Response": request.form["Behavioral_Response"],
            "Work_Stress": request.form["Work_Stress"],
            "Productivity": request.form["Productivity"],
            "Suicidal_Thoughts": request.form["Suicidal_Thoughts"],
            "FreeTime": request.form["FreeTime"],
        }

        if "image" not in request.files:
            return jsonify({"error": "Image is required"}), 400

        image = request.files["image"]
        image_bytes = image.read()

        result = predict_mental_scenario(raw_input, image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
