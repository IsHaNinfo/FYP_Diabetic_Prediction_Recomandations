from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os
import io
from PIL import Image

import warnings
from functools import lru_cache
import concurrent.futures
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
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


from src.pipeline.Mental_Risk_Predict_pipeline.predict_pipeline_mental import predict_mental_scenario
from src.components.Diabetic_Risk_Prediction.risk_validation import validate_and_plot
from src.components.Exersices_Recommendations.exercise_recommander import recommend
from src.pipeline.Exercises_Recommandations.exercises_recommand_pipeline import format_paragraphs
from pathlib import Path
from src.components.Nutrition_Recommandations import generate_health_aware_meal_plan, load_food_df,analyze_meal_plan_with_contribution

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize prediction pipelines as global variables
diabetic_pipeline = PredictPipeline()
nutrition_pipeline = NutritionRiskPredictPipeline()
physical_pipeline = PhysicalRiskPredictPipeline()


# Define the path to the saved model and data
BASE_PATH = Path('G:/FYP_Diabetic_Prediction_Recomandations/notebook/data')
model_data_path = BASE_PATH / 'model_data.pkl'

@app.route("/")
def index():
    return "API is working"

@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    try:
        data_json = request.get_json()

        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)

        data = CustomData(
            age=int(data_json["age"]),
            gender=str(data_json["gender"]),
            height=height,
            weight=weight,
            waist_circumference=float(data_json["Waist_Circumference"]),
            diet_food_habits=int(data_json["Diet_Food_Habits"]),
            family_history=1.0 if data_json["Family_History"] == "Yes" else 0.0,
            high_blood_pressure=1.0 if data_json["Blood_Pressure"] == "Yes" else 0.0,
            cholesterol_lipid_levels=1.0 if data_json["Cholesterol_Lipid_Levels"] == "Yes" else 0.0,
            thirst=1.0 if data_json["Thirst"] == "Yes" else 0.0,
            fatigue=1.0 if data_json["Fatigue"] == "Yes" else 0.0,
            urination=1.0 if data_json["Urination"] == "Yes" else 0.0,
            vision_changes=1.0 if data_json["Vision_Changes"] == "Yes" else 0.0,
            bmi=bmi,
            risk_level=2.0 if data_json["RiskLevel"] == "High" else (1.0 if data_json["RiskLevel"] == "Moderate" else 0.0)
        )

        # Use the new predict_with_validation method
        results = diabetic_pipeline.predict_with_validation(data)

        # Format the response with clear labels and percentages
        response_data = {
            "predictions": {
                "your_model": {
                    "probability": results["your_model_prediction"],
                    "percentage": results["your_risk_percentage"],
                    "risk_level": results["risk_level"]
                },
                "pima_model": {
                    "probability": results["pima_prediction"],
                    "percentage": results["pima_risk_percentage"],
                    "risk_level": results["pima_risk_level"]
                }
            },
            "comparison": {
                "prediction_difference": results["prediction_difference"],
                "models_agree": results["models_agree"],
                "similar_cases": {
                    "count": results["similar_cases_count"],
                    "average_prediction": results["similar_cases_avg_prediction"]
                }
            },
            "validation_metrics": {
                "brier_score": results["brier_score"],
                "roc_auc": results["roc_auc"]
            },
            "summary": {
                "primary_risk": f"{results['your_risk_percentage']}% ({results['risk_level']})",
                "benchmark_risk": f"{results['pima_risk_percentage']}% ({results['pima_risk_level']})",
                "agreement_status": "Models agree on risk level" if results["models_agree"] else "Models disagree on risk level",
                "confidence": "High" if results["brier_score"] < 0.1 else "Medium" if results["brier_score"] < 0.2 else "Low"
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/nutritionriskprediction", methods=["POST"])
def nutrition_risk_prediction():
    try:
        data_json = request.get_json()
        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)
        
        # Convert Protein_Intake and Fat_Intake to numerical values
        protein_intake_map = {"Yes": 1.0, "No": 0.0}
        fat_intake_map = {"Healthy fats": 1.0, "Unhealthy fats": 0.0}
        
        nutrition_data = NutritionRiskCustomData(
            age=int(data_json["age"]),
            gender=int(data_json["gender"]),
            height=height,
            weight=weight,
            carbohydrate_consumption=float(data_json["Carbohydrate_Consumption"]),
            protein_intake=protein_intake_map.get(data_json["Protein_Intake"], 0.0),
            fat_intake=fat_intake_map.get(data_json["Fat_Intake"], 0.0),
            regularity_of_meals=1.0 if data_json["Regularity_of_Meals"] == "Yes" else 0.0,
            portion_control=float(data_json["Portion_Control"]),
            caloric_balance=float(data_json["Caloric_Balance"]),
            sugar_consumption=float(data_json["Sugar_Consumption"]),
            DiabetesRisk=float(data_json["DiabetesRisk"]),
            bmi=bmi,
        )

        input_df = nutrition_data.get_data_as_data_frame()
        results, feature_contributions = nutrition_pipeline.predict(input_df)

        # Convert numpy array to list if needed
        if isinstance(results, np.ndarray):
            results = results.tolist()

        return jsonify({
            "prediction": results,
            "feature_contributions": feature_contributions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/physicalriskprediction", methods=["POST"])
def physical_risk_prediction():
    try:
        data_json = request.get_json()
        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)
        
        # Create mapping for categorical variables
        yes_no_map = {"Yes": 1.0, "No": 0.0}
        
        physical_data = PhysicalRiskCustomData(
            age=int(data_json["age"]),
            gender=int(data_json["gender"]),
            height=height,
            weight=weight,
            energy_levels=float(data_json["EnergyLevels"]),
            physical_activity=float(data_json["Physical_Activity"]),
            sitting_time=yes_no_map.get(data_json["Sitting_Time"], 0.0),
            cardiovascular_health=yes_no_map.get(data_json["Cardiovascular_Health"], 0.0),
            muscle_strength=yes_no_map.get(data_json["Muscle_Strength"], 0.0),
            flexibility=yes_no_map.get(data_json["Flexibility"], 0.0),
            balance=yes_no_map.get(data_json["Balance"], 0.0),
            thirsty=float(data_json["Thirsty"]),
            pain_or_discomfort=yes_no_map.get(data_json["Pain_or_Discomfort"], 0.0),
            available_time=float(data_json["Available_Time"]),
            DiabetesRisk=float(data_json["DiabetesRisk"]),
            bmi=bmi,
        )

        input_df = physical_data.get_data_as_data_frame()
        results, feature_contributions = physical_pipeline.predict(input_df)

        # Convert numpy array to list if needed
        if isinstance(results, np.ndarray):
            results = results.tolist()

        return jsonify({
            "prediction": results,
            "feature_contributions": feature_contributions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/exerciserecommendations", methods=["POST"])
def exercise_recommendations():
    try:
        data_json = request.get_json()
        
        # Calculate BMI
        height = float(data_json["height"])  # in cm
        weight = float(data_json["weight"])  # in kg
        bmi = weight / ((height / 100) ** 2)
        
        # Get user prompt if provided
        user_prompt = data_json.get("user_prompt", None)
        
        # Create mapping for categorical variables
        yes_no_map = {
            "Sitting_Time": {"Yes": 1.0, "No": 0.0},
            "Cardiovascular_Health": {"Yes": 1.0, "No": 0.0},
            "Muscle_Strength": {"Yes": 1.0, "No": 0.0},
            "Flexibility": {"Yes": 1.0, "No": 0.0},
            "Balance": {"Yes": 1.0, "No": 0.0},
            "Pain_or_Discomfort": {"Yes": 1.0, "No": 0.0}
        }
        
        # Prepare input for the recommendation model
        user_input = {
            "age": int(data_json["age"]),
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "energy_levels": float(data_json["energy_levels"]),
            "physical_activity": float(data_json["physical_activity"]),
            "sitting_time": yes_no_map["Sitting_Time"].get(data_json["sitting_time"], 0.0),
            "physical_activity_risk": float(data_json["physical_activity_risk"]),
            "available_time": float(data_json["available_time"]),
            "diabetesRisk": float(data_json["diabetes_risk"]),
            "gender": int(data_json["gender"]),
            "pain_or_discomfort": yes_no_map["Pain_or_Discomfort"].get(data_json["pain_or_discomfort"], 0.0),
            "cardiovascular_health": yes_no_map["Cardiovascular_Health"].get(data_json["cardiovascular_health"], 0.0),
            "muscle_strength": yes_no_map["Muscle_Strength"].get(data_json["muscle_strength"], 0.0),
            "flexibility": yes_no_map["Flexibility"].get(data_json["flexibility"], 0.0),
            "balance": yes_no_map["Balance"].get(data_json["balance"], 0.0),
            "goal": str(data_json["goal"])
        }

        # Get recommendations from the model with user prompt
        recommendations = recommend(user_input, user_prompt)
        
        return jsonify({
            "recommendations": recommendations
        })

    except Exception as e:
        print(f"Error in exercise recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/generate_meal_plan", methods=["POST"])
def generate_meal_plan():
    try:
        data_json = request.get_json()
        # Extract user data from the request
        user_data = {
            "age": int(data_json["age"]),
            "gender": int(data_json["gender"]),
            "bmi": float(data_json["bmi"]),
            "diabetes_risk": float(data_json["diabetes_risk"]),
            "nutrition_risk": float(data_json["nutrition_risk"]),
            "preferences": data_json["preferences"]
        }

        # Generate meal plan
        meal_plan = generate_health_aware_meal_plan(user_data, load_food_df())
        result = analyze_meal_plan_with_contribution(meal_plan, user_data)

        return jsonify(result)
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