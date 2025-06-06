from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import re
from peft import PeftModel

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
from src.components.Nutrition_Recommendations.utils import (
    recommend_foods,
    generate_meal_plan,
)
from src.components.Nutrition_Recommendations.config import FOOD_DATA_PATH, TARGET_COLS

from src.pipeline.Exercises_Recommandations.exercises_recommandations import (
    ExercisesRecommendationsCustomData,
    build_prompt,
    format_paragraphs,
)

# ...existing code...
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
        predicted_nutrition = preds[0]  # Assuming single user

        # Format prediction into dictionary with target column names
        pred_dict = {k: float(v) for k, v in zip(TARGET_COLS, predicted_nutrition)}

        # Load food database
        food_df = pd.read_csv(FOOD_DATA_PATH)

        # Recommend foods and generate meal plan
        recommended = recommend_foods(
            food_df,
            pred_dict["Protein_Intake"],
            pred_dict["Fat_Intake"],
            pred_dict["Carbohydrate_Consumption"],
        )

        meal_plan = generate_meal_plan(pred_dict, food_df)

        return jsonify(
            {
                "predicted_nutrition": pred_dict,
                "recommended_foods": recommended.head(5).to_dict(orient="records"),
                "meal_plan": meal_plan,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


base_model_name = "EleutherAI/gpt-neo-1.3B"
adapter_path = "./adapter"  # adjust if different

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

@app.route("/exercisesrecommendations", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        custom_data = ExercisesRecommendationsCustomData(**data)
        prompt = build_prompt(custom_data.get_data_as_data_frame().iloc[0].to_dict())

        output = generator(
            prompt,
            max_new_tokens=400,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        result = output[0]["generated_text"][len(prompt):].strip()
        formatted = format_paragraphs(result)

        return jsonify({"recommendation": formatted})

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
