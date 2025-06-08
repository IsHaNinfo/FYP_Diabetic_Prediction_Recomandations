from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os
import warnings
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
from src.pipeline.Nutrition_Recommandations.nutrition_recommandations import (
    NutritionRecommendationsCustomData,
    NutritionRecommendationsPredictPipeline,
)
from src.components.Nutrition_Recommendations.utils import (
    recommend_foods,
    generate_meal_plan,
)
from src.components.Nutrition_Recommendations.config import FOOD_DATA_PATH, TARGET_COLS

from src.components.Diabetic_Risk_Prediction.risk_validation import validate_and_plot


app = Flask(__name__)
CORS(app)

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
            risk_level=1.0 if data_json["RiskLevel"] == "Yes" else 0.0
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        
        try:
            results = predict_pipeline.predict(data)
           
        except Exception as pred_error:
            raise pred_error

        risk_validation_results = validate_and_plot(data)

        response_data = { 
            "prediction": float(results[0]),
            "risk_validation": risk_validation_results
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        
        return jsonify({"error": str(e)}), 400

@app.route("/nutritionriskprediction", methods=["POST"])
def nutrition_risk_prediction():
    try:
        data_json = request.get_json()
        print("Received data for nutrition risk prediction:", data_json)
        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)
        nutrition_data = NutritionRiskCustomData(
            age=int(data_json["age"]),
            gender=data_json["gender"],
            height=height,
            weight=weight,
            carbohydrate_consumption=float(data_json["Carbohydrate_Consumption"]),
            protein_intake=float(data_json["Protein_Intake"]),
            fat_intake=float(data_json["Fat_Intake"]),
            regularity_of_meals=float(data_json["Regularity_of_Meals"]),
            portion_control=float(data_json["Portion_Control"]),
            caloric_balance=float(data_json["Caloric_Balance"]),
            sugar_consumption=float(data_json["Sugar_Consumption"]),
            DiabetesRisk=float(data_json["DiabetesRisk"]),
            bmi=bmi,
        )

        input_df = nutrition_data.get_data_as_data_frame()
        predict_pipeline = NutritionRiskPredictPipeline()
        results = predict_pipeline.predict(input_df)

        return jsonify({
            "prediction": results,
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/physicalriskprediction", methods=["POST"])
def physical_risk_prediction():
    try:
        data_json = request.get_json()
        print("Received data for physical risk prediction:", data_json)
        height = float(data_json["height"])
        weight = float(data_json["weight"])
        bmi = weight / ((height / 100) ** 2)
        nutrition_data = PhysicalRiskCustomData(
            age=int(data_json["age"]),
            gender=data_json["gender"],
            height=height,
            weight=weight,
            energy_levels=float(data_json["EnergyLevels"]),
            physical_activity=float(data_json["Physical_Activity"]),
            sitting_time=float(data_json["Sitting_Time"]),
            cardiovascular_health=float(data_json["Cardiovascular_Health"]),
            muscle_strength=float(data_json["Muscle_Strength"]),
            flexibility=float(data_json["Flexibility"]),
            balance=float(data_json["Balance"]),
            thirsty=float(data_json["Thirsty"]),
            pain_or_discomfort=float(data_json["Pain_or_Discomfort"]),
            available_time=float(data_json["Available_Time"]),
            DiabetesRisk=float(data_json["DiabetesRisk"]),
            bmi=bmi,
        )

        input_df = nutrition_data.get_data_as_data_frame()
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

        custom_data = NutritionRecommendationsCustomData(**data_json)
        input_df = custom_data.get_data_as_data_frame()

        pipeline_ = NutritionRecommendationsPredictPipeline()
        preds = pipeline_.predict(input_df)
        predicted_nutrition = preds[0]

        # Convert predictions to dictionary
        pred_dict = {k: float(v) for k, v in zip(TARGET_COLS, predicted_nutrition)}
        
        # Combine predictions with user data
        user_pred = {
            'DiabetesRisk': pred_dict['DiabetesRisk'],
            'NutritionRisk': pred_dict['NutritionRisk'],
            'Protein_Intake': pred_dict['Protein_Intake'],
            'Fat_Intake': pred_dict['Fat_Intake'],
            'Carbohydrate_Consumption': pred_dict['Carbohydrate_Consumption'],
            'Sugar_Consumption': pred_dict['Sugar_Consumption'],
            'Caloric_Balance': pred_dict['Caloric_Balance']
        }

        food_df = pd.read_csv(FOOD_DATA_PATH)
        recommended = recommend_foods(
            food_df,
            user_pred["Protein_Intake"],
            user_pred["Fat_Intake"],
            user_pred["Carbohydrate_Consumption"],
        )

        meal_plan = generate_meal_plan(user_pred, food_df)

        return jsonify({
            "predicted_nutrition": pred_dict,
            "recommended_foods": recommended.head(5).to_dict(orient="records"),
            "meal_plan": meal_plan,
        })
    except Exception as e:
        print(f"Error in nutrition recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 400






if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


    