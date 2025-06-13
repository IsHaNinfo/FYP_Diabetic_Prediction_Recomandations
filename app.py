from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os
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
from src.components.Exersices_Recommendations.exercise_recommander import recommend
from src.pipeline.Exercises_Recommandations.exercises_recommand_pipeline import  format_paragraphs

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize prediction pipelines as global variables
diabetic_pipeline = PredictPipeline()
nutrition_pipeline = NutritionRiskPredictPipeline()
physical_pipeline = PhysicalRiskPredictPipeline()
nutrition_recommendations_pipeline = NutritionRecommendationsPredictPipeline()

# Cache food data
@lru_cache(maxsize=1)
def get_food_data():
    return pd.read_csv(FOOD_DATA_PATH)

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
        
        # Run prediction and validation in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prediction_future = executor.submit(diabetic_pipeline.predict, data)
            validation_future = executor.submit(validate_and_plot, data)
            
            results = prediction_future.result()
            risk_validation_results = validation_future.result()

        # Convert numpy array to list if needed
        if isinstance(results, np.ndarray):
            results = results.tolist()
        if isinstance(risk_validation_results, np.ndarray):
            risk_validation_results = risk_validation_results.tolist()

        response_data = { 
            "prediction": float(results[0]) if isinstance(results, list) else float(results),
            "risk_validation": risk_validation_results
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

@app.route("/nutritionrecommendations", methods=["POST"])
def nutrition_recommendations():
    try:
        data_json = request.get_json()
        custom_data = NutritionRecommendationsCustomData(**data_json)
        input_df = custom_data.get_data_as_data_frame()

        # Get predictions
        preds = nutrition_recommendations_pipeline.predict(input_df)
        predicted_nutrition = preds[0]

        # Convert predictions to dictionary and ensure all values are Python native types
        pred_dict = {k: float(v) for k, v in zip(TARGET_COLS, predicted_nutrition)}
        
        # Combine predictions with user data
        user_pred = {
            'DiabetesRisk': float(pred_dict['DiabetesRisk']),
            'NutritionRisk': float(pred_dict['NutritionRisk']),
            'Protein_Intake': float(pred_dict['Protein_Intake']),
            'Fat_Intake': float(pred_dict['Fat_Intake']),
            'Carbohydrate_Consumption': float(pred_dict['Carbohydrate_Consumption']),
            'Sugar_Consumption': float(pred_dict['Sugar_Consumption']),
            'Caloric_Balance': float(pred_dict['Caloric_Balance'])
        }

        # Get cached food data
        food_df = get_food_data()

        # Run recommendations and meal plan generation in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            recommended_future = executor.submit(
                recommend_foods,
                food_df,
                user_pred["Protein_Intake"],
                user_pred["Fat_Intake"],
                user_pred["Carbohydrate_Consumption"],
            )
            meal_plan_future = executor.submit(generate_meal_plan, user_pred, food_df)
            
            recommended = recommended_future.result()
            meal_plan = meal_plan_future.result()

        # Ensure all data is JSON serializable
        recommended_foods = recommended.head(5).to_dict(orient="records")
        for food in recommended_foods:
            for key, value in food.items():
                if isinstance(value, np.number):
                    food[key] = float(value)

        return jsonify({
            "predicted_nutrition": pred_dict,
            "recommended_foods": recommended_foods,
            "meal_plan": meal_plan,
        })
    except Exception as e:
        print(f"Error in nutrition recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/exerciserecommendations", methods=["POST"])
def exercise_recommendations():
    try:
        data_json = request.get_json()
        
        # Prepare input for the recommendation model
        user_input = {
            "age": int(data_json["age"]),
            "bmi": float(data_json["bmi"]),
            "energy_levels": float(data_json["energy_levels"]),
            "physical_activity": float(data_json["physical_activity"]),
            "available_time": float(data_json["available_time"]),
            "diabetesRisk": float(data_json["diabetes_risk"]),
            "gender": int(data_json["gender"]),
            "pain_or_discomfort": int(data_json["pain_or_discomfort"]),
            "cardiovascular_health": int(data_json["cardiovascular_health"]),
            "muscle_strength": int(data_json["muscle_strength"]),
            "flexibility": int(data_json["flexibility"]),
            "balance": int(data_json["balance"])
        }

        # Get recommendations from the model
        recommendations = recommend(user_input)
        
        # Format the recommendations into readable text
        formatted_recommendations = format_paragraphs(recommendations)

        return jsonify({
            "recommendations": recommendations,
        })

    except Exception as e:
        print(f"Error in exercise recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


    