import pandas as pd
import os
import numpy as np
from pathlib import Path
from src.components.Nutrition_Recommandations.NutritonRecommander import (
    get_meal_plan,
    MealPlanRecommender,
    print_meal_plan
)
import pickle
import time

class NutritionRecommendationsCustomData:
    def __init__(self,
                 age: int,
                 gender: float,
                 height: float,
                 weight: float,
                 Carbohydrate_Consumption: float,
                 Protein_Intake: str,
                 Fat_Intake: str,
                 Regularity_of_Meals: str,
                 Portion_Control: float,
                 Caloric_Balance: float,
                 Sugar_Consumption: float,
                 Diabetic_Risk: float,
                 nutritionRisk: float):
        
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.Carbohydrate_Consumption = Carbohydrate_Consumption
        self.Protein_Intake = 1.0 if Protein_Intake == "Yes" else 0.0
        self.Fat_Intake = 1.0 if Fat_Intake == "Healthy fats" else 0.0
        self.Regularity_of_Meals = 1.0 if Regularity_of_Meals == "Yes" else 0.0
        self.Portion_Control = Portion_Control
        self.Caloric_Balance = Caloric_Balance
        self.Sugar_Consumption = Sugar_Consumption
        self.DiabetesRisk = Diabetic_Risk
        self.NutritionRisk = nutritionRisk
        self.BMI = weight / ((height / 100) ** 2)
        self.Activity_Level = "Moderate"  # Default value

    def get_data_as_data_frame(self):
        return pd.DataFrame([{
            'Age': self.age,
            'Gender': self.gender,
            'Height': self.height,
            'Weight': self.weight,
            'Carbohydrate_Consumption': self.Carbohydrate_Consumption,
            'Protein_Intake': self.Protein_Intake,
            'Fat_Intake': self.Fat_Intake,
            'Regularity_of_Meals': self.Regularity_of_Meals,
            'Portion_Control': self.Portion_Control,
            'Caloric_Balance': self.Caloric_Balance,
            'Sugar_Consumption': self.Sugar_Consumption,
            'DiabetesRisk': self.DiabetesRisk,
            'NutritionRisk': self.NutritionRisk,
            'BMI': self.BMI,
            'Activity_Level': self.Activity_Level
        }])

class NutritionRecommendationsPredictPipeline:
    def __init__(self):
        self.base_path = Path('G:/FYP_Diabetic_Prediction_Recomandations/notebook/data')
        self.food_data_path = self.base_path / 'RecommandationDatasets/NutritionsDatasets/SrilankanCommonFoods.csv'
        
        # Load food data
        self._load_food_data()
        
    def _load_food_data(self):
        """Load the food data"""
        try:
            if not self.food_data_path.exists():
                raise FileNotFoundError(f"Food data not found at {self.food_data_path}")
            
            self.food_df = pd.read_csv(self.food_data_path)
            print("Successfully loaded food data")
            
        except Exception as e:
            print(f"Error loading food data: {str(e)}")
            raise
    
    def generate_recommendations(self, user_data):
        try:
            # Generate recommendations
            meal_plans = get_meal_plan(self.food_df, user_data)
            
            # Format recommendations
            formatted_recommendations = self._format_recommendations(meal_plans)
            
            return formatted_recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            raise
            
    def _format_recommendations(self, recommendations):
        formatted = []
        for day, plan in enumerate(recommendations, 1):
            day_plan = {
                "day": day,
                "meals": {}
            }
            
            for meal_type, foods in plan.items():
                day_plan["meals"][meal_type] = [
                    {
                        "food_name": food["food_name"],
                        "nutritional_info": food["nutritional_info"],
                        "health_benefits": food["health_benefits"]
                    }
                    for food in foods
                ]
            
            formatted.append(day_plan)
            
        return {"meal_plan": formatted}

def run_nutrition_recommendation_pipeline(user_profile):
    try:
        print("Pipeline received profile:", user_profile)
        
        # Load Sri Lankan Common Foods data
        BASE_PATH = Path('G:/FYP_Diabetic_Prediction_Recomandations/notebook/data')
        food_df = pd.read_csv(BASE_PATH / 'RecommandationDatasets/NutritionsDatasets/SrilankanCommonFoods.csv')
        
        # Convert user profile keys to match recommender expectations
        converted_profile = {
            'Age': user_profile['Age'],
            'Gender': user_profile['Gender'],
            'Height': user_profile['Height'],
            'Weight': user_profile['Weight'],
            'BMI': user_profile['BMI'],
            'DiabetesRisk': user_profile['DiabetesRisk'],
            'NutritionRisk': user_profile['NutritionRisk'],
            'Protein_Intake': user_profile['Protein_Intake'],
            'Fat_Intake': user_profile['Fat_Intake'],
            'Carbohydrate_Consumption': user_profile['Carbohydrate_Consumption'],
            'Sugar_Consumption': user_profile['Sugar_Consumption'],
            'Regularity_of_Meals': user_profile['Regularity_of_Meals'],
            'Portion_Control': user_profile['Portion_Control'],
            'Activity_Level': user_profile['Activity_Level']
        }
        
        print("Converted profile:", converted_profile)
        
        # Generate meal plan
        meal_plan = get_meal_plan(food_df, converted_profile)
        
        return meal_plan
        
    except Exception as e:
        print(f"Error in nutrition recommendation pipeline: {str(e)}")
        print(f"User profile keys: {user_profile.keys()}")
        raise

# For backward compatibility
def generate_recommendations(user_profile):
    return run_nutrition_recommendation_pipeline(user_profile)




