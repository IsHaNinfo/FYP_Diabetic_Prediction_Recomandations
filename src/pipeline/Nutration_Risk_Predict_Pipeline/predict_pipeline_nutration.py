import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import os
import shap


class NutritionRiskPredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifact/nutrition", "model.pkl")
        self.preprocessor_path = os.path.join("artifact/nutrition", "nutrition_preprocessor.pkl")
        self.model = None
        self.preprocessor = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = load_object(file_path=self.model_path)
            self.preprocessor = load_object(file_path=self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def calculate_feature_contributions(self, features):
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Transform features using preprocessor
            data_scaled = self.preprocessor.transform(features)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(data_scaled)
            
            # Get feature names
            feature_names = features.columns
            
            # Calculate contribution percentages
            contributions = {}
            for i, feature in enumerate(feature_names):
                # Calculate absolute contribution
                abs_contribution = np.abs(shap_values[0][i])
                # Calculate percentage contribution
                total_contribution = np.sum(np.abs(shap_values[0]))
                percentage = (abs_contribution / total_contribution) * 100
                contributions[feature] = round(float(percentage), 2)
            
            return contributions
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            
            # Calculate feature contributions
            feature_contributions = self.calculate_feature_contributions(features)
            
            return preds, feature_contributions

        except Exception as e:
            raise CustomException(e, sys)


class NutritionRiskCustomData:
    def __init__(
        self,
        age: int,
        gender: int,
        height: float,
        weight: float,
        carbohydrate_consumption: float,
        protein_intake: float,
        fat_intake: float,
        regularity_of_meals: int,
        portion_control: int,
        caloric_balance: int,
        sugar_consumption: float,
        DiabetesRisk:float,
        bmi: float,
    ):
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.carbohydrate_consumption = carbohydrate_consumption
        self.protein_intake = protein_intake
        self.fat_intake = fat_intake
        self.regularity_of_meals = regularity_of_meals
        self.portion_control = portion_control
        self.caloric_balance = caloric_balance
        self.sugar_consumption = sugar_consumption
        self.DiabetesRisk = DiabetesRisk
        self.bmi = bmi

    def get_data_as_data_frame(self):
        try:
            nutrition_data_input_dict = {
                "Age": [self.age],
                "Gender": [self.gender],
                "Height": [self.height],
                "Weight": [self.weight],
                "Carbohydrate_Consumption": [self.carbohydrate_consumption],
                "Protein_Intake": [self.protein_intake],
                "Fat_Intake": [self.fat_intake],
                "Regularity_of_Meals": [self.regularity_of_meals],
                "Portion_Control": [self.portion_control],
                "Caloric_Balance": [self.caloric_balance],
                "Sugar_Consumption": [self.sugar_consumption],
                "DiabetesRisk": [self.DiabetesRisk],
                "BMI": [self.bmi],
            }

            return pd.DataFrame(nutrition_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)