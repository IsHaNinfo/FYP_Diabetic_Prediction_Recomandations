import sys
import pandas as pd
import numpy as np
import shap
from src.exception import CustomException
from src.utils import load_object
import os


class PhysicalRiskPredictPipeline:
    _model = None
    _preprocessor = None
    
    def __init__(self):
        pass

    def _load_model_and_preprocessor(self):
        if PhysicalRiskPredictPipeline._model is None or PhysicalRiskPredictPipeline._preprocessor is None:
            model_path = os.path.join("artifact/physical", "model.pkl")
            preprocessor_path = os.path.join(
                "artifact/physical", "physical_preprocessor.pkl"
            )
            PhysicalRiskPredictPipeline._model = load_object(file_path=model_path)
            PhysicalRiskPredictPipeline._preprocessor = load_object(file_path=preprocessor_path)

    def calculate_feature_contributions(self, features):
        try:
            # Transform features using preprocessor
            data_scaled = PhysicalRiskPredictPipeline._preprocessor.transform(features)
            
            # Create SHAP explainer for decision tree
            explainer = shap.TreeExplainer(PhysicalRiskPredictPipeline._model)
            
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
            self._load_model_and_preprocessor()
            data_scaled = PhysicalRiskPredictPipeline._preprocessor.transform(features)
            preds = PhysicalRiskPredictPipeline._model.predict(data_scaled)
            
            # Calculate feature contributions using SHAP
            feature_contributions = self.calculate_feature_contributions(features)
            
            return preds, feature_contributions
            
        except Exception as e:
            raise CustomException(e, sys)


class PhysicalRiskCustomData:
    def __init__(
        self,
        age: int,
        gender: int,
        height: float,
        weight: float,
        energy_levels: float,
        physical_activity: float,
        sitting_time: float,
        cardiovascular_health: int,
        muscle_strength: int,
        flexibility: float,
        balance: float,
        thirsty: float,
        pain_or_discomfort: float,
        available_time: float,
        DiabetesRisk: float,
        bmi: float,
    ):
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.energy_levels = energy_levels
        self.physical_activity = physical_activity
        self.sitting_time = sitting_time
        self.cardiovascular_health = cardiovascular_health
        self.muscle_strength = muscle_strength
        self.flexibility = flexibility
        self.balance = balance
        self.thirsty = thirsty
        self.pain_or_discomfort = pain_or_discomfort
        self.available_time = available_time
        self.DiabetesRisk = DiabetesRisk
        self.bmi = bmi

    def get_data_as_data_frame(self):
        try:
            physical_data_input_dict = {
                "Age": [self.age],
                "Gender": [self.gender],
                "Height": [self.height],
                "Weight": [self.weight],
                "EnergyLevels": [self.energy_levels],
                "Physical_Activity": [self.physical_activity],
                "Sitting_Time": [self.sitting_time],
                "Cardiovascular_Health": [self.cardiovascular_health],
                "Muscle_Strength": [self.muscle_strength],
                "Flexibility": [self.flexibility],
                "Balance": [self.balance],
                "Thirsty": [self.thirsty],
                "Pain_or_Discomfort": [self.pain_or_discomfort],
                "Available_Time": [self.available_time],
                "DiabetesRisk": [self.DiabetesRisk],
                "BMI": [self.bmi],
            }

            return pd.DataFrame(physical_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)