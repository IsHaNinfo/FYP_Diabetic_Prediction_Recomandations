import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
from src.utils import to_01


class PredictPipeline:
    _model = None
    _preprocessor = None
    
    def __init__(self):
        pass

    def _load_model_and_preprocessor(self):
        if PredictPipeline._model is None or PredictPipeline._preprocessor is None:
            model_path = os.path.join("artifact/common", "model.pkl")
            preprocessor_path = os.path.join(
                "artifact/common", "diabetic_preprocessor.pkl"
            )
            PredictPipeline._model = load_object(file_path=model_path)
            PredictPipeline._preprocessor = load_object(file_path=preprocessor_path)

    def predict(self, features):
        try:
            if hasattr(features, 'get_data_as_data_frame'):
                features = features.get_data_as_data_frame()
            
            self._load_model_and_preprocessor()
            data_scaled = PredictPipeline._preprocessor.transform(features)
            preds = PredictPipeline._model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        gender: str,
        height: float,
        weight: float,
        waist_circumference: float,
        diet_food_habits: int,
        family_history: float,
        high_blood_pressure: float,
        cholesterol_lipid_levels: float,
        thirst: float,
        fatigue: float,
        urination: float,
        vision_changes: float,
        bmi: float,
        risk_level: float,
    ):
        
        
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.waist_circumference = waist_circumference
        self.diet_food_habits = diet_food_habits
        self.family_history = to_01(family_history)
        self.high_blood_pressure = to_01(high_blood_pressure)
        self.cholesterol_lipid_levels = to_01(cholesterol_lipid_levels)
        self.thirst = to_01(thirst)
        self.fatigue = to_01(fatigue)
        self.urination = to_01(urination)
        self.vision_changes = to_01(vision_changes)
        self.bmi = bmi
        self.risk_level = to_01(risk_level)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Gender": [self.gender],
                "Height": [self.height],
                "Weight": [self.weight],
                "Waist_Circumference": [self.waist_circumference],
                "Diet_Food_Habits": [self.diet_food_habits],
                "Family_History": [self.family_history],
                "Blood_Pressure": [self.high_blood_pressure],
                "Cholesterol_Lipid_Levels": [self.cholesterol_lipid_levels],
                "Thirst": [self.thirst],
                "Fatigue": [self.fatigue],
                "Urination": [self.urination],
                "Vision Changes": [self.vision_changes],
                "BMI": [self.bmi],
                "RiskLevel": [self.risk_level],
            }

            df = pd.DataFrame(custom_data_input_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)
