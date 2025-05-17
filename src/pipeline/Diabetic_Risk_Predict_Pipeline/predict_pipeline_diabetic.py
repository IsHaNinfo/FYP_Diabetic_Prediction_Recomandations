import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join(
                "artifact/common", "diabetic_preprocessor.pkl"
            )
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
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
        bmi: float,  # Added BMI
        risk_level: float,  # Added RiskLevel
    ):
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.waist_circumference = waist_circumference
        self.diet_food_habits = diet_food_habits
        self.family_history = family_history
        self.high_blood_pressure = high_blood_pressure
        self.cholesterol_lipid_levels = cholesterol_lipid_levels
        self.thirst = thirst
        self.fatigue = fatigue
        self.urination = urination
        self.vision_changes = vision_changes
        self.bmi = bmi  # Store BMI
        self.risk_level = risk_level  # Store RiskLevel

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
                "BMI": [self.bmi],  # Include BMI in the DataFrame
                "RiskLevel": [self.risk_level],  # Include RiskLevel in the DataFrame
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
