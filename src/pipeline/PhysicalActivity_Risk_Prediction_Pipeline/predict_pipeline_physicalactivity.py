import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PhysicalRiskPredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Corrected model and preprocessor paths
            model_path = os.path.join("artifact/physical", "model.pkl")
            preprocessor_path = os.path.join(
                "artifact/physical", "physical_preprocessor.pkl"
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