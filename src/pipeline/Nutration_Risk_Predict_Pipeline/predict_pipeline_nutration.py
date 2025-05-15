import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class NutritionRiskPredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        
        try:
            model_path = os.path.join("artifact/nutrition", "model.pkl")
            preprocessor_path = os.path.join(
                "artifact/nutrition", "nutrition_preprocessor.pkl"
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