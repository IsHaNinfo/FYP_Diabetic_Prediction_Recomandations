import sys
import pandas as pd
import numpy as np
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

    def predict_proba(self, features):
        """Get raw model predictions for diabetes risk"""
        try:
            if hasattr(features, 'get_data_as_data_frame'):
                features = features.get_data_as_data_frame()
            
            self._load_model_and_preprocessor()
            data_scaled = PredictPipeline._preprocessor.transform(features)
            
            raw_pred = PredictPipeline._model.predict(data_scaled)
            
            # Debug: Print the raw prediction
            print(f"Raw model prediction: {raw_pred}")
            print(f"Type: {type(raw_pred)}")
            print(f"Shape: {raw_pred.shape if hasattr(raw_pred, 'shape') else 'No shape'}")
            
            # For now, just divide by 100 to get reasonable values
            if not isinstance(raw_pred, np.ndarray):
                raw_pred = np.array([raw_pred])
            
            normalized = raw_pred / 100.0
            normalized = np.clip(normalized, 0, 1)
            
            print(f"Normalized prediction: {normalized}")
            
            return normalized
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict_with_validation(self, features):
        """Get prediction with PIMA model validation"""
        try:
            # Get your model's prediction (already in 0-1 range, e.g., 0.39 = 39%)
            your_prediction = self.predict_proba(features)[0]
            
            # Import here to avoid circular import
            from src.components.Diabetic_Risk_Prediction.risk_validation import validate_and_plot
            
            # Get PIMA validation results
            validation_results = validate_and_plot(features)
            
            # Combine results
            results = {
                "your_model_prediction": round(your_prediction, 4),
                "your_risk_percentage": round(your_prediction * 100, 2),
                "pima_prediction": validation_results.get("pima_prediction", 0.0),
                "pima_risk_percentage": round(validation_results.get("pima_prediction", 0.0) * 100, 2),
                "similar_cases_avg_prediction": validation_results.get("similar_cases_avg_prediction", 0.0),
                "similar_cases_count": validation_results.get("similar_cases_count", 0),
                "brier_score": validation_results.get("brier_score", 0.0),
                "roc_auc": validation_results.get("roc_auc", 0.0),
                "prediction_difference": round(abs(your_prediction - validation_results.get("pima_prediction", 0.0)), 4),
                "risk_level": self._classify_risk_level(your_prediction),
                "pima_risk_level": self._classify_risk_level(validation_results.get("pima_prediction", 0.0)),
                "models_agree": self._check_agreement(your_prediction, validation_results.get("pima_prediction", 0.0))
            }
            
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def _classify_risk_level(self, prediction):
        """Classify risk level based on prediction probability"""
        if prediction < 0.3:
            return "Low Risk"
        elif prediction < 0.5:
            return "Moderate Risk"
        elif prediction < 0.7:
            return "High Risk"
        else:
            return "Very High Risk"

    def _check_agreement(self, your_pred, pima_pred):
        """Check if both models agree on risk level"""
        your_level = self._classify_risk_level(your_pred)
        pima_level = self._classify_risk_level(pima_pred)
        return your_level == pima_level


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
