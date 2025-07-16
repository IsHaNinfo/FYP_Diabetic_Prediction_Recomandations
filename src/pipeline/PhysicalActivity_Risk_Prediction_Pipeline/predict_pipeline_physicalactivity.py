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
        
"""
Here's a clear explanation of why SHAP is valuable in your health prediction system:

1. **Understanding "Why" Behind Predictions**
   - Instead of just getting a risk score number, SHAP shows exactly which factors contributed to that score
   - For example, if someone gets a 75% risk score, SHAP explains that it's because:
     * Physical activity contributed 18.7%
     * Sitting time contributed 15.3%
     * Energy levels contributed 12.1%
     * And so on...

2. **Personalized Health Insights**
   - Shows which health factors are most important for each individual
   - Helps users understand their specific risk factors
   - Guides them toward the most relevant health improvements
   - Makes health recommendations more targeted and effective

3. **Building Trust in the System**
   - Makes the prediction system transparent
   - Users can see why they received a particular risk score
   - Healthcare providers can explain predictions to patients
   - Increases confidence in the health recommendations

4. **Better Health Decision Making**
   - Helps prioritize which health factors to address first
   - Shows the relative importance of different risk factors
   - Guides users toward the most impactful lifestyle changes
   - Supports healthcare providers in making treatment decisions

5. **Real-world Health Impact**
   - Instead of just saying "your risk is high", it explains:
     * Which specific factors are causing the high risk
     * How much each factor contributes
     * What areas need the most attention
   - Makes health recommendations more actionable and specific

6. **Healthcare Provider Benefits**
   - Helps doctors understand the model's reasoning
   - Supports evidence-based treatment planning
   - Makes it easier to explain risks to patients
   - Guides intervention strategies

7. **User Empowerment**
   - Users understand their health risks better
   - Can make informed decisions about lifestyle changes
   - See clear connections between their behaviors and health risks
   - Feel more in control of their health outcomes

8. **Quality of Health Recommendations**
   - Recommendations become more specific and targeted
   - Users know exactly which factors to focus on
   - Health improvements can be more effectively prioritized
   - Better tracking of health progress

9. **Transparency in Health Predictions**
   - No "black box" predictions
   - Clear explanation of how each factor affects health risk
   - Understandable breakdown of risk factors
   - Builds confidence in the health assessment

10. **Practical Health Applications**
    - Helps users make better lifestyle choices
    - Guides healthcare interventions
    - Supports preventive health measures
    - Makes health risk assessment more meaningful

In your health prediction system, SHAP transforms it from just a number-generating tool into a comprehensive health insight system that:
- Explains health risks clearly
- Guides health improvements effectively
- Builds trust in the predictions
- Supports better health decisions
- Makes health recommendations more actionable

This is particularly important in healthcare, where understanding the reasons behind health assessments is crucial for effective health management and improvement.
"""