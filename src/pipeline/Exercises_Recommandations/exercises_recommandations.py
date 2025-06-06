import pandas as pd
import re

class ExercisesRecommendationsCustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        return pd.DataFrame([self.data])


    def build_prompt(data):
        return (
        f"Age: {data.get('age')}, Gender: {data.get('gender')}, Height: {data.get('height')} cm, "
        f"Weight: {data.get('weight')} kg, Energy Levels: {data.get('energy_levels')}, "
        f"Physical Activity: {data.get('physical_activity')}, Sitting Time: {data.get('sitting_time')}, "
        f"Cardiovascular Health: {data.get('cardiovascular_health')}, Muscle Strength: {data.get('muscle_strength')}, "
        f"Flexibility: {data.get('flexibility')}, Balance: {data.get('balance')}, Thirsty: {data.get('thirsty')}, "
        f"Pain or Discomfort: {data.get('pain_or_discomfort')}, Available Time: {data.get('available_time')} minutes/week, "
        f"Diabetes Risk: {data.get('diabetes_risk')}, Nutrition Risk: {data.get('nutrition_risk')}. "
        f"Recommend a personalized workout."
    )

def format_paragraphs(text):
    parts = re.split(r'(?:(?<=\n)|(?<=\.))\s*(?=(?:-|\d+\.|\•|[A-Z]))', text.strip())
    paragraphs = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(paragraphs) 