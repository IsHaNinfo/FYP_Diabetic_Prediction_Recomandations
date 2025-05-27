import pandas as pd
from src.components.Exersices_Recommendations.recommender import recommend_exercises
from src.components.Exersices_Recommendations.data_loader import load_user_data, load_gym_data
from src.components.Exersices_Recommendations.config import FEATURE_COLS

class ExercisesRecommendationsCustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        return pd.DataFrame([self.data])

class ExercisesRecommendationsPredictPipeline:
    def __init__(self):
        self.gym_df = load_gym_data()

    def predict(self, user_df):
        # Assume user_df is a DataFrame with one row
        user_row = user_df.iloc[0]
        recommended = recommend_exercises(user_row, self.gym_df)
        return recommended.to_dict(orient="records")