import pandas as pd
from .config import EXERCISE_DATA_PATH, GYM_DATA_PATH

def load_user_data():
    return pd.read_csv(EXERCISE_DATA_PATH)

def load_gym_data():
    return pd.read_csv(GYM_DATA_PATH)