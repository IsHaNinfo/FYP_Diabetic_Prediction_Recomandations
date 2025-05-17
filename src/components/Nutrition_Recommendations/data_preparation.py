import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.components.Nutrition_Recommendations.config import DATA_PATH, FEATURE_COLS, TARGET_COLS

def load_user_data():
    df = pd.read_csv(DATA_PATH)
    return df

def scale_features(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y = scaler.fit_transform(df[TARGET_COLS])
    return X, y, scaler