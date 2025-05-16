from src.components.Nutrition_Recommendations.data_preparation import (
    load_user_data,
    scale_features,
)
from src.components.Nutrition_Recommendations.graph_builder import build_edge_index
from src.components.Nutrition_Recommendations.model import GCN
from src.components.Nutrition_Recommendations.train import train_model
from src.components.Nutrition_Recommendations.inference import predict_nutrition
from src.components.Nutrition_Recommendations.utils import (
    recommend_foods,
    generate_meal_plan,
)
from src.components.Nutrition_Recommendations.config import (
    DATA_PATH,
    SIMILARITY_THRESHOLD,
    FOOD_DATA_PATH,
    FEATURE_COLS,
    TARGET_COLS,
)
import torch
from torch_geometric.data import Data
import pandas as pd
import torch

# 1. Load and prepare user data
user_df = load_user_data()
X, y, scaler = scale_features(user_df)

# 2. Build graph
edge_index = build_edge_index(X, SIMILARITY_THRESHOLD)
x = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=y_tensor)

# 3. Train model
# 3. Train model
model = train_model(data, input_dim=x.shape[1], output_dim=y.shape[1])

# 4. Save the trained model
import os

os.makedirs("artifact/nutrition_recommendations", exist_ok=True)
torch.save(model.state_dict(), "artifact/nutrition_recommendations/model.pkl")

# 4. Inference
predicted_nutrition = predict_nutrition(model, data, TARGET_COLS)
print(predicted_nutrition.head())

# 5. Recommend foods for the first user
food_df = pd.read_csv(FOOD_DATA_PATH)
first_user_pred = predicted_nutrition.iloc[0]
recommended = recommend_foods(
    food_df,
    protein_target=first_user_pred["Protein_Intake"],
    fat_target=first_user_pred["Fat_Intake"],
    carbs_target=first_user_pred["Carbohydrate_Consumption"],
)
print("\nRecommended Foods for User 1:")
print(
    recommended[
        ["Food", "Measure", "Grams", "Calories", "Protein", "Fat", "Carbs", "Category"]
    ]
)

# 6. Generate meal plan for the first user
meal_plan = generate_meal_plan(first_user_pred, food_df)
from pprint import pprint

print("\n7-Day Meal Plan for User 1:")
pprint(meal_plan)
