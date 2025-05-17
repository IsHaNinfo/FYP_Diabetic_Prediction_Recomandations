from src.components.Nutrition_Recommendations.data_preparation import load_user_data, scale_features
from src.components.Nutrition_Recommendations.graph_builder import build_edge_index
from src.components.Nutrition_Recommendations.model import GCN
from src.components.Nutrition_Recommendations.train import train_model
from src.components.Nutrition_Recommendations.inference import predict_nutrition
from src.components.Nutrition_Recommendations.utils import recommend_foods, generate_meal_plan
from src.components.Nutrition_Recommendations.config import FEATURE_COLS, TARGET_COLS, FOOD_DATA_PATH

import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data
user_df = load_user_data()
X, y, scaler = scale_features(user_df)

# 2. Build graph
edge_index = build_edge_index(X, k=5)
x = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=y_tensor)

# 3. Train/val split
indices = list(range(len(X)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_mask = torch.zeros(len(X), dtype=torch.bool)
val_mask = torch.zeros(len(X), dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True

# 4. Train model
model = GCN(input_dim=x.shape[1], hidden_dim=64, output_dim=y.shape[1])
trained_model = train_model(data, model, train_mask, val_mask)

# 5. Inference & evaluation
model.eval()
preds = model(data.x, data.edge_index).detach().cpu().numpy()
true = data.y.detach().cpu().numpy()
preds = scaler.inverse_transform(preds)
true = scaler.inverse_transform(true)
print("MAE:", mean_absolute_error(true, preds))
print("RMSE:", mean_squared_error(true, preds) ** 0.5)
print("R² Score:", r2_score(true, preds))

# 6. Food recommendation & meal plan
food_df = pd.read_csv(FOOD_DATA_PATH)
predicted_nutrition = predict_nutrition(model, data, TARGET_COLS)
first_user_pred = predicted_nutrition.iloc[0]
recommended = recommend_foods(food_df, first_user_pred["Protein_Intake"], first_user_pred["Fat_Intake"], first_user_pred["Carbohydrate_Consumption"])
print(recommended.head())
meal_plan = generate_meal_plan(first_user_pred, food_df)
from pprint import pprint
pprint(meal_plan)