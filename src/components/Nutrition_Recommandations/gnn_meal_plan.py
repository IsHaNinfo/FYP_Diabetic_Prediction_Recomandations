import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch.nn import Linear
from torch_geometric.nn import GATConv, HeteroConv
import torch.nn.functional as F
import json
import os
from collections import defaultdict

# Set the base directory for the datasets
base_dir = os.path.join('notebook', 'data', 'RecommandationDatasets', 'NutritionDatasets')

# Load datasets
user_df = pd.read_csv(os.path.join(base_dir, 'Updated_User_Nutrition_Parameters.csv'))

def load_food_df():
    return pd.read_csv(os.path.join(base_dir, 'Foods_Datasets.csv'))

disease_df = pd.read_csv(os.path.join(base_dir, 'diseases.csv'))
nutrition_df = pd.read_csv(os.path.join(base_dir, 'nutrients.csv'))
edges = pd.read_csv(os.path.join(base_dir, 'edges.csv'))
# Create HeteroData graph
data = HeteroData()

# Add user features
user_features = user_df[['Age', 'Gender', 'Height', 'Weight', 'Carbohydrate_Consumption',
                         'Protein_Intake', 'Fat_Intake', 'Regularity_of_Meals', 'Portion_Control',
                         'Caloric_Balance', 'Sugar_Consumption', 'BMI',
                         'DiabetesRisk', 'NutritionRisk']].astype(float)
data['user'].x = torch.tensor(user_features.values, dtype=torch.float)
user_id_map = {uid: i for i, uid in enumerate(user_df['user_id'])}

# Add food features
food_df = load_food_df()
food_features = food_df[['calories', 'carbs', 'protein', 'fat', 'glycemic_index', 'estimated_weight_g']].astype(float)
data['food'].x = torch.tensor(food_features.values, dtype=torch.float)
food_id_map = {fid: i for i, fid in enumerate(food_df['food_id'])}

# Add nutrient and disease nodes
data['nutrient'].x = torch.eye(len(nutrition_df))
data['disease'].x = torch.eye(len(disease_df))

# Add edges
edge_index_dict = defaultdict(list)
for _, row in edges.iterrows():
    src, rel, tgt = row['source'], row['relation'], row['target']
    
    if rel == "hasRisk" and src in user_id_map:
        edge_index_dict[('user', 'hasRisk', 'disease')].append([user_id_map[src], 0])
    
    elif rel == "contains" and src in food_id_map:
        edge_index_dict[('food', 'contains', 'nutrient')].append([food_id_map[src], int(tgt[1:]) - 1])

# Convert edges to tensors
for edge_type, edge_list in edge_index_dict.items():
    edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    data[edge_type].edge_index = edge_tensor

# Add reverse edges for message passing INTO 'user' and 'food'
if ('user', 'hasRisk', 'disease') in data.edge_index_dict:
    edge = data['user', 'hasRisk', 'disease'].edge_index
    data['disease', 'rev_hasRisk', 'user'].edge_index = edge.flip(0)

if ('food', 'contains', 'nutrient') in data.edge_index_dict:
    edge = data['food', 'contains', 'nutrient'].edge_index
    data['nutrient', 'rev_contains', 'food'].edge_index = edge.flip(0)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'hasRisk', 'disease'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('food', 'contains', 'nutrient'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('disease', 'rev_hasRisk', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('nutrient', 'rev_contains', 'food'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='sum')
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: self.lin(v) for k, v in x_dict.items()}
        return x_dict

# Initialize and train the model
model = GNN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Example training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    # Dummy loss for demonstration
    loss = torch.tensor(0.0, requires_grad=True)
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)
user_emb = out['user']
food_emb = out['food']

# Generate meal plan
scores = torch.matmul(user_emb, food_emb.T)
top_k = 5
_, top_indices = torch.topk(scores, k=top_k, dim=1)

# Define suitability function
def is_suitable_risk_aware(food, user):
    diabetes_risk = user["diabetes_risk"]
    if diabetes_risk > 0.6 and (food["glycemic_index"] > 55 or food["carbs"] > 60):
        return False
    if user["bmi"] > 25 and (food["calories"] > 700 or food["fat"] > 25):
        return False
    if food["estimated_weight_g"] > 800:
        return False
    if "preferences" in user and food.get("culture") != user["preferences"]:
        return False
    return True

# Define meal plan generator
def generate_health_aware_meal_plan(user_data, food_df):
    used_food_ids = set()
    meal_plan = {}
    categories = ["Breakfast", "Lunch", "Dinner", "Snack"]

    for day in range(1, 8):
        daily_plan = {}
        for meal in categories:
            options = food_df[
                (food_df["meal_type"] == meal) &
                (~food_df["food_id"].isin(used_food_ids))
            ].copy()

            # Apply the suitability function row-wise
            options = options[options.apply(lambda row: is_suitable_risk_aware(row.to_dict(), user_data), axis=1)]

            if options.empty:
                daily_plan[meal] = {
                    "food": "No suitable meal",
                    "estimated_weight_g": 0,
                    "calories": 0,
                    "carbs": 0,
                    "protein": 0,
                    "fat": 0,
                    "glycemic_index": "-"
                }
            else:
                chosen = options.sample(1).iloc[0]
                used_food_ids.add(str(chosen["food_id"]))
                daily_plan[meal] = {
                    "food": str(chosen["food_item"]),
                    "estimated_weight_g": int(chosen["estimated_weight_g"]),
                    "calories": float(chosen["calories"]),
                    "carbs": float(chosen["carbs"]),
                    "protein": float(chosen["protein"]),
                    "fat": float(chosen["fat"]),
                    "glycemic_index": int(chosen["glycemic_index"])
                }

        meal_plan[f"Day {day}"] = daily_plan

    return meal_plan


def analyze_meal_plan_with_contribution(meal_plan, user_data):
    initial_diabetes_risk = user_data["diabetes_risk"]
    initial_nutrition_risk = user_data["nutrition_risk"]
    
    updated_plan = {}
    total_diabetes_reduction = 0.0
    total_nutrition_reduction = 0.0

    for day_key, meals in meal_plan.items():
        day_info = {}
        for meal_type, meal in meals.items():
            contribution = {
                "diabetes_reduction": 0.0,
                "nutrition_reduction": 0.0
            }

            if meal["food"] == "No suitable meal":
                meal["contribution"] = contribution
                day_info[meal_type] = meal
                continue

            # Diabetes Risk Contributions
            if meal["glycemic_index"] != "-" and int(meal["glycemic_index"]) <= 55:
                contribution["diabetes_reduction"] += 0.02
            if meal["carbs"] <= 60:
                contribution["diabetes_reduction"] += 0.02

            # Nutrition Risk Contributions
            if meal["calories"] <= 700:
                contribution["nutrition_reduction"] += 0.02
            if meal["fat"] <= 25:
                contribution["nutrition_reduction"] += 0.02
            if meal["estimated_weight_g"] <= 800:
                contribution["nutrition_reduction"] += 0.02

            meal["contribution"] = contribution
            total_diabetes_reduction += contribution["diabetes_reduction"]
            total_nutrition_reduction += contribution["nutrition_reduction"]
            day_info[meal_type] = meal
        
        updated_plan[day_key] = day_info

    # Estimate new risks
    new_diabetes_risk = max(0.0, initial_diabetes_risk - total_diabetes_reduction)
    new_nutrition_risk = max(0.0, initial_nutrition_risk - total_nutrition_reduction)

    return {
        "updated_meal_plan": updated_plan,
        "summary": {
            "initial_risks": {
                "diabetes_risk": round(initial_diabetes_risk, 4),
                "nutrition_risk": round(initial_nutrition_risk, 4)
            },
            "total_contributions": {
                "diabetes_risk_reduced_by": round(total_diabetes_reduction, 4),
                "nutrition_risk_reduced_by": round(total_nutrition_reduction, 4)
            },
            "final_risks": {
                "estimated_diabetes_risk": round(new_diabetes_risk, 4),
                "estimated_nutrition_risk": round(new_nutrition_risk, 4)
            }
        }
    }
