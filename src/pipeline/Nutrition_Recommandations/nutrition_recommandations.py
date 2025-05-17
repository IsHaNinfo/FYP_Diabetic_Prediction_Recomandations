import torch
import pandas as pd
import os
from src.components.Nutrition_Recommendations.model import GCN
from src.components.Nutrition_Recommendations.data_preparation import scale_features
from src.components.Nutrition_Recommendations.graph_builder import build_edge_index
from src.components.Nutrition_Recommendations.config import FEATURE_COLS, TARGET_COLS

class NutritionRecommendationsPredictPipeline:
    def __init__(self):
        artifact_dir = "artifact/nutrition_recommendations"
        model_path = os.path.join(artifact_dir, "gcn_model.pkl")
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.input_dim = len(FEATURE_COLS)
        self.output_dim = len(TARGET_COLS)
        self._load_model()

    def _load_model(self):
        self.model = GCN(input_dim=self.input_dim, hidden_dim=64, output_dim=self.output_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, user_df):
        # Scale features
        X, _, scaler = scale_features(user_df)
        self.scaler = scaler
        edge_index = build_edge_index(X, k=5)
        x = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            preds = self.model(x, edge_index).cpu().numpy()
        preds = self.scaler.inverse_transform(preds)
        return preds

class NutritionRecommendationsCustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        return pd.DataFrame([self.data])