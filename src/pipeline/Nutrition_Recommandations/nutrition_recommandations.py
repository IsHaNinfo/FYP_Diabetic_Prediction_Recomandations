import torch
import pandas as pd
import os
from src.components.Nutrition_Recommendations.model import GraphSAGE
from src.components.Nutrition_Recommendations.data_preparation import scale_features
from src.components.Nutrition_Recommendations.graph_builder import build_edge_index
from src.components.Nutrition_Recommendations.config import FEATURE_COLS, TARGET_COLS
from src.components.Nutrition_Recommendations.utils import calculate_nutrition_targets

class NutritionRecommendationsPredictPipeline:
    _model = None
    _scaler = None
    
    def __init__(self):
        self.model_path = os.path.join("artifact/nutrition_recommendations", "gcn_model.pkl")
        self.input_dim = len(FEATURE_COLS)
        self.output_dim = len(TARGET_COLS)

    def _load_model(self):
        if NutritionRecommendationsPredictPipeline._model is None:
            NutritionRecommendationsPredictPipeline._model = GraphSAGE(
                input_dim=self.input_dim, 
                hidden_dim=128, 
                output_dim=self.output_dim
            )
            NutritionRecommendationsPredictPipeline._model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device('cpu'))
            )
            NutritionRecommendationsPredictPipeline._model.eval()

    def predict(self, user_df):
        # Scale features
        X, _, scaler = scale_features(user_df)
        self._scaler = scaler

        # Handle single-user prediction (no k-NN possible)
        if len(X) == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop
        else:
            edge_index = build_edge_index(X, k=min(5, len(X)-1))  # avoid k > n-1

        x = torch.tensor(X, dtype=torch.float)
        self._load_model()
        
        with torch.no_grad():
            preds = NutritionRecommendationsPredictPipeline._model(x, edge_index).cpu().numpy()
        preds = self._scaler.inverse_transform(preds)
        return preds

class NutritionRecommendationsCustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        return pd.DataFrame([self.data])