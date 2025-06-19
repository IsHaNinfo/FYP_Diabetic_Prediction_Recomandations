from pathlib import Path
import pandas as pd
from NutritonRecommander import MealPlanRecommender
import pickle
import re

def clean_nutritional_value(value):
    """Clean nutritional value by removing units and converting to float"""
    if pd.isna(value):
        return 0.0
    # Remove any text after the number and convert to float
    numeric_value = re.sub(r'[^\d.]', '', str(value).split()[0])
    return float(numeric_value) if numeric_value else 0.0

def precompute_all_embeddings():
    try:
        print("Starting pre-computation of embeddings...")
        
        # Load food data
        BASE_PATH = Path('G:/FYP_Diabetic_Prediction_Recomandations/notebook/data')
        food_df = pd.read_csv(BASE_PATH / 'RecommandationDatasets/NutritionsDatasets/SrilankanCommonFoods.csv')
        
        # Print column names for debugging
        print("Available columns:", food_df.columns.tolist())
        
        # Clean nutritional values
        nutritional_columns = [
            'Calories (kcal)',
            'Carbohydrate (g)',
            'Protein (g)',
            'Fat (g)'
        ]
        
        for col in nutritional_columns:
            if col in food_df.columns:
                food_df[col] = food_df[col].apply(clean_nutritional_value)
            else:
                print(f"Warning: Column {col} not found in dataset")
        
        # Initialize recommender
        recommender = MealPlanRecommender(food_df)
        
        # Pre-compute embeddings
        recommender._precompute_food_embeddings()
        
        # Save embeddings
        output_path = Path('G:/FYP_Diabetic_Prediction_Recomandations/artifact/nutrition_recommendations/food_embeddings.pkl')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(recommender.food_embeddings, f)
            
        print("Successfully pre-computed and saved embeddings")
        
    except Exception as e:
        print(f"Error in pre-computation: {e}")
        raise

if __name__ == "__main__":
    precompute_all_embeddings() 