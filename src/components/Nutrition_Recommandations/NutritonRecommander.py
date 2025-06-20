import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from pathlib import Path
import time
import json
import re

class MealPlanRecommender:
    def __init__(self, food_df, user_profile=None):
        self.food_df = food_df
        self.user_profile = user_profile
        self.used_foods = set()  # Track used foods
        self.food_embeddings = {}  # Initialize empty embeddings dictionary
        
    def _adjust_criteria_based_on_risks(self, base_criteria):
        """Adjust meal criteria based on diabetic and nutrition risks"""
        if not self.user_profile:
            return base_criteria
            
        adjusted_criteria = base_criteria.copy()
        
        # Get risk levels - handle both DataFrame and dictionary input
        if isinstance(self.user_profile, pd.DataFrame):
            diabetic_risk = self.user_profile['DiabetesRisk'].iloc[0]
            nutrition_risk = self.user_profile['NutritionRisk'].iloc[0]
        else:
            diabetic_risk = self.user_profile.get('DiabetesRisk', 50)  # Default to moderate risk
            nutrition_risk = self.user_profile.get('NutritionRisk', 50)  # Default to moderate risk
        
        # Adjust based on diabetic risk
        if diabetic_risk >= 70:  # High risk
            for meal_type in adjusted_criteria:
                adjusted_criteria[meal_type]['max_carbs'] *= 0.7
                adjusted_criteria[meal_type]['min_fiber'] *= 1.3
        elif diabetic_risk >= 40:  # Moderate risk
            for meal_type in adjusted_criteria:
                adjusted_criteria[meal_type]['max_carbs'] *= 0.85
                adjusted_criteria[meal_type]['min_fiber'] *= 1.2
                
        # Adjust based on nutrition risk
        if nutrition_risk >= 70:  # High risk
            for meal_type in adjusted_criteria:
                adjusted_criteria[meal_type]['min_protein'] *= 1.2
                adjusted_criteria[meal_type]['min_fiber'] *= 1.2
                adjusted_criteria[meal_type]['max_fat'] *= 0.8
        elif nutrition_risk >= 40:  # Moderate risk
            for meal_type in adjusted_criteria:
                adjusted_criteria[meal_type]['min_protein'] *= 1.1
                adjusted_criteria[meal_type]['min_fiber'] *= 1.1
                
        return adjusted_criteria

    def _is_suitable_for_meal(self, food, meal_type, lenient=False):
        """Check if food is suitable for the meal type"""
        # Base criteria for all meals
        base_criteria = {
            'breakfast': {
                'min_calories': 50 if lenient else 100,
                'max_calories': 600 if lenient else 500,
                'min_protein': 1 if lenient else 2,
                'max_protein': 25 if lenient else 20,
                'min_carbs': 5 if lenient else 10,
                'max_carbs': 70 if lenient else 60,
                'max_fat': 20 if lenient else 15
            },
            'lunch': {
                'min_calories': 150 if lenient else 200,
                'max_calories': 900 if lenient else 800,
                'min_protein': 3 if lenient else 5,
                'max_protein': 35 if lenient else 30,
                'min_carbs': 15 if lenient else 20,
                'max_carbs': 90 if lenient else 80,
                'max_fat': 30 if lenient else 25
            },
            'dinner': {
                'min_calories': 150 if lenient else 200,
                'max_calories': 800 if lenient else 700,
                'min_protein': 3 if lenient else 5,
                'max_protein': 35 if lenient else 30,
                'min_carbs': 15 if lenient else 20,
                'max_carbs': 70 if lenient else 60,
                'max_fat': 30 if lenient else 25
            },
            'snack': {
                'min_calories': 30 if lenient else 50,
                'max_calories': 350 if lenient else 300,
                'min_protein': 0 if lenient else 1,
                'max_protein': 20 if lenient else 15,
                'min_carbs': 3 if lenient else 5,
                'max_carbs': 45 if lenient else 40,
                'max_fat': 20 if lenient else 15
            }
        }
        
        try:
            # Extract quantity and convert to standard 100g serving
            quantity_str = str(food['Quantity']).lower().strip()
            
            # Convert different unit types to grams
            if 'tbsp' in quantity_str:
                # 1 tbsp is approximately 15g
                quantity = float(quantity_str.split('tbsp')[0].strip()) * 15
            elif 'tsp' in quantity_str:
                # 1 tsp is approximately 5g
                quantity = float(quantity_str.split('tsp')[0].strip()) * 5
            elif 'cup' in quantity_str:
                # 1 cup is approximately 240g
                quantity = float(quantity_str.split('cup')[0].strip()) * 240
            elif 'ml' in quantity_str:
                # 1 ml is approximately 1g for most liquids
                quantity = float(quantity_str.split('ml')[0].strip())
            elif 'g' in quantity_str:
                quantity = float(quantity_str.split('g')[0].strip())
            else:
                # Default to 100g if unit is not recognized
                quantity = 100
            
            conversion_factor = 100 / quantity

            # Convert nutritional values to per 100g basis
            calories = float(food['Calories (kcal)']) * conversion_factor
            protein = float(food['Protein (g)']) * conversion_factor
            carbs = float(food['Carbohydrate (g)']) * conversion_factor
            fat = float(food['Fat (g)']) * conversion_factor
            
            # Get criteria for this meal type
            criteria = base_criteria[meal_type]
            
            # More lenient checks for high-risk users
            if self.user_profile is not None and not lenient:
                diabetic_risk = self.user_profile.get('DiabetesRisk', 50)
                nutrition_risk = self.user_profile.get('NutritionRisk', 50)
                
                if diabetic_risk >= 70:
                    criteria['max_carbs'] *= 0.8
                if nutrition_risk >= 70:
                    criteria['max_fat'] *= 0.8
            
            # Check nutritional criteria
            if not (criteria['min_calories'] <= calories <= criteria['max_calories'] and
                    criteria['min_protein'] <= protein <= criteria['max_protein'] and
                    criteria['min_carbs'] <= carbs <= criteria['max_carbs'] and
                    fat <= criteria['max_fat']):
                return False
                
            return True
            
        except (KeyError, ValueError) as e:
            print(f"Error processing food: {food['Food'] if 'Food' in food else 'Unknown'}")
            print(f"Error details: {str(e)}")
            return False

    def get_meal_plan(self, num_days=7):
        """Generate a meal plan with variety"""
        meal_plans = []
        
        # Generate all meal plans
        for day in range(num_days):
            day_plan = {
                "day": day + 1,
                "meals": {
                    "breakfast": [],
                    "lunch": [],
                    "dinner": [],
                    "snack": []
                }
            }
            
            day_used_foods = set()
            
            for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
                try:
                    # Try to get suitable foods with initial criteria
                    suitable_foods = self.food_df[self.food_df.apply(
                        lambda x: self._is_suitable_for_meal(x, meal_type) and 
                                 x['Food'] not in day_used_foods, 
                        axis=1
                    )]
                    
                    # If not enough foods, try with more lenient criteria
                    if len(suitable_foods) < 2:
                        suitable_foods = self.food_df[self.food_df.apply(
                            lambda x: self._is_suitable_for_meal(x, meal_type, lenient=True) and 
                                     x['Food'] not in day_used_foods, 
                        axis=1
                    )]
                    
                    # If still not enough foods, try without the day_used_foods restriction
                    if len(suitable_foods) < 2:
                        suitable_foods = self.food_df[self.food_df.apply(
                            lambda x: self._is_suitable_for_meal(x, meal_type, lenient=True), 
                            axis=1
                        )]
                    
                    # If still no foods found, use any available food for this meal type
                    if len(suitable_foods) == 0:
                        suitable_foods = self.food_df.sample(n=min(2, len(self.food_df)))
                    
                    # Select foods for the meal
                    selected_foods = suitable_foods.sample(n=min(2, len(suitable_foods)))
                    
                    # Add selected foods to day's plan
                    day_plan["meals"][meal_type] = [
                        {
                            'food_name': food['Food'],
                            'quantity': food['Quantity'],
                            'nutritional_info': {
                                'calories': float(food['Calories_(kcal)']),
                                'carbs': float(food['Carbohydrate_(g)']),
                                'protein': float(food['Protein_(g)']),
                                'fat': float(food['Fat_(g)']),
                                'quantity': food['Quantity']
                            },
                            'meal_type': meal_type,
                            'health_benefits': self._get_health_benefits(food)
                        }
                        for _, food in selected_foods.iterrows()
                    ]
                    
                    # Track used foods
                    for _, food in selected_foods.iterrows():
                        day_used_foods.add(food['Food'])
                        self.used_foods.add(food['Food'])
                        
                except Exception as e:
                    print(f"Error processing meal type {meal_type}: {str(e)}")
                    continue
            
            meal_plans.append(day_plan)
            
            # Clear used foods after 3 days
            if day % 3 == 0:
                self.used_foods.clear()
        
        return {"meal_plan": meal_plans}
        
    def _get_health_benefits(self, food):
        """Get health benefits of the food"""
        benefits = []
        
        try:
            # Extract quantity and convert to standard 100g serving
            quantity_str = food['Quantity']
            quantity = float(quantity_str.split('g')[0].strip()) if 'g' in quantity_str else float(quantity_str.split('ml')[0].strip())
            conversion_factor = 100 / quantity

            # Convert nutritional values to per 100g basis
            protein = float(food['Protein (g)']) * conversion_factor
            carbs = float(food['Carbohydrate (g)']) * conversion_factor
            fat = float(food['Fat (g)']) * conversion_factor
            
            if protein >= 15:
                benefits.append("Good source of protein")
                
            if carbs <= 20:
                benefits.append("Low in carbohydrates")
                
            if fat <= 5:
                benefits.append("Low in fat")
                
            # Add risk-specific benefits
            if self.user_profile is not None:
                diabetic_risk = self.user_profile.get('DiabetesRisk', 50)
                nutrition_risk = self.user_profile.get('NutritionRisk', 50)
                
                if diabetic_risk >= 70 and carbs <= 15:
                    benefits.append("Suitable for diabetes management")
                    
                if nutrition_risk >= 70 and protein >= 20:
                    benefits.append("High protein content beneficial for nutritional needs")
                
        except (KeyError, ValueError) as e:
            print(f"Error getting health benefits for {food['Food'] if 'Food' in food else 'Unknown'}")
            print(f"Error details: {str(e)}")
            
        return benefits

    def _precompute_food_embeddings(self):
        """Pre-compute embeddings for all foods"""
        def clean_nutritional_value(value):
            """Clean nutritional value by removing units and converting to float"""
            if pd.isna(value):
                return 0.0
            # Remove any text after the number and convert to float
            numeric_value = re.sub(r'[^\d.]', '', str(value).split()[0])
            return float(numeric_value) if numeric_value else 0.0

        for _, food in self.food_df.iterrows():
            try:
                self.food_embeddings[food['Food']] = {
                    'calories': clean_nutritional_value(food['Calories (kcal)']),
                    'carbs': clean_nutritional_value(food['Carbohydrate (g)']),
                    'protein': clean_nutritional_value(food['Protein (g)']),
                    'fat': clean_nutritional_value(food['Fat (g)']),
                    'quantity': food['Quantity']
                }
            except Exception as e:
                print(f"Error processing food {food['Food']}: {str(e)}")
                continue

def get_meal_plan(food_df, user_profile):
    """Get meal plan"""
    try:
        # Ensure all numeric columns are float
        numeric_columns = ['Calories (kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Fat (g)', 
                         'Fiber_(g)', 'Free_sugars_(g)', 'Cholesterol_(mg)', 'Sodium_(mg)']
        
        for col in numeric_columns:
            if col in food_df.columns:
                food_df[col] = pd.to_numeric(food_df[col], errors='coerce')
        
        # Create meal planner with the user profile
        meal_planner = MealPlanRecommender(food_df, user_profile)
        return meal_planner.get_meal_plan(num_days=7)
    except Exception as e:
        print(f"Error generating meal plan: {str(e)}")
        raise

def calculate_bmi(height, weight):
    """Calculate BMI from height (cm) and weight (kg)"""
    height_m = height / 100
    return weight / (height_m * height_m)

def get_activity_level(caloric_balance):
    """Determine activity level based on caloric balance"""
    if caloric_balance > 500:
        return 'High'
    elif caloric_balance > 0:
        return 'Moderate'
    else:
        return 'Low'

def print_meal_plan(meal_plans):
    """Print the meal plan in a readable format"""
    if isinstance(meal_plans, str):
        meal_plans = json.loads(meal_plans)
        
    for day in meal_plans["meal_plan"]:
        print(f"\n=== Day {day['day']} ===")
        
        for meal_type, foods in day["meals"].items():
            print(f"\n{meal_type.title()}:")
            for food in foods:
                print(f"\nFood: {food['food_name']}")
                print(f"Calories: {food['nutritional_info']['calories']}")
                print(f"Protein: {food['nutritional_info']['protein']}g")
                print(f"Carbs: {food['nutritional_info']['carbs']}g")
                print(f"Fat: {food['nutritional_info']['fat']}g")
                print(f"Fiber: {food['nutritional_info']['fiber']}g")
                print(f"Sugars: {food['nutritional_info']['sugars']}g")
                print(f"Sodium: {food['nutritional_info']['sodium']}mg")
                print("Health Benefits:")
                for benefit in food['health_benefits']:
                    print(f"- {benefit}")

# Remove the direct instantiation of MealPlanRecommender
# The class should be imported and used in other files instead







