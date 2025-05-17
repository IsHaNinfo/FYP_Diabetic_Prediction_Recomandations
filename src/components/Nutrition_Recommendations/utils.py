import pandas as pd

def recommend_foods(food_df, protein_target, fat_target, carbs_target, top_n=5):
    food_df_clean = food_df.copy()
    food_df_clean[['Protein', 'Fat', 'Carbs']] = food_df_clean[['Protein', 'Fat', 'Carbs']].apply(pd.to_numeric, errors='coerce').fillna(0)
    food_df_clean['score'] = (
        abs(food_df_clean['Protein'] - protein_target) +
        abs(food_df_clean['Fat'] - fat_target) +
        abs(food_df_clean['Carbs'] - carbs_target)
    )
    return food_df_clean.sort_values('score').head(top_n)

def generate_meal_plan(user_pred, food_df, meals_per_day=['breakfast', 'lunch', 'dinner', 'snack']):
    import random
    plan = {}
    daily_targets = {
        'breakfast': 0.25,
        'lunch': 0.35,
        'dinner': 0.30,
        'snack': 0.10
    }
    for day in range(1, 8):
        day_plan = {}
        for meal in meals_per_day:
            ratio = daily_targets[meal]
            protein_target = user_pred['Protein_Intake'] * ratio
            fat_target = user_pred['Fat_Intake'] * ratio
            carbs_target = user_pred['Carbohydrate_Consumption'] * ratio
            recommended_meals = recommend_foods(food_df, protein_target, fat_target, carbs_target, top_n=5)
            chosen = recommended_meals.sample(1).iloc[0]
            day_plan[meal] = {
                'Food': chosen['Food'],
                'Calories': chosen['Calories'],
                'Protein': chosen['Protein'],
                'Fat': chosen['Fat'],
                'Carbs': chosen['Carbs'],
                'Category': chosen.get('Category', 'Unknown')
            }
        plan[f'Day {day}'] = day_plan
    return plan