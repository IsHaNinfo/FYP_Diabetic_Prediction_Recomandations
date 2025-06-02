import pandas as pd
import random
def calculate_nutrition_targets(user_info):
    

    weight = user_info.get('Weight', 70)
    height_cm = user_info.get('Height', 170) 
    height = height_cm / 100
    calories = user_info.get('Caloric_Balance', 2000)
    diabetes_risk = user_info.get('DiabetesRisk', 0)
    nutrition_risk = user_info.get('NutritionRisk', 0)

    # Calculate BMI
    bmi = weight / (height ** 2)
    
    # Adjust calories based on BMI
    if bmi > 25:
        calories -= 300  # reduce for overweight
    elif bmi < 18.5:
        calories += 300  # increase for underweight

    # Adjust macro ratios based on risk
    carb_ratio = 0.5
    fat_ratio = 0.3
    protein_per_kg = 0.8

    if diabetes_risk > 50:
        carb_ratio = 0.4  # lower carbs
        fat_ratio = 0.35
        protein_per_kg = 1.0  # more protein

    if nutrition_risk >50:
        protein_per_kg = 1.2  # even more protein

    protein = protein_per_kg * weight
    fat = (fat_ratio * calories) / 9
    carbs = (carb_ratio * calories) / 4

    return {
        'Protein_Intake': protein,
        'Fat_Intake': fat,
        'Carbohydrate_Consumption': carbs,
        'Calories': calories,
        'BMI': bmi
    }
def recommend_foods(
    food_df, 
    protein_target, 
    fat_target, 
    carbs_target, 
    diabetes_risk=None, 
    nutrition_risk=None, 
    sugar_limit=None, 
    top_n=5
):
    food_df_clean = food_df.copy()
    food_df_clean[['Protein_(g)', 'Fat_(g)', 'Carbohydrate_(g)']] = food_df_clean[
        ['Protein_(g)', 'Fat_(g)', 'Carbohydrate_(g)']
    ].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # If sugar_limit is set (e.g., for high diabetes risk), penalize high sugar foods
    if sugar_limit is not None and 'Free_sugars_(g)' in food_df_clean.columns:
        food_df_clean['Free_sugars_(g)'] = pd.to_numeric(food_df_clean['Free_sugars_(g)'], errors='coerce').fillna(0)
        food_df_clean = food_df_clean[food_df_clean['Free_sugars_(g)'] <= sugar_limit]

    # Calculate score based on absolute difference from targets and risk factors
    food_df_clean['score'] = (
        abs(food_df_clean['Protein_(g)'] - protein_target) +
        abs(food_df_clean['Fat_(g)'] - fat_target) +
        abs(food_df_clean['Carbohydrate_(g)'] - carbs_target)
    )
    # Optionally, add more penalties for risk (customize as needed)
    if diabetes_risk is not None and 'Free_sugars_(g)' in food_df_clean.columns:
        food_df_clean['score'] += diabetes_risk * food_df_clean['Free_sugars_(g)']
    if nutrition_risk is not None:
        # Example: penalize high fat for high nutrition risk
        food_df_clean['score'] += nutrition_risk * food_df_clean['Fat_(g)']

    return food_df_clean.sort_values('score').head(top_n)

def build_meal_portion(food_df, protein_target, fat_target, carbs_target, diabetes_risk=None, nutrition_risk=None, sugar_limit=None, max_items=3):
    selected = []
    total_protein = 0
    total_fat = 0
    total_carbs = 0
    remaining_df = food_df.copy()
    for _ in range(max_items):
        # Calculate remaining targets
        p_rem = max(protein_target - total_protein, 0)
        f_rem = max(fat_target - total_fat, 0)
        c_rem = max(carbs_target - total_carbs, 0)
        # Recommend best food for remaining target
        rec = recommend_foods(remaining_df, p_rem, f_rem, c_rem, diabetes_risk, nutrition_risk, sugar_limit, top_n=1)
        if rec.empty:
            break
        food = rec.iloc[0]
        # Portion factor: don't exceed any macro target
        factors = []
        if food['Protein_(g)'] > 0: factors.append(p_rem / food['Protein_(g)'])
        if food['Fat_(g)'] > 0: factors.append(f_rem / food['Fat_(g)'])
        if food['Carbohydrate_(g)'] > 0: factors.append(c_rem / food['Carbohydrate_(g)'])
        portion = min(factors + [1.0])  # Don't oversize
        portion = max(portion, 0.1)     # Don't go below 10%
        # Add to meal
        selected.append((food, portion))
        # Update totals
        total_protein += food['Protein_(g)'] * portion
        total_fat += food['Fat_(g)'] * portion
        total_carbs += food['Carbohydrate_(g)'] * portion
        # Remove this food from further selection
        remaining_df = remaining_df[remaining_df['Food'] != food['Food']]
        # Stop if close enough to targets
        if (total_protein >= protein_target*0.95 and
            total_fat >= fat_target*0.95 and
            total_carbs >= carbs_target*0.95):
            break
    return selected

def generate_meal_plan(
    user_pred, 
    food_df, 
    meals_per_day=['breakfast', 'lunch', 'dinner', 'snack']
):
    # Calculate nutrition targets from user info
    nutrition_targets = calculate_nutrition_targets(user_pred)
    user_pred.update(nutrition_targets)

    plan = {}
    meal_category_map = {
        'breakfast': 'Breakfast',
        'lunch': 'Lunch',
        'dinner': 'Dinner',
        'snack': 'Snack'
    }
    daily_targets = {
        'breakfast': 0.25,
        'lunch': 0.35,
        'dinner': 0.30,
        'snack': 0.10
    }

    # Ensure no negative predicted values
    for key in ['Protein_Intake', 'Fat_Intake', 'Carbohydrate_Consumption']:
        user_pred[key] = max(user_pred[key], 0)

    # Extract risk factors if present
    diabetes_risk = user_pred.get('DiabetesRisk', None)
    nutrition_risk = user_pred.get('NutritionRisk', None)
    sugar_limit = 5 if diabetes_risk is not None and diabetes_risk > 0.5 else None

    for day in range(1, 8):
        day_plan = {}
        for meal in meals_per_day:
            ratio = daily_targets[meal]
            protein_target = max(user_pred['Protein_Intake'] * ratio, 0)
            fat_target = max(user_pred['Fat_Intake'] * ratio, 0)
            carbs_target = max(user_pred['Carbohydrate_Consumption'] * ratio, 0)

            category = meal_category_map[meal]
            food_df_filtered = food_df[food_df['Category'].str.lower() == category.lower()]
            if food_df_filtered.empty:
                food_df_filtered = food_df

            # Build a meal with multiple foods and proper portions
            meal_items = build_meal_portion(
                food_df_filtered, protein_target, fat_target, carbs_target,
                diabetes_risk=diabetes_risk, nutrition_risk=nutrition_risk, sugar_limit=sugar_limit, max_items=3
            )
            # Summarize the meal
            meal_summary = {
                'Foods': [],
                'Total_Protein_(g)': 0,
                'Total_Fat_(g)': 0,
                'Total_Carbohydrate_(g)': 0,
                'Total_Calories_(kcal)': 0,
            }
            for food, portion in meal_items:
                meal_summary['Foods'].append(f"{round(portion*food['Quantity'],1)}g {food['Food']}")
                meal_summary['Total_Protein_(g)'] += food['Protein_(g)'] * portion
                meal_summary['Total_Fat_(g)'] += food['Fat_(g)'] * portion
                meal_summary['Total_Carbohydrate_(g)'] += food['Carbohydrate_(g)'] * portion
                meal_summary['Total_Calories_(kcal)'] += food['Calories_(kcal)'] * portion

            day_plan[meal] = meal_summary
        plan[f'Day {day}'] = day_plan
    return plan