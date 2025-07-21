import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. Columns/Preprocessing
# =========================

NUMERIC_COLS = [
    "Age", "Height", "Weight", "BMI", "EnergyLevels",
    "Physical_Activity", "Sitting_Time", "PhysicalActivityRisk",
    "Available_Time", "DiabetesRisk"
]

CATEGORICAL_COLS = [
    "Gender", "Pain_or_Discomfort", "Cardiovascular_Health",
    "Muscle_Strength", "Flexibility", "Balance", "goal"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_COLS),
    ("cat", OneHotEncoder(), CATEGORICAL_COLS)
])

# =========================
# 2. Natural Language Processing (Prompt)
# =========================

def process_user_prompt(prompt):
    # Expandable: Map pain/conditions to risk factors and target areas
    condition_mapping = {
        "pain": {
            "body_parts": {
                "elbow": {
                    "keywords": ["elbow", "arm", "forearm"],
                    "avoid_exercises": [
                        "push-ups", "bench press", "dips", "tricep extensions",
                        "barbell curl", "dumbbell curl", "bear crawl",
                        "plank", "push-up", "diamond push-up", "close grip",
                        "overhead press", "shoulder press", "lateral raise",
                        "front raise", "upright row", "pull-up", "chin-up"
                    ],
                    "avoid_muscles": ["biceps", "triceps", "forearms", "shoulders"],
                    "focus_areas": ["legs", "core", "back"]
                },
                "knee": {
                    "keywords": ["knee", "leg", "thigh"],
                    "avoid_exercises": [
                        "squats", "lunges", "jumping", "running",
                        "leg press", "leg extension", "leg curl",
                        "box jumps", "burpees", "mountain climbers"
                    ],
                    "avoid_muscles": ["quadriceps", "hamstrings", "calves"],
                    "focus_areas": ["upper body", "core"]
                },
                "back": {
                    "keywords": ["back", "spine", "lower back", "upper back"],
                    "avoid_exercises": [
                        "deadlifts", "bent over rows", "back extensions",
                        "good mornings", "hyperextensions", "squats",
                        "overhead press", "military press"
                    ],
                    "avoid_muscles": ["lower back", "middle back", "traps"],
                    "focus_areas": ["legs", "core", "arms"]
                },
                "shoulder": {
                    "keywords": ["shoulder", "rotator cuff"],
                    "avoid_exercises": [
                        "overhead press", "lateral raises", "pull-ups",
                        "push-ups", "bench press", "dips", "shoulder press",
                        "upright rows", "front raises"
                    ],
                    "avoid_muscles": ["shoulders", "deltoids", "rotator cuff"],
                    "focus_areas": ["legs", "core", "back"]
                },
                "neck": {
                    "keywords": ["neck", "cervical"],
                    "avoid_exercises": [
                        "neck bridges", "neck extensions", "head rotations",
                        "shoulder shrugs", "upright rows", "overhead press"
                    ],
                    "avoid_muscles": ["neck", "traps", "upper back"],
                    "focus_areas": ["legs", "core", "arms"]
                },
                "wrist": {
                    "keywords": ["wrist", "hand", "carpal"],
                    "avoid_exercises": [
                        "wrist curls", "push-ups", "planks",
                        "bench press", "dumbbell press", "pull-ups",
                        "dips", "handstand push-ups"
                    ],
                    "avoid_muscles": ["forearms", "wrists", "hands"],
                    "focus_areas": ["legs", "core", "back"]
                }
            }
        },
        "health_conditions": {
            "diabetes": ["diabetes", "blood sugar", "diabetic"],
            "heart": ["heart", "cardiac", "chest pain", "cardiovascular"],
            "blood_pressure": ["blood pressure", "hypertension", "high bp"],
            "arthritis": ["arthritis", "joint pain", "rheumatoid"]
        }
    }
    prompt = prompt.lower()
    extracted_info = {
        "pain_or_discomfort": 0,
        "cardiovascular_health": 0,
        "specific_conditions": [],
        "avoid_exercises": [],
        "avoid_muscles": [],
        "focus_areas": []
    }
    # Pain check
    for body_part, info in condition_mapping["pain"]["body_parts"].items():
        if any(keyword in prompt for keyword in info["keywords"]):
            extracted_info["pain_or_discomfort"] = 1
            extracted_info["specific_conditions"].append(f"{body_part}_pain")
            extracted_info["avoid_exercises"].extend(info["avoid_exercises"])
            extracted_info["avoid_muscles"].extend(info["avoid_muscles"])
            extracted_info["focus_areas"].extend(info["focus_areas"])
    # Health check
    for condition, keywords in condition_mapping["health_conditions"].items():
        if any(keyword in prompt for keyword in keywords):
            if condition == "heart":
                extracted_info["cardiovascular_health"] = 1
            extracted_info["specific_conditions"].append(condition)
    return extracted_info

# =========================
# 3. Class/Goal Label Assignment
# =========================

def assign_goal_label(row):
    dr = row.get("DiabetesRisk", 0)
    bmi = row.get("BMI", 0)
    pa = row.get("Physical_Activity", 0)
    ms = row.get("Muscle_Strength", 0)
    st = row.get("Sitting_Time", 0)
    age = row.get("Age", 0)
    cv = row.get("Cardiovascular_Health", 0)
    pain = row.get("Pain_or_Discomfort", 0)
    flex = row.get("Flexibility", 2)
    # Diabetes-aware
    if dr > 80 and pa < 2 and bmi > 30:
        return pd.Series(["diabetes_urgent_management", "class_1"])
    elif dr > 70 and bmi > 30:
        return pd.Series(["diabetes_management_obesity", "class_1"])
    elif dr > 70 and cv == 1:
        return pd.Series(["diabetes_management_cardiac", "class_1"])
    elif dr > 60 and pain == 1:
        return pd.Series(["diabetes_management_pain", "class_1"])
    elif dr > 60 and pa < 2.5:
        return pd.Series(["diabetes_management_low_activity", "class_1"])
    elif dr > 60:
        return pd.Series(["diabetes_management", "class_1"])
    elif dr > 40 and age < 30:
        return pd.Series(["pre_diabetes_young", "class_1"])
    elif dr > 40 and bmi > 25:
        return pd.Series(["pre_diabetes_prevention", "class_3"])
    elif dr > 40:
        return pd.Series(["pre_diabetes_prevention", "class_1"])
    elif dr > 20 and age > 60:
        return pd.Series(["elderly_diabetes_management", "class_1"])
    elif dr > 20 and pa > 3:
        return pd.Series(["diabetes_maintenance_fitness", "class_2"])
    # Others
    if bmi > 35:
        return pd.Series(["obesity_weight_loss", "class_3"])
    elif bmi > 30:
        return pd.Series(["overweight_weight_loss", "class_3"])
    elif pain == 1 and flex <= 1:
        return pd.Series(["flexibility_pain", "class_5"])
    elif st > 10:
        return pd.Series(["sedentary_general", "class_2"])
    elif row.get("goal", "") == "muscle_gain" and dr > 20:
        return pd.Series(["muscle_gain_diabetes", "class_4"])
    elif row.get("goal", "") == "muscle_gain":
        return pd.Series(["muscle_gain", "class_4"])
    elif row.get("goal", "") == "flexibility":
        return pd.Series(["flexibility", "class_5"])
    elif row.get("goal", "") == "balance_improvement":
        return pd.Series(["balance_improvement", "class_5"])
    elif age > 50 and pa < 2:
        return pd.Series(["posture_balance", "class_5"])
    else:
        return pd.Series(["general_fitness", "class_2"])

# =========================
# 4. Model Training
# =========================

def load_real_data(filepath):
    df = pd.read_csv(filepath)
    return df

def train_and_save_models():
    df = load_real_data('G:\\FYP_Diabetic_Prediction_Recomandations\\notebook\\data\\RecommandationDatasets\\GymDatasets\\PhysicalActivity_Goal.csv')
    if 'label' not in df.columns:
        raise ValueError("The 'label' column is missing from the dataset.")
    X = df.drop("label", axis=1)
    y = df["label"]
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(), CATEGORICAL_COLS)
        ])),
        ("clf", RandomForestClassifier(n_estimators=100))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    os.makedirs("artifact/physical_recommandations", exist_ok=True)
    joblib.dump(model, "artifact/physical_recommandations/exercise_recommendation_model.pkl")
    pipeline_preprocessor = model.named_steps['prep']
    iso_model = IsolationForest(contamination=0.1).fit(pipeline_preprocessor.transform(X))
    joblib.dump(iso_model, "artifact/physical_recommandations/novelty_detector.pkl")

# =========================
# 5. Load Exercise Data
# =========================

def load_exercise_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "exercises.json")
    with open(json_path, "r") as f:
        exercises_data = json.load(f)
    ex_df = pd.json_normalize(exercises_data)
    def tag_role(row):
        cat = row["category"]
        if cat == "stretching":
            return "Stretching"
        elif cat == "strength":
            return "Strength"
        elif cat in ["plyometrics", "powerlifting"]:
            return "Cardio/Strength"
        else:
            return "Warm-Up"
    ex_df["routine_role"] = ex_df.apply(tag_role, axis=1)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    ex_df['instructions_text'] = ex_df['instructions'].apply(
        lambda steps: " ".join(steps) if isinstance(steps, list) else ""
    )
    exercise_embeddings = sbert_model.encode(
        ex_df['instructions_text'].tolist(), convert_to_numpy=True
    )
    ex_df['embedding'] = list(exercise_embeddings)
    return ex_df, sbert_model

# =========================
# 6. Routine Structure & Home Exercises
# =========================

# Update routine_structure to include all possible classes
routine_structure = {
    "class_1": {  # Diabetes management
        "Warm-Up": 1,
        "Stretching": 2,
        "Strength": 2,
        "Cardio": 1
    },
    "class_2": {  # Sedentary reduction
        "Warm-Up": 1,
        "Stretching": 2,
        "Strength": 2,
        "Cardio": 1
    },
    "class_3": {  # Weight loss
        "Warm-Up": 1,
        "Stretching": 2,
        "Strength": 2,
        "Cardio": 2
    },
    "class_4": {  # Muscle gain
        "Warm-Up": 1,
        "Stretching": 2,
        "Strength": 3,
        "Cardio": 1
    },
    "class_5": {  # Flexibility
        "Warm-Up": 1,
        "Stretching": 3,
        "Strength": 1,
        "Cardio": 1
    },
    "class_flexibility": {  # Add new class for flexibility
        "Warm-Up": 1,
        "Stretching": 3,
        "Strength": 1,
        "Cardio": 1
    },
    "class_flexibility_pain": {  # Add class for flexibility with pain
        "Warm-Up": 1,
        "Stretching": 3,
        "Strength": 1,
        "Cardio": 1
    }
}

home_exercises = {
    "Warm-Up": [
        "Walking in Place", "Arm Circles", "Marching in Place", "Jumping Jacks (Modified)",
        "High Knees (Modified)", "Butt Kicks", "Side Steps", "Hip Circles"
    ],
    "Strength": [
        "Bodyweight Squats", "Wall Push-Ups", "Chair Dips", "Plank",
        "Glute Bridges", "Bird Dogs", "Superman", "Mountain Climbers",
        "Lunges", "Step-Ups"
    ],
    "Cardio": [
        "Walking in Place", "Marching in Place", "Step-Ups",
        "Modified Jumping Jacks", "Modified High Knees",
        "Modified Mountain Climbers", "Arm Circles", "Side Steps"
    ]
}

# =========================
# 7. Risk Sentences
# =========================

def get_risk_sentences(user_input):
    risk_sentences = []
    if user_input.get('diabetesRisk', 0) > 0.7:
        risk_sentences.append("High diabetes risk, avoid strenuous exercises.")
    if user_input.get('physical_activity', 0) < 2:
        risk_sentences.append("Low physical activity, avoid high intensity.")
    if user_input.get('sitting_time', 0) > 8:
        risk_sentences.append("High sedentary time, focus on movement and posture.")
    if user_input.get('physical_activity_risk', 0) > 0.7:
        risk_sentences.append("High physical activity risk, start with low intensity.")
    if user_input.get('pain_or_discomfort', 0) == 1:
        risk_sentences.append("User experiences pain or discomfort, avoid risky movements.")
    if user_input.get('cardiovascular_health', 0) == 1:
        risk_sentences.append("Cardiovascular health concerns, avoid intense cardio.")
    return risk_sentences

# =========================
# 8. Main Recommend Function
# =========================
def recommend(user_input, user_prompt=None):
    try:
        # Process user prompt if provided
        prompt_info = process_user_prompt(user_prompt) if user_prompt else None
        
        # Update user input with prompt information if available
        if prompt_info:
            user_input.update({
                "Pain_or_Discomfort": prompt_info["pain_or_discomfort"],
                "Cardiovascular_Health": prompt_info["cardiovascular_health"]
            })
        
        # Create DataFrame with single row
        user_df = pd.DataFrame([user_input])
        
        # Ensure all required columns are present
        required_columns = set([
            "Age", "Gender", "Height", "Weight", "BMI", 
            "EnergyLevels", "Physical_Activity", "Sitting_Time",
            "PhysicalActivityRisk", "Available_Time", "DiabetesRisk",
            "Pain_or_Discomfort", "Cardiovascular_Health",
            "Muscle_Strength", "Flexibility", "Balance"
        ])
        
        missing_columns = required_columns - set(user_df.columns)
        if missing_columns:
            raise ValueError(f"columns are missing: {missing_columns}")

        # Load models and continue with recommendation logic
        model = joblib.load("artifact/physical_recommandations/exercise_recommendation_model.pkl")
        iso = joblib.load("artifact/physical_recommandations/novelty_detector.pkl")
        ex_df, sbert_model = load_exercise_data()
        
        pred_class = model.predict(user_df)[0]
        
        # Handle class mapping
        if pred_class not in routine_structure:
            # Map unknown classes to closest matching routine
            if "flexibility" in pred_class.lower():
                pred_class = "class_5"  # Map to flexibility class
            elif "pain" in pred_class.lower():
                pred_class = "class_5"  # Map to flexibility class for pain cases
            else:
                pred_class = "class_2"  # Default to general fitness class
        
        pipeline_preprocessor = model.named_steps['prep']
        is_novel = iso.predict(pipeline_preprocessor.transform(user_df))[0] == -1
        recommendations = {
            "class": pred_class,
            "is_novel": bool(is_novel),
            "routine": [],
            "goal": user_input.get('goal', 'general_fitness'),
            "bmi": user_input['BMI'],
            "sitting_time": user_input.get('Sitting_Time', 0),
            "physical_activity_risk": user_input.get('PhysicalActivityRisk', 0)
        }
        filtered = ex_df[ex_df["level"] == "beginner"]

        # Filtering based on prompt info
        if prompt_info:
            # Filter out exercises to avoid
            if prompt_info["avoid_exercises"]:
                filtered = filtered[~filtered['name'].str.lower().isin(
                    [ex.lower() for ex in prompt_info["avoid_exercises"]]
                )]
            # Filter out exercises that target muscles to avoid
            if prompt_info["avoid_muscles"]:
                filtered = filtered[~filtered['primaryMuscles'].apply(
                    lambda x: any(muscle.lower() in [m.lower() for m in x] 
                                for muscle in prompt_info["avoid_muscles"])
                )]
            # Filter out exercises that use secondary muscles to avoid
            if prompt_info["avoid_muscles"]:
                filtered = filtered[~filtered['secondaryMuscles'].apply(
                    lambda x: any(muscle.lower() in [m.lower() for m in x] 
                                for muscle in prompt_info["avoid_muscles"])
                )]
            # If focus areas are specified, prioritize those exercises
            if prompt_info["focus_areas"]:
                def get_focus_score(row):
                    score = 0
                    for area in prompt_info["focus_areas"]:
                        if area.lower() in row['name'].lower():
                            score += 2
                        if any(area.lower() in muscle.lower() 
                               for muscle in row['primaryMuscles']):
                            score += 1
                    return score
                filtered['focus_score'] = filtered.apply(get_focus_score, axis=1)
                filtered = filtered.sort_values('focus_score', ascending=False)

        # Apply risk-based filtering
        risk_sents = get_risk_sentences(user_input)
        if risk_sents:
            user_risk_embedding = sbert_model.encode(risk_sents, convert_to_numpy=True).mean(axis=0)
            if len(filtered) > 0:
                exercise_embeddings_matrix = np.vstack(filtered['embedding'].values)
                similarities = cosine_similarity([user_risk_embedding], exercise_embeddings_matrix)[0]
                filtered = filtered[similarities < 0.5]

        # Generate routine based on class and goal
        routine_plan = routine_structure[pred_class]
        for role, count in routine_plan.items():
            # For home exercises, use predefined list if prompt mentions home
            use_home = user_prompt and "home" in user_prompt.lower() and role in home_exercises
            if use_home:
                home_exs = filtered[filtered['name'].isin(home_exercises[role])]
                block = home_exs if len(home_exs) > 0 else filtered[filtered["routine_role"] == role]
            else:
                block = filtered[filtered["routine_role"] == role]
            if len(block) == 0:
                continue
            samples = block.sample(n=min(count, len(block)))
            role_exercises = []
            for _, row in samples.iterrows():
                # Customize reps and sets based on class
                if pred_class == "class_1":
                    reps_sets = "2x10"
                elif pred_class == "class_2":
                    reps_sets = "3x12"
                elif pred_class == "class_3":
                    reps_sets = "4x15"
                elif pred_class == "class_4":
                    reps_sets = "4x8"
                elif pred_class == "class_5":
                    reps_sets = "30s hold"
                else:
                    reps_sets = "3x12"
                exercise = {
                    "name": row['name'],
                    "level": row['level'],
                    "equipment": row['equipment'],
                    "force": row.get('force', 'N/A'),
                    "primaryMuscles": row.get('primaryMuscles', []),
                    "secondaryMuscles": row.get('secondaryMuscles', []),
                    "instructions": row["instructions"],
                    "reps_sets": reps_sets
                }
                role_exercises.append(exercise)
            recommendations["routine"].append({
                "role": role,
                "exercises": role_exercises
            })
        return recommendations
    except Exception as e:
        print(f"Error in recommend function: {str(e)}")
        raise e

# =========================
# 9. Train If Not Exists
# =========================

if not os.path.exists("artifact/physical_recommandations/exercise_recommendation_model.pkl"):
    train_and_save_models()

# END OF FILE
