import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import re

# Define preprocessing columns
num = [
    "age", "height", "weight", "bmi", "energy_levels", 
    "physical_activity", "sitting_time", "physical_activity_risk",
    "available_time", "diabetesRisk"
]
cat = [
    "gender", "pain_or_discomfort", "cardiovascular_health", 
    "muscle_strength", "flexibility", "balance", "goal"
]

# Create preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(), cat)
])

def process_user_prompt(prompt):
    """
    Process natural language input from user and extract relevant information
    """
    # Initialize SBERT model for semantic similarity
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define common conditions and their corresponding risk factors
    condition_mapping = {
        "pain": {
            "keywords": ["pain", "hurt", "ache", "sore", "injury"],
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
    
    # Process the prompt
    prompt = prompt.lower()
    extracted_info = {
        "pain_or_discomfort": 0,
        "cardiovascular_health": 0,
        "specific_conditions": [],
        "avoid_exercises": [],
        "avoid_muscles": [],
        "focus_areas": []
    }
    
    # Check for pain conditions
    for body_part, info in condition_mapping["pain"]["body_parts"].items():
        if any(keyword in prompt for keyword in info["keywords"]):
            extracted_info["pain_or_discomfort"] = 1
            extracted_info["specific_conditions"].append(f"{body_part}_pain")
            extracted_info["avoid_exercises"].extend(info["avoid_exercises"])
            extracted_info["avoid_muscles"].extend(info["avoid_muscles"])
            extracted_info["focus_areas"].extend(info["focus_areas"])
    
    # Check for health conditions
    for condition, keywords in condition_mapping["health_conditions"].items():
        if any(keyword in prompt for keyword in keywords):
            if condition == "heart":
                extracted_info["cardiovascular_health"] = 1
            extracted_info["specific_conditions"].append(condition)
    
    return extracted_info

def generate_training_data():
    df = pd.DataFrame({
        "age": np.random.randint(20, 70, 300),
        "height": np.random.uniform(150, 190, 300),  # height in cm
        "weight": np.random.uniform(45, 120, 300),   # weight in kg
        "bmi": np.random.uniform(18, 40, 300),
        "energy_levels": np.random.uniform(1, 5, 300),
        "physical_activity": np.random.uniform(0.5, 5.0, 300),
        "sitting_time": np.random.uniform(0, 2, 300),  # hours per day
        "physical_activity_risk": np.random.uniform(0, 1, 300),
        "available_time": np.random.randint(10, 60, 300),
        "diabetesRisk": np.random.uniform(0, 1, 300),
        "gender": np.random.randint(0, 2, 300),
        "pain_or_discomfort": np.random.randint(0, 2, 300),
        "cardiovascular_health": np.random.randint(0, 2, 300),
        "muscle_strength": np.random.randint(0, 2, 300),
        "flexibility": np.random.randint(0, 2, 300),
        "balance": np.random.randint(0, 2, 300),
        "goal": np.random.choice([
            "weight_loss", "muscle_gain", "flexibility", 
            "endurance", "general_fitness", "diabetes_management"
        ], 300)
    })

    def label(row):
        # Enhanced labeling based on more factors
        if row["diabetesRisk"] > 0.7 and row["physical_activity"] < 2:
            return "class_1"  # Low intensity, focus on diabetes management
        elif row["physical_activity"] < 2.5 or row["sitting_time"] > 8:
            return "class_2"  # Moderate intensity, focus on reducing sedentary behavior
        elif row["goal"] == "weight_loss" and row["bmi"] > 25:
            return "class_3"  # Higher intensity, focus on weight loss
        elif row["goal"] == "muscle_gain":
            return "class_4"  # Strength training focus
        elif row["goal"] == "flexibility":
            return "class_5"  # Flexibility and mobility focus
        else:
            return "class_6"  # General fitness

    df["label"] = df.apply(label, axis=1)
    return df

def train_and_save_models():
    df = generate_training_data()
    X = df.drop("label", axis=1)
    y = df["label"]

    # Create the pipeline with preprocessor
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(), cat)
        ])),
        ("clf", RandomForestClassifier(n_estimators=100))
    ])

    # Train the pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    
    # Create directory if it doesn't exist
    os.makedirs("artifact/physical_recommandations", exist_ok=True)
    
    # Save the entire pipeline
    joblib.dump(model, "artifact/physical_recommandations/exercise_recommendation_model.pkl")
    
    # For novelty detection, use the same preprocessor
    pipeline_preprocessor = model.named_steps['prep']
    iso_model = IsolationForest(contamination=0.1).fit(pipeline_preprocessor.transform(X))
    joblib.dump(iso_model, "artifact/physical_recommandations/novelty_detector.pkl")

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
    
    # Initialize SBERT model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare instructions text for SBERT embedding
    ex_df['instructions_text'] = ex_df['instructions'].apply(
        lambda steps: " ".join(steps) if isinstance(steps, list) else ""
    )
    
    # Compute embeddings
    exercise_embeddings = sbert_model.encode(
        ex_df['instructions_text'].tolist(), 
        convert_to_numpy=True
    )
    ex_df['embedding'] = list(exercise_embeddings)
    
    return ex_df, sbert_model

# Define routine structure based on class and goal
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
    }
}

# Define home-friendly exercises for each category
home_exercises = {
    "Warm-Up": [
        "Walking in Place",
        "Arm Circles",
        "Marching in Place",
        "Jumping Jacks (Modified)",
        "High Knees (Modified)",
        "Butt Kicks",
        "Side Steps",
        "Hip Circles"
    ],
    "Strength": [
        "Bodyweight Squats",
        "Wall Push-Ups",
        "Chair Dips",
        "Plank",
        "Glute Bridges",
        "Bird Dogs",
        "Superman",
        "Mountain Climbers",
        "Lunges",
        "Step-Ups"
    ],
    "Cardio": [
        "Walking in Place",
        "Marching in Place",
        "Step-Ups",
        "Modified Jumping Jacks",
        "Modified High Knees",
        "Modified Mountain Climbers",
        "Arm Circles",
        "Side Steps"
    ]
}

def get_risk_sentences(user_input):
    risk_sentences = []
    if user_input['diabetesRisk'] > 0.7:
        risk_sentences.append("High diabetes risk, avoid strenuous exercises.")
    if user_input['physical_activity'] < 2:
        risk_sentences.append("Low physical activity, avoid high intensity.")
    if user_input['sitting_time'] > 8:
        risk_sentences.append("High sedentary time, focus on movement and posture.")
    if user_input['physical_activity_risk'] > 0.7:
        risk_sentences.append("High physical activity risk, start with low intensity.")
    if user_input['pain_or_discomfort'] == 1:
        risk_sentences.append("User experiences pain or discomfort, avoid risky movements.")
    if user_input['cardiovascular_health'] == 1:
        risk_sentences.append("Cardiovascular health concerns, avoid intense cardio.")
    return risk_sentences

def recommend(user_input, user_prompt=None):
    try:
        # Process user prompt if provided
        prompt_info = None
        if user_prompt:
            prompt_info = process_user_prompt(user_prompt)
            # Update user_input with information from prompt
            user_input.update({
                "pain_or_discomfort": prompt_info["pain_or_discomfort"],
                "cardiovascular_health": prompt_info["cardiovascular_health"]
            })
        
        # Calculate BMI if not provided
        if 'bmi' not in user_input and 'height' in user_input and 'weight' in user_input:
            user_input['bmi'] = user_input['weight'] / ((user_input['height'] / 100) ** 2)

        # Load models
        model = joblib.load("artifact/physical_recommandations/exercise_recommendation_model.pkl")
        iso = joblib.load("artifact/physical_recommandations/novelty_detector.pkl")
        
        # Load exercise data
        ex_df, sbert_model = load_exercise_data()
        
        # Prepare user input
        user_df = pd.DataFrame([user_input])
        
        # Get prediction and novelty detection
        pred_class = model.predict(user_df)[0]
        pipeline_preprocessor = model.named_steps['prep']
        is_novel = iso.predict(pipeline_preprocessor.transform(user_df))[0] == -1
        
        recommendations = {
            "class": pred_class,
            "is_novel": bool(is_novel),
            "routine": [],
            "goal": user_input.get('goal', 'general_fitness'),
            "bmi": user_input['bmi'],
            "sitting_time": user_input.get('sitting_time', 0),
            "physical_activity_risk": user_input.get('physical_activity_risk', 0)
        }
        
        # Filter exercises based on user's level and goals
        filtered = ex_df[ex_df["level"] == "beginner"]
        
        # If we have prompt information, apply comprehensive filtering
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
                # Create a score for each exercise based on focus areas
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
            # For home exercises, use the predefined list
            if role in home_exercises and "home" in user_prompt.lower():
                home_exs = filtered[filtered['name'].isin(home_exercises[role])]
                if len(home_exs) > 0:
                    block = home_exs
                else:
                    block = filtered[filtered["routine_role"] == role]
            else:
                block = filtered[filtered["routine_role"] == role]
            
            if len(block) == 0:
                continue
                
            samples = block.sample(n=min(count, len(block)))
            role_exercises = []
            
            for _, row in samples.iterrows():
                # Customize reps and sets based on goal
                if pred_class == "class_1":  # Diabetes management
                    reps_sets = "2x10"
                elif pred_class == "class_2":  # Sedentary reduction
                    reps_sets = "3x12"
                elif pred_class == "class_3":  # Weight loss
                    reps_sets = "4x15"
                elif pred_class == "class_4":  # Muscle gain
                    reps_sets = "4x8"
                elif pred_class == "class_5":  # Flexibility
                    reps_sets = "30s hold"
                else:  # General fitness
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

# Train models if they don't exist
if not os.path.exists("artifact/physical_recommandations/exercise_recommendation_model.pkl"):
    train_and_save_models()
