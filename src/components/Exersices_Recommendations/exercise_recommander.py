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

# Define preprocessing columns
num = ["age", "bmi", "energy_levels", "physical_activity", "available_time", "diabetesRisk"]
cat = ["gender", "pain_or_discomfort", "cardiovascular_health", "muscle_strength", "flexibility", "balance"]

# Create preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(), cat)
])

# ========== 1. Generate Training Data ==========
def generate_training_data():
    df = pd.DataFrame({
        "age": np.random.randint(20, 70, 300),
        "bmi": np.random.uniform(18, 40, 300),
        "energy_levels": np.random.uniform(1, 5, 300),
        "physical_activity": np.random.uniform(0.5, 5.0, 300),
        "available_time": np.random.randint(10, 60, 300),
        "diabetesRisk": np.random.uniform(0, 1, 300),
        "gender": np.random.randint(0, 2, 300),
        "pain_or_discomfort": np.random.randint(0, 2, 300),
        "cardiovascular_health": np.random.randint(0, 2, 300),
        "muscle_strength": np.random.randint(0, 2, 300),
        "flexibility": np.random.randint(0, 2, 300),
        "balance": np.random.randint(0, 2, 300)
    })

    def label(row):
        if row["diabetesRisk"] > 0.7 and row["physical_activity"] < 2:
            return "class_1"
        elif row["physical_activity"] < 2.5:
            return "class_2"
        else:
            return "class_3"

    df["label"] = df.apply(label, axis=1)
    return df

# ========== 2. Train and Save Models ==========
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

# ========== 3. Load Exercise Data ==========
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

# ========== 4. Routine Structure ==========
routine_structure = {
    "class_1": {
        "Warm-Up": 1,
        "Stretching": 3,
        "Cool Down": 1
    },
    "class_2": {
        "Warm-Up": 1,
        "Stretching": 2,
        "Strength": 2,
        "Cool Down": 1
    },
    "class_3": {
        "Warm-Up": 1,
        "Cardio/Strength": 3,
        "Flexibility": 1,
        "Cool Down": 1
    }
}

def get_risk_sentences(user_input):
    risk_sentences = []
    if user_input['diabetesRisk'] > 0.7:
        risk_sentences.append("High diabetes risk, avoid strenuous exercises.")
    if user_input['physical_activity'] < 2:
        risk_sentences.append("Low physical activity, avoid high intensity.")
    if user_input['pain_or_discomfort'] == 1:
        risk_sentences.append("User experiences pain or discomfort, avoid risky movements.")
    if user_input['cardiovascular_health'] == 1:
        risk_sentences.append("Cardiovascular health concerns, avoid intense cardio.")
    return risk_sentences

def recommend(user_input):
    try:
        # Load models
        model = joblib.load("artifact/physical_recommandations/exercise_recommendation_model.pkl")
        iso = joblib.load("artifact/physical_recommandations/novelty_detector.pkl")
        
        # Load exercise data
        ex_df, sbert_model = load_exercise_data()
        
        # Prepare user input
        user_df = pd.DataFrame([user_input])
        
        # Use the preprocessor from the pipeline instead of the global one
        # The preprocessor is already fitted as part of the pipeline
        pred_class = model.predict(user_df)[0]
        
        # For novelty detection, use the preprocessor from the pipeline
        pipeline_preprocessor = model.named_steps['prep']
        is_novel = iso.predict(pipeline_preprocessor.transform(user_df))[0] == -1
        
        recommendations = {
            "class": pred_class,
            "is_novel": bool(is_novel),
            "routine": []
        }
        
        # Filter exercises
        filtered = ex_df[ex_df["level"] == "beginner"]
        
        # Compute risk embeddings
        risk_sents = get_risk_sentences(user_input)
        if risk_sents:
            user_risk_embedding = sbert_model.encode(risk_sents, convert_to_numpy=True).mean(axis=0)
            
            # Filter risky exercises
            if len(filtered) > 0:
                exercise_embeddings_matrix = np.vstack(filtered['embedding'].values)
                similarities = cosine_similarity([user_risk_embedding], exercise_embeddings_matrix)[0]
                filtered = filtered[similarities < 0.5]
        
        # Generate routine
        routine_plan = routine_structure[pred_class]
        
        for role, count in routine_plan.items():
            block = filtered[filtered["routine_role"] == role]
            if len(block) == 0:
                continue
                
            samples = block.sample(n=min(count, len(block)))
            role_exercises = []
            
            for _, row in samples.iterrows():
                exercise = {
                    "name": row['name'],
                    "level": row['level'],
                    "equipment": row['equipment'],
                    "force": row.get('force', 'N/A'),
                    "primaryMuscles": row.get('primaryMuscles', []),
                    "secondaryMuscles": row.get('secondaryMuscles', []),
                    "instructions": row["instructions"],
                    "reps_sets": "2x10" if pred_class == "class_1" else "3x12" if pred_class == "class_2" else "4x15"
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
