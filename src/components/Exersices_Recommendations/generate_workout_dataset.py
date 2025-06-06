import json
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

print(os.getcwd())
# Load your exercises.json file
with open("notebook/data/RecommandationDatasets/GymDatasets/exercises.json", "r") as f:
    exercises_data = f.read()
    # If your JSON file is not a list, fix parsing here
    # Remove trailing commas and wrap in [] if necessary
    if exercises_data.strip().startswith("{"):
        exercises_data = f"[{exercises_data}]"
    exercises = json.loads(exercises_data)
    

# Filter for beginner, bodyweight exercises
beginner_exercises = [
    ex
    for ex in exercises
    if (str(ex.get("level", "") or "").lower() == "beginner"
        and str(ex.get("equipment", "") or "").lower() == "body only")
]


# Helper to generate random user data
def get_reps(energy, activity):
    if energy <= 2 or activity <= 2:
        return random.randint(8, 12)
    elif energy == 3 or activity == 3:
        return random.randint(12, 20)
    else:
        return random.randint(10, 15)

def random_profile():
    age = random.randint(18, 40)
    gender = random.choice(['Male', 'Female'])
    height = random.randint(150, 190)
    weight = random.randint(45, 90)
    energy = random.randint(1, 3)
    activity = random.randint(1, 3)
    sitting = random.randint(1, 3)
    cardio_health = random.choice(['Yes', 'No'])
    muscle_strength = random.choice(['Yes', 'No'])
    flexibility = random.choice(['Yes', 'No'])
    balance = random.choice(['Yes', 'No'])
    thirsty = random.choice(['Yes', 'No'])
    pain = random.randint(1, 3)
    available_time = random.randint(90, 300)
    diabetes_risk = random.randint(10, 90)
    nutrition_risk = random.randint(10, 90)
    return {
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Weight": weight,
        "Energy Levels": energy,
        "Physical Activity": activity,
        "Sitting Time": sitting,
        "Cardiovascular Health": cardio_health,
        "Muscle Strength": muscle_strength,
        "Flexibility": flexibility,
        "Balance": balance,
        "Thirsty": thirsty,
        "Pain or Discomfort": pain,
        "Available Time": available_time,
        "Diabetes Risk": diabetes_risk,
        "Nutrition Risk": nutrition_risk
    }

def make_prompt(profile):
    return (
        f"Age: {profile['Age']}, Gender: {profile['Gender']}, Height: {profile['Height']} cm, "
        f"Weight: {profile['Weight']} kg, Energy Levels: {profile['Energy Levels']}, "
        f"Physical Activity: {profile['Physical Activity']}, Sitting Time: {profile['Sitting Time']}, "
        f"Cardiovascular Health: {profile['Cardiovascular Health']}, Muscle Strength: {profile['Muscle Strength']}, "
        f"Flexibility: {profile['Flexibility']}, Balance: {profile['Balance']}, Thirsty: {profile['Thirsty']}, "
        f"Pain or Discomfort: {profile['Pain or Discomfort']}, Available Time: {profile['Available Time']} minutes/week, "
        f"Diabetes Risk: {profile['Diabetes Risk']}, Nutrition Risk: {profile['Nutrition Risk']}. "
        "Recommend a personalized workout."
    )

def make_completion(profile, selected_exercises):
    energy = profile['Energy Levels']
    activity = profile['Physical Activity']
    exercises_info = []
    for ex in selected_exercises:
        reps = get_reps(energy, activity)
        exercises_info.append({
            "name": ex['name'],
            "reps": reps,
            "steps": ex['instructions']
        })
    exercise_lines = [
        f"- {ex['name']}: {ex['reps']} reps\n  Steps: " + " | ".join(ex['steps'])
        for ex in exercises_info
    ]
    return (
        "Based on your profile, start with these beginner bodyweight exercises:\n" +
        "\n".join(exercise_lines) +
        "\nInclude gentle stretching and short walks as tolerated. Gradually increase intensity as your energy and strength improve. Rest as needed."
    )

# Generate 1000 data entries
dataset = []
for _ in range(1000):
    profile = random_profile()
    selected_exercises = random.sample(beginner_exercises, k=min(3, len(beginner_exercises)))
    prompt = make_prompt(profile)
    completion = make_completion(profile, selected_exercises)
    dataset.append({
        "prompt": prompt,
        "completion": completion
    })

# Save to JSON file
with open('personalized_workout_dataset_with_reps_steps.json', 'w') as f:
    json.dump(dataset, f, indent=2)