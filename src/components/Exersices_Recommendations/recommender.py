import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import load_user_data, load_gym_data
from .config import FEATURE_COLS, TARGET_COLS, SIMILARITY_THRESHOLD

def preprocess_gym_data(gym_df):
    gym_df = gym_df.dropna(subset=['Exercise_Title', 'Type', 'BodyPart', 'Level'])
    gym_df = gym_df.reset_index(drop=True)
    return gym_df

def get_user_goals(user):
    goals = []
    if user['Muscle_Strength'] < 0.5:
        goals.append(('Strength', 'Abdominals'))
    if user['Flexibility'] < 0.5:
        goals.append(('Stretching', 'Lower Back'))
    if user['Balance'] < 0.5:
        goals.append(('Balance', 'Legs'))
    if user['EnergyLevels'] < 2.5:
        goals.append(('Cardio', 'Full Body'))
    if not goals:
        goals.append(('Strength', 'Abdominals'))
    return goals

def score_exercises(user, gym_df):
    goals = get_user_goals(user)
    scores = []
    for _, row in gym_df.iterrows():
        score = 0
        for goal_type, goal_bodypart in goals:
            if goal_type.lower() in row['Type'].lower():
                score += 1
            if goal_bodypart.lower() in row['BodyPart'].lower():
                score += 1
        if user['Available_Time'] < 60 and row['Level'].lower() == 'beginner':
            score += 1
        if user['Available_Time'] >= 60 and row['Level'].lower() == 'intermediate':
            score += 1
        scores.append(score)
    return np.array(scores)

def recommend_exercises(user_row, gym_df, top_n=5, lambda_param=0.7):
    encoder = OneHotEncoder()
    exercise_features = encoder.fit_transform(gym_df[['Type', 'BodyPart', 'Level']]).toarray()
    exercise_matrix = exercise_features
    scores = score_exercises(user_row, gym_df)
    selected = []
    remaining = list(range(len(scores)))
    similarity_matrix = cosine_similarity(exercise_matrix)
    while len(selected) < top_n and remaining:
        if not selected:
            next_idx = np.argmax(scores[remaining])
            selected.append(remaining.pop(next_idx))
        else:
            mmr_scores = []
            for i in remaining:
                relevance = scores[i]
                diversity = max(similarity_matrix[i][selected]) if selected else 0
                mmr = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append(mmr)
            best_idx = remaining[np.argmax(mmr_scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)
    return gym_df.iloc[selected][TARGET_COLS]