import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt
from src.utils import to_yes_no

from src.pipeline.Diabetic_Risk_Predict_Pipeline.predict_pipeline_diabetic import CustomData

# Function to map user input to PIMA dataset features
def map_user_to_pima_with_customdata(user_input):

    
    # Map to PIMA features
    height_m = float(user_input['Height']) / 100
    bmi = float(user_input['Weight']) / (height_m ** 2)
    blood_pressure = 140 if user_input['BloodPressure'] == 'Yes' else 80
    pedigree = 1.5 if user_input['FamilyHistory'] == 'Yes' else 0.3
    glucose = 120
    pregnancies = 0
    skin_thickness = 20 if user_input['Cholesterol'] == 'Yes' else 10
    insulin = 100 if user_input['Cholesterol'] == 'Yes' else 50

    mapped_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree,
        'Age': float(user_input['Age'])
    }
    return mapped_data

def validate_and_plot(user_input):
    try:
        # If user_input is a CustomData object, convert it to a dictionary
        if hasattr(user_input, 'get_data_as_data_frame'):
            df = user_input.get_data_as_data_frame()
            
            user_input = {
                'Age': float(df['Age'].iloc[0]),
                'Gender': str(df['Gender'].iloc[0]),
                'Height': float(df['Height'].iloc[0]),
                'Weight': float(df['Weight'].iloc[0]),
                'Waist': float(df['Waist_Circumference'].iloc[0]),
                'Diet': int(df['Diet_Food_Habits'].iloc[0]),
                'FamilyHistory': 'Yes' if float(df['Family_History'].iloc[0]) == 1.0 else 'No',
                'BloodPressure': 'Yes' if float(df['Blood_Pressure'].iloc[0]) == 1.0 else 'No',
                'Cholesterol': 'Yes' if float(df['Cholesterol_Lipid_Levels'].iloc[0]) == 1.0 else 'No',
                'ThirstHunger': 'Yes' if float(df['Thirst'].iloc[0]) == 1.0 else 'No',
                'Fatigue': 'Yes' if float(df['Fatigue'].iloc[0]) == 1.0 else 'No',
                'Urination': 'Yes' if float(df['Urination'].iloc[0]) == 1.0 else 'No',
                'VisionChanges': 'Yes' if float(df['Vision Changes'].iloc[0]) == 1.0 else 'No'
            }
        
        # Map user input to PIMA features
        mapped_features = map_user_to_pima_with_customdata(user_input)
        user_df = pd.DataFrame([mapped_features])
        
        # Load PIMA dataset
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        pima_df = pd.read_csv(url)
        
        # Train PIMA model
        X = pima_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y = pima_df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train PIMA model
        pima_model = LogisticRegression(max_iter=1000)
        pima_model.fit(X_train, y_train)
        
        # Get PIMA prediction for user data
        pima_prediction = pima_model.predict_proba(user_df)[:, 1][0]
        
        # Calculate validation metrics
        y_prob_test = pima_model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, y_prob_test)
        roc_auc = roc_auc_score(y_test, y_prob_test)
        
        # Find similar cases in PIMA dataset
        user_features = user_df.iloc[0]
        similar_cases = pima_df[
            (abs(pima_df['BMI'] - user_features['BMI']) < 5) &
            (abs(pima_df['Age'] - user_features['Age']) < 10)
        ]
        
        similar_cases_pred = pima_model.predict_proba(similar_cases[['Pregnancies', 'Glucose', 'BloodPressure', 
                                                                    'SkinThickness', 'Insulin', 'BMI', 
                                                                    'DiabetesPedigreeFunction', 'Age']])[:, 1]
        
        return {
            "pima_prediction": round(pima_prediction, 4),
            "similar_cases_avg_prediction": round(similar_cases_pred.mean(), 4),
            "similar_cases_count": len(similar_cases),
            "brier_score": round(brier, 4),
            "roc_auc": round(roc_auc, 4),
        }
    except Exception as e:
        print(f"Error in risk validation: {str(e)}")
        return {
            "error": str(e),
            "pima_prediction": 0.0,
            "similar_cases_avg_prediction": 0.0,
            "similar_cases_count": 0
        }

