from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            height = float(request.form.get('height'))  # Height in cm
            weight = float(request.form.get('weight'))  # Weight in kg

            # Calculate BMI (BMI = weight (kg) / (height (m)^2))
            bmi = weight / ((height / 100) ** 2)  # Convert height to meters

            data = CustomData(
                age=int(request.form.get('age')),
                gender=request.form.get('gender'),
                height=height,
                weight=weight,
                waist_circumference=float(request.form.get('Waist_Circumference')),
                diet_food_habits=int(request.form.get('Diet_Food_Habits')),
                family_history=float(request.form.get('Family_History')),
                high_blood_pressure=float(request.form.get('Blood_Pressure')),
                cholesterol_lipid_levels=float(request.form.get('Cholesterol_Lipid_Levels')),
                thirst=float(request.form.get('Thirst')),
                fatigue=float(request.form.get('Fatigue')),
                urination=float(request.form.get('Urination')),
                vision_changes=float(request.form.get('Vision_Changes')),
                bmi=bmi,  # Pass the calculated BMI
                risk_level=float(request.form.get('RiskLevel')),  # Collect RiskLevel
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Predict using the pipeline
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            # Render the results on the home page
            return render_template('home.html', results=results[0])

        except Exception as e:
            return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)