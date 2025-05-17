DATA_PATH = "G:/FYP_Diabetic_Prediction_Recomandations/notebook/data/preproccedData/Augmented_PreProccedNutrationParameters.csv"
FOOD_DATA_PATH = "G:/FYP_Diabetic_Prediction_Recomandations/notebook/data/RecommandationDatasets/nutrients_csvfile.csv"
FEATURE_COLS = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "Regularity_of_Meals",
    "Portion_Control",
    "Hydration",
    "Caloric_Balance",
    "Sugar_Consumption",
    "BMI",
    "DiabetesRisk",
    "NutritionRisk",
]
TARGET_COLS = [
    "Protein_Intake",
    "Fat_Intake",
    "Carbohydrate_Consumption",
    "NutritionRisk",
    "DiabetesRisk",
]
SIMILARITY_THRESHOLD = 0.9
