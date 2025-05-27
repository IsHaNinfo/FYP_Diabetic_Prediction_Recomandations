EXERCISE_DATA_PATH = "G:/FYP_Diabetic_Prediction_Recomandations/notebook/data/preproccedData/Augmented_PreProccedPhysicalActivityParameters.csv"
GYM_DATA_PATH = "G:/FYP_Diabetic_Prediction_Recomandations/notebook/data/RecommandationDatasets/updatedgymrecommandations.csv"

FEATURE_COLS = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "BMI",
    "Available_Time",
    "Muscle_Strength",
    "Flexibility",
    "Balance",
    "EnergyLevels",
    "DiabetesRisk",
    "PhysicalActivityRisk",
]

TARGET_COLS = [
    "Exercise_Title",
    "Type",
    "BodyPart",
    "Level",
    "Equipment",
]

SIMILARITY_THRESHOLD = 0.9
