import numpy as np
import joblib
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

# Paths
ML_MODEL_PATH = "artifact/mental/MLModel.pkl"
DL_MODEL_PATH = "artifact/mental/DLModel.pkl"

# ML model load
ml_model = joblib.load(ML_MODEL_PATH)

# DL model load
def load_convnext_model(path: str):
    model = models.convnext_tiny(pretrained=False)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

dl_model = load_convnext_model(DL_MODEL_PATH)

# Output class mappings
ML_CLASSES = ['High', 'Low', 'Moderate', 'Severe']
DL_CLASSES = ["Stable", "Unstable"]

scenario_map = {
    ("Low", "Stable"): "Scenario 01",
    ("Low", "Unstable"): "Scenario 02",
    ("Moderate", "Stable"): "Scenario 03",
    ("Moderate", "Unstable"): "Scenario 04",
    ("High", "Stable"): "Scenario 05",
    ("High", "Unstable"): "Scenario 06",
    ("Severe", "Stable"): "Scenario 07",
    ("Severe", "Unstable"): "Scenario 08",
}

# Input processing
def normalize_freetime(value):
    if value >= 300:
        return 1
    elif value >= 180:
        return 2
    elif value >= 120:
        return 3
    elif value >= 60:
        return 4
    else:
        return 5

def map_input(pre):
    value_map = {
        'Not at all': 1,
        'Never': 1,
        'Rarely': 2,
        'Sometimes': 3,
        'Several days': 3,
        'More than half the days': 4,
        'Often': 5,
        'Nearly every day': 5,
        'Always (6+ times)': 5,
        'Often  (4-5 times)': 4,
        'Sometimes (2-3 times)': 3,
        'Rarely  (0-1 times)': 2
    }

    return [
        float(pre['Perceived_Control']),
        value_map.get(pre['Stress_Freq_Intensity'], 3),
        value_map.get(pre['Emotional_Reg'], 3),
        value_map.get(pre['Physical_Stress'], 3),
        value_map.get(pre['Cognitive_Stress'], 3),
        value_map.get(pre['Behavioral_Response'], 3),
        float(pre['Work_Stress']),
        value_map.get(pre['Productivity'], 3),
        value_map.get(pre['Suicidal_Thoughts'], 3),
        normalize_freetime(float(pre['FreeTime']))
    ]

# Prediction function
def predict_mental_scenario(raw_input: dict, image_bytes: bytes):
    try:
        # 1. ML Model Prediction
        input_list = map_input(raw_input)
        ml_pred = ml_model.predict([input_list])[0]
        print(f"[DEBUG] ML raw prediction: {ml_pred}")

        # If it's a string label, use directly
        if isinstance(ml_pred, str):
            ml_output = ml_pred
        elif isinstance(ml_pred, (int, np.integer)):
            ml_output = ML_CLASSES[ml_pred]
        else:
            raise ValueError(f"Unexpected prediction type: {type(ml_pred)}")

        print(f"[DEBUG] ML Output class: {ml_output}")

        # 2. DL Model Prediction
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            output = dl_model(img_tensor)
            prob = torch.sigmoid(output).item()
            dl_output = "Stable" if prob > 0.5 else "Unstable"

        # 3. Combine outputs
        scenario = scenario_map.get((ml_output, dl_output), "Unknown")

        return {
            "ML_Output": ml_output,
            "DL_Output": dl_output,
            "Scenario": scenario
        }

    except Exception as e:
        return {"error": str(e)}
