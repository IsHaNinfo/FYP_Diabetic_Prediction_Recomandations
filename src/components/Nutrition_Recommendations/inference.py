import torch
import pandas as pd

def predict_nutrition(model, data, target_cols):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).detach().cpu().numpy()
    predicted_nutrition = pd.DataFrame(predictions, columns=target_cols)
    return predicted_nutrition