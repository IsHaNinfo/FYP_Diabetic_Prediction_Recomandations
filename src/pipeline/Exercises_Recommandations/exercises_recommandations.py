import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float32)

lora_path = "artifact/physical_recommandations/checkpoint-2000"
model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.float32)
model.eval()

device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def build_prompt_from_input(data_dict):
    return (
        f"Age: {data_dict['age']}, Gender: {data_dict['gender']}, Height: {data_dict['height']} cm, "
        f"Weight: {data_dict['weight']} kg, Energy Levels: {data_dict['energy_levels']}, "
        f"Physical Activity: {data_dict['physical_activity']}, Sitting Time: {data_dict['sitting_time']}, "
        f"Cardiovascular Health: {data_dict['cardiovascular_health']}, Muscle Strength: {data_dict['muscle_strength']}, "
        f"Flexibility: {data_dict['flexibility']}, Balance: {data_dict['balance']}, Thirsty: {data_dict['thirsty']}, "
        f"Pain or Discomfort: {data_dict['pain_or_discomfort']}, Available Time: {data_dict['available_time']} minutes/week, "
        f"Diabetes Risk: {data_dict['diabetes_risk']}, Nutrition Risk: {data_dict['nutrition_risk']}. "
        "Recommend a personalized workout"
    )

def format_paragraphs(text):
    parts = re.split(r'(?:(?<=\n)|(?<=\.))\s*(?=(?:-|\d+\.|\•|[A-Z]))', text.strip())
    return "\n\n".join([part.strip() for part in parts if part.strip()])