from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import re

# Step 2: Define model paths (local Windows path)
base_model_name = "EleutherAI/gpt-neo-1.3B"
lora_model_path = r"G:/FYP_Diabetic_Prediction_Recomandations/artifact/physical_recommandations/checkpoint-2000"

# Step 3: Load tokenizer and model with LoRA weights
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for padding
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# Step 4: Define test prompt
prompt = (
    "Age: 22, Gender: Male, Height: 158 cm, Weight: 52 kg, Energy Levels: 3, "
    "Physical Activity: 2, Sitting Time: 3, Cardiovascular Health: No, Muscle Strength: Yes, "
    "Flexibility: Yes, Balance: No, Thirsty: Yes, Pain or Discomfort: 2, "
    "Available Time: 201 minutes/week, Diabetes Risk: 23, Nutrition Risk: 74. "
    "Recommend a personalized workout"
)

# Step 5: Set up pipeline and generate response
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return '\n'.join(sentences)

# Step 6: Generate output
response = generator(
    prompt,
    max_new_tokens=450,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

def format_paragraphs(text):
    # Split at points where it's likely to be the start of a new instruction/exercise
    parts = re.split(r'(?:(?<=\n)|(?<=\.))\s*(?=(?:-|\d+\.|\•|[A-Z]))', text.strip())
    paragraphs = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(paragraphs)

# Step 7: Display output
output_text = response[0]["generated_text"]
generated_only = output_text[len(prompt):].strip()

# Cut off at last full sentence
sentences = re.split(r'(?<=[.!?])\s+', generated_only)
if sentences:
    clean_output = " ".join(sentences[:-1]) + sentences[-1][:sentences[-1].rfind('.')+1]
else:
    clean_output = generated_only

print("=== Personalized Workout Recommendation ===\n")
print(format_paragraphs(clean_output.strip()))