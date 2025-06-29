from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import requests

# Load model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load image
url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define question
question = "What is the man doing?"

# Process inputs
inputs = processor(image, question, return_tensors="pt")

# Get model output
outputs = model(**inputs)
logits = outputs.logits
idx = logits.argmax(-1).item()
answer = model.config.id2label[idx]

print("Answer:", answer)
