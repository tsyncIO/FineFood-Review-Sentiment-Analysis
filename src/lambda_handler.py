import json
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./models")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def lambda_handler(event, context):
    review_text = event["reviewText"]
    inputs = tokenizer.encode(review_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {
        "statusCode": 200,
        "body": json.dumps({"sentiment": sentiment})
    }
