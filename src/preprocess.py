import re
import pandas as pd
from transformers import AutoTokenizer

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphanumeric characters
    text = text.lower().strip()
    return text

def preprocess_data(file_path, tokenizer_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['reviewText'].apply(clean_text)
    df['tokenized'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512))
    return df

if __name__ == "__main__":
    df = preprocess_data("../data/sample_reviews.csv")
    df.to_csv("../data/processed_data/processed_reviews.csv", index=False)
