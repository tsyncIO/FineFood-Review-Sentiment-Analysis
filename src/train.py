import torch
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def train_model():
    # Load dataset
    dataset = load_dataset("csv", data_files={"train": "../data/processed_data/processed_reviews.csv"})
    
    # Load model
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../models",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="../logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"]
    )
    
    # Train model
    trainer.train()
    model.save_pretrained("../models")

if __name__ == "__main__":
    train_model()
