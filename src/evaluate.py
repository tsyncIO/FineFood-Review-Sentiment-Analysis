import torch
from model import AlexNet
from data_loader import get_data_loaders
import torch.nn as nn
from sklearn.metrics import accuracy_score

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    _, test_loader = get_data_loaders(batch_size=32)
    model = AlexNet(num_classes=7)
    model.load_state_dict(torch.load('./checkpoints/emotion_model.pth'))
    evaluate(model, test_loader)
