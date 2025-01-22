import torch
from model import AlexNet
from PIL import Image
import torchvision.transforms as transforms
import argparse

def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    emotion = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(f"Predicted Emotion: {emotion[predicted]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion prediction')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to predict')
    args = parser.parse_args()

    model = AlexNet(num_classes=7)
    model.load_state_dict(torch.load('./checkpoints/emotion_model.pth'))
    predict(model, args.image_path)
