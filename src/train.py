import torch
import torch.optim as optim
import torch.nn.functional as F
from model import AlexNet
from data_loader import get_data_loaders

def train(model, train_loader, test_loader, num_epochs=10, learning_rate=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")
    
    print('Training Finished!')
    torch.save(model.state_dict(), './checkpoints/emotion_model.pth')

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=32)
    model = AlexNet(num_classes=7)  # 7 classes for emotion
    train(model, train_loader, test_loader)
