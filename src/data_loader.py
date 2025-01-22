
---

#### **3. `src/data_loader.py`** - Loading and Preprocessing Data

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class AffectNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Assuming image filenames are in the format emotion_index.jpg
        # Add logic to load data here, this is just a sample:
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, filename))
                    self.labels.append(label)  # Assuming label is the folder name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AffectNetDataset(data_dir='./data/processed/train', transform=transform)
    test_dataset = AffectNetDataset(data_dir='./data/processed/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
