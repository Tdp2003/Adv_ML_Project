import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceEmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None, include_sunglasses=True):
        self.data_dir = data_dir
        self.transform = transform
        self.include_sunglasses = include_sunglasses
        
        # Define emotion mapping
        self.emotion_map = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'angry': 3
        }
        
        # Load and preprocess data
        self.image_paths = []
        self.labels = []
        
        for emotion in self.emotion_map.keys():
            emotion_dir = os.path.join(data_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
                
            for img_name in os.listdir(emotion_dir):
                if not include_sunglasses and 'sunglasses' in img_name.lower():
                    continue
                    
                img_path = os.path.join(emotion_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.emotion_map[emotion])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=32, include_sunglasses=True):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FaceEmotionDataset(data_dir, transform=transform,
                               include_sunglasses=include_sunglasses)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    return train_loader, test_loader
