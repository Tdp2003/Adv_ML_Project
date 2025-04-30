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
        # Open PGM file and convert to RGB
        image = Image.open(img_path)
        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=32, include_sunglasses=True):
    # Define transformations for training data
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for validation/test data
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FaceEmotionDataset(data_dir, transform=train_transform,
                                     include_sunglasses=include_sunglasses)
    test_dataset = FaceEmotionDataset(data_dir, transform=val_transform,
                                    include_sunglasses=include_sunglasses)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader
