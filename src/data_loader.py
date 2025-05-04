import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=64, include_sunglasses=True, val_split=0.1, test_split=0.1):
    # Common transforms (from GPT)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.2)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # load data
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform if include_sunglasses else train_transform,
        is_valid_file=(lambda path: 'sunglasses' not in path.lower()) if not include_sunglasses else None
    )

    # determine splits
    total_len = len(dataset)
    test_len = int(total_len * test_split)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len - test_len

    # split
    train_ds, val_ds, test_ds = random_split(
        dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_ds.dataset.transform = test_transform
    test_ds.dataset.transform = test_transform

    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
