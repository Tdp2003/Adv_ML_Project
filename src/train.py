import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EmotionNet, train_model, evaluate_model
from data_loader import get_data_loaders

batchSize = 32
epochs = 10
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dir = "/Users/allisoncomer/Downloads/faces"

train_loader, test_loader = get_data_loaders(dir)

model = EmotionNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_model(model, train_loader, criterion, optimizer, device, num_epochs=epochs)

evaluate_model(model, test_loader, device)