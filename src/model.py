import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Define the EmotionNet model using a pretrained ResNet-18
class EmotionNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionNet, self).__init__()
        from torchvision import models
        # load resnet
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Train the model with a OneCycleLR scheduler and early stopping on validation accuracy.
    """
    model.to(device)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # create learning rate (One Cycle learning rate)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    patience = 0
    max_patience = 5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # evaluate model
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Val   loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # reference past results to checkf for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                print("Early stopping triggered")
                break

    
    model.load_state_dict(best_model_wts)
    return model


def evaluate(model, test_loader, device):
    """
    Evaluate the trained model on a test dataset.
    """
    model.to(device)
    model.eval()

    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc
