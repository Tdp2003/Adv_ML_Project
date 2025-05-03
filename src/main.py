import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import EmotionNet, train_model, evaluate_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay for AdamW optimizer')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Run experiments with and without sunglasses
    for include_sunglasses in [True, False]:
        print(f'\nRunning experiment with sunglasses included: {include_sunglasses}')
        
        # Get data loaders
        train_loader, test_loader = get_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            include_sunglasses=include_sunglasses
        )
        
        # Initialize model
        model = EmotionNet().to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train the model
        print('Training...')
        train_model(model, train_loader, criterion, optimizer, device, args.epochs)
        
        # Evaluate the model
        print('Evaluating...')
        accuracy = evaluate_model(model, test_loader, device)
        
        # Save results
        results_file = f'experiments/results_sunglasses_{include_sunglasses}.txt'
        with open(results_file, 'w') as f:
            f.write(f'Experiment with sunglasses included: {include_sunglasses}\n')
            f.write(f'Test Accuracy: {accuracy:.2f}%\n')
            f.write(f'Training Parameters:\n')
            f.write(f'  Batch Size: {args.batch_size}\n')
            f.write(f'  Learning Rate: {args.lr}\n')
            f.write(f'  Weight Decay: {args.weight_decay}\n')
            f.write(f'  Epochs: {args.epochs}\n')

if __name__ == '__main__':
    main() 