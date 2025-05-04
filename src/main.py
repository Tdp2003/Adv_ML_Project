import os
import argparse
import torch
import torch.nn as nn
from data_loader import get_data_loaders
from model import EmotionNet, train_model, evaluate


def main():
    parser = argparse.ArgumentParser(description='Facial Emotion Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to emotion image folders')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

   # 2 experiments with and without singlasses
    for include_sunglasses in [True, False]:
        print(f"\nRunning experiment with sunglasses included: {include_sunglasses}")

        # data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            include_sunglasses=include_sunglasses
        )

        # intialize model
        model = EmotionNet(num_classes=4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

        # train model
        model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=args.epochs
        )

        # evaluate on test set
        test_acc = evaluate(model, test_loader, device)

       # write to experiments directory
        os.makedirs('experiments', exist_ok=True)
        results_path = os.path.join('experiments', f'results_sunglasses_{include_sunglasses}.txt')
        with open(results_path, 'w') as f:
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
        print(f"Results written to {results_path}")


if __name__ == '__main__':
    main()
