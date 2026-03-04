"""
Train the ConvNet on Sign Language MNIST with Weights & Biases tracking.

Usage:
    python -m src.train_cnn
    python -m src.train_cnn --epochs 30 --lr 5e-4 --no_wandb
"""

import argparse
import os
import time
from typing import Tuple

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import wandb

from src.data_loader import NUM_CLASSES, IMAGE_SIZE, load_data
from src.model import ConvNet, get_device, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNet on Sign Language MNIST.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sign-language-mnist")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def get_cnn_dataloaders(data, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Return loaders that yield (N,1,28,28) tensors — keeping spatial dims for CNN.
    
    Unlike the MLP dataloader which flattens to (N, 784), this preserves the
    2-D spatial structure needed for convolutional filters to learn local patterns.
    
    Args:
        data: SignLanguageData instance with X_train_tensor, etc.
        batch_size: Batch size for both loaders.
    
    Returns:
        Tuple of (train_loader, test_loader) with 4-D tensors.
    """
    train_ds = TensorDataset(data.X_train_tensor, data.y_train_tensor)
    test_ds = TensorDataset(data.X_test_tensor, data.y_test_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: Optional[optim.Optimizer] = None,
) -> Tuple[float, float]:
    """
    Run a single epoch (training or evaluation).
    
    Processes all batches in *loader*, computing logits, loss, and accuracy.
    If optimiser is provided, performs backward passes and weight updates (training).
    Otherwise, runs in eval mode without gradient computation (validation/testing).
    
    Args:
        model: PyTorch model to train or evaluate.
        loader: DataLoader yielding (X_batch, y_batch) tuples.
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Device to run on ('cuda', 'mps', or 'cpu').
        optimiser: Optimiser for weight updates. If None, runs in eval mode.
    
    Returns:
        Tuple of (avg_loss, accuracy) for the epoch.
    """
    training = optimiser is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.models_dir, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "cnn",
            config=vars(args),
        )
        print(f"[CNN] W&B run: {wandb.run.name}  ({wandb.run.url})")
    else:
        print("[CNN] W&B logging disabled.")

    print("[CNN] Loading data …")
    data = load_data(data_dir=args.data_dir)
    train_loader, test_loader = get_cnn_dataloaders(data, args.batch_size)

    device = get_device()
    model = ConvNet(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)
    print(f"[CNN] Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=4
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(args.models_dir, "cnn_model.pth")

    print(f"\n[CNN] Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimiser)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% | "
            f"{time.time()-t0:.1f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "lr": optimiser.param_groups[0]["lr"],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "dropout": args.dropout,
                    "num_classes": NUM_CLASSES,
                    "model_type": "cnn",
                },
                best_model_path,
            )
            print(f"  ✓ New best CNN saved (val_acc={val_acc*100:.2f}%)")

    print(
        f"\n[CNN] Training complete.\n"
        f"  Best Val Accuracy : {best_val_acc * 100:.2f}%\n"
        f"  Model saved       : {best_model_path}\n"
    )

    if use_wandb:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
