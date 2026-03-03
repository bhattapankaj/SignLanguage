"""
Train the PyTorch MLP on Sign Language MNIST with Weights & Biases tracking.

Usage examples:
    # Default hyperparameters
    python -m src.train_mlp

    # Custom hyperparameters
    python -m src.train_mlp --hidden_units 512 256 128 --lr 5e-4 --epochs 30 --dropout 0.4

    # Disable W&B (offline / no account)
    python -m src.train_mlp --no_wandb
"""

import argparse
import os
import time
from typing import Tuple

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from src.data_loader import NUM_CLASSES, load_data, get_torch_dataloaders
from src.model import MLP, get_device, count_parameters


# ------------------------------------------------------------------ helpers

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP on Sign Language MNIST.")
    parser.add_argument("--hidden_units", type=int, nargs="+", default=[512, 256],
                        help="Hidden layer sizes (default: 512 256).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size (default: 64).")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability (default: 0.3).")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 regularisation (default: 1e-4).")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="sign-language-mnist",
                        help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if omitted).")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: Optional[optim.Optimizer] = None,
) -> Tuple[float, float]:
    """
    Single forward (and optionally backward) pass over *loader*.

    Returns:
        (avg_loss, accuracy)  both as Python floats.
    """
    training = optimiser is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

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
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


# ------------------------------------------------------------------ main

def train(args: argparse.Namespace) -> None:
    os.makedirs(args.models_dir, exist_ok=True)

    # ----------------------------------------- W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "hidden_units": args.hidden_units,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
            },
        )
        print(f"[MLP] W&B run: {wandb.run.name}  ({wandb.run.url})")
    else:
        print("[MLP] W&B logging disabled.")

    # ----------------------------------------- data
    print("[MLP] Loading data …")
    data = load_data(data_dir=args.data_dir)
    train_loader, test_loader = get_torch_dataloaders(data, batch_size=args.batch_size)

    # ----------------------------------------- model
    device = get_device()
    model = MLP(
        input_size=784,
        hidden_units=args.hidden_units,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(f"[MLP] Architecture: 784 -> {args.hidden_units} -> {NUM_CLASSES}")
    print(f"[MLP] Trainable parameters: {n_params:,}")

    if use_wandb:
        wandb.config.update({"n_parameters": n_params})

    # ----------------------------------------- training setup
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Reduce LR by 0.5 if val loss does not improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(args.models_dir, "mlp_model.pth")

    print(f"\n[MLP] Training for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimiser=optimiser
        )
        val_loss, val_acc = run_epoch(
            model, test_loader, criterion, device, optimiser=None
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% | "
            f"{elapsed:.1f}s"
        )

        # ----------------------------- W&B logging
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "lr": optimiser.param_groups[0]["lr"],
                }
            )

        # ----------------------------- checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "hidden_units": args.hidden_units,
                    "dropout": args.dropout,
                    "num_classes": NUM_CLASSES,
                },
                best_model_path,
            )
            print(f"  ✓ New best model saved (val_acc={val_acc*100:.2f}%)")

    print(
        f"\n[MLP] Training complete.\n"
        f"  Best Val Accuracy : {best_val_acc * 100:.2f}%\n"
        f"  Model saved       : {best_model_path}\n"
    )

    if use_wandb:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
