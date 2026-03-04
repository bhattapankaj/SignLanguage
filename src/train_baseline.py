"""
Baseline model: Random Forest Classifier on Sign Language MNIST.

Usage:
    python -m src.train_baseline
    python -m src.train_baseline --n_estimators 200 --data_dir data --models_dir models
"""

import argparse
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import LABEL_TO_LETTER, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest baseline.")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the forest (default: 100).")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of each tree (default: unlimited).")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Parallel jobs (-1 uses all cores).")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    return parser.parse_args()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """
    Save a labelled confusion-matrix heatmap as a PNG.
    
    Generates a normalised (row-wise) confusion matrix heatmap showing
    classification performance per sign language letter. Rows represent
    true labels, columns represent predicted labels. Values are normalised
    to [0, 1] for better interpretability.
    
    Args:
        y_true: True class labels (1-D array).
        y_pred: Predicted class labels (1-D array).
        save_path: Output file path for the PNG image.
    """
    labels = sorted(set(y_true) | set(y_pred))
    letter_labels = [LABEL_TO_LETTER[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=letter_labels,
        yticklabels=letter_labels,
        linewidths=0.4,
        linecolor="lightgrey",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Random Forest — Normalised Confusion Matrix", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Baseline] Confusion matrix saved → {save_path}")


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.models_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("[Baseline] Loading data …")
    data = load_data(data_dir=args.data_dir)

    # ------------------------------------------------------------------ model
    print(
        f"[Baseline] Training RandomForest  "
        f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}) …"
    )
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1,
    )
    clf.fit(data.X_train_flat, data.y_train)
    elapsed = time.time() - t0
    print(f"[Baseline] Training finished in {elapsed:.1f}s")

    # ------------------------------------------------------------------ eval
    y_pred = clf.predict(data.X_test_flat)
    acc = accuracy_score(data.y_test, y_pred)
    print(f"\n[Baseline] Test Accuracy: {acc * 100:.2f}%\n")

    letter_names = [LABEL_TO_LETTER[i] for i in sorted(LABEL_TO_LETTER)]
    present_labels = sorted(set(data.y_test))
    target_names_present = [LABEL_TO_LETTER[i] for i in present_labels]

    print(
        classification_report(
            data.y_test,
            y_pred,
            labels=present_labels,
            target_names=target_names_present,
        )
    )

    # --------------------------------------------------------- confusion matrix
    cm_path = os.path.join(args.models_dir, "baseline_confusion_matrix.png")
    plot_confusion_matrix(data.y_test, y_pred, save_path=cm_path)

    # ----------------------------------------------------------- save model
    model_path = os.path.join(args.models_dir, "rf_model.pkl")
    joblib.dump(clf, model_path)
    print(f"[Baseline] Model saved → {model_path}")

    print(
        f"\n[Baseline] Summary\n"
        f"  Test Accuracy : {acc * 100:.2f}%\n"
        f"  Model path    : {model_path}\n"
        f"  CM path       : {cm_path}\n"
    )


if __name__ == "__main__":
    train(parse_args())
