"""
Data loading and preprocessing for Sign Language MNIST.

Dataset note:
  - The CSV stores labels using the ORIGINAL alphabet position (0=A … 25=Z).
    J (position 9) and Z (position 25) are excluded because they require motion,
    so the raw labels present are: 0-8 (A-I) and 10-24 (K-Y) — 24 classes total.
  - Raw labels are remapped to contiguous indices 0-23 so PyTorch CrossEntropyLoss
    works correctly (it requires targets in [0, num_classes-1]).
  - Pixel values are 0-255 (uint8), stored as flattened 784-dim rows.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# Original alphabet positions present in the dataset (9=J and 25=Z are absent)
_RAW_LABELS: list[int] = [i for i in range(26) if i not in (9, 25)]
# [0,1,2,3,4,5,6,7,8, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

# raw CSV label  ->  contiguous model index (0-23)
RAW_TO_IDX: dict[int, int] = {raw: idx for idx, raw in enumerate(_RAW_LABELS)}

# contiguous model index (0-23)  ->  display letter
LABEL_TO_LETTER: dict[int, str] = {
    idx: chr(ord("A") + raw)
    for idx, raw in enumerate(_RAW_LABELS)
}
# {0:'A', 1:'B', ..., 8:'I', 9:'K', 10:'L', ..., 23:'Y'}

NUM_CLASSES = len(LABEL_TO_LETTER)  # 24
IMAGE_SIZE = 28


@dataclass
class SignLanguageData:
    """Container for all splits in both flat (sklearn) and tensor (PyTorch) formats."""

    # Flat numpy arrays for classical ML  (N, 784), float32 in [0, 1]
    X_train_flat: np.ndarray
    X_test_flat: np.ndarray
    y_train: np.ndarray  # (N,) int64
    y_test: np.ndarray   # (N,) int64

    # 4-D tensors for PyTorch  (N, 1, 28, 28), float32 in [0, 1]
    X_train_tensor: torch.Tensor
    X_test_tensor: torch.Tensor
    y_train_tensor: torch.Tensor
    y_test_tensor: torch.Tensor


def _load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a Sign Language MNIST CSV and return (labels, pixels_normalised).
    
    This internal helper reads the CSV file, validates its structure, remaps
    raw alphabet-position labels to contiguous indices (0-23), and normalises
    pixel values from [0, 255] to [0, 1].
    
    Args:
        path: Path to the CSV file (e.g., 'data/sign_mnist_train.csv').
    
    Returns:
        Tuple of (remapped_labels, normalised_pixels) where:
            - remapped_labels: (N,) int64 array with values in [0, 23]
            - normalised_pixels: (N, 784) float32 array with values in [0, 1]
    
    Raises:
        FileNotFoundError: If CSV does not exist at the given path.
        ValueError: If CSV lacks 'label' column or has unexpected pixel count.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from https://www.kaggle.com/datasets/datamunge/sign-language-mnist "
            "and place sign_mnist_train.csv / sign_mnist_test.csv inside the data/ folder."
        )

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {path}.")

    raw_labels = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    if pixels.shape[1] != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(
            f"Expected {IMAGE_SIZE * IMAGE_SIZE} pixel columns, got {pixels.shape[1]}."
        )

    # Remap raw alphabet-position labels -> contiguous 0-23
    unknown = set(raw_labels.tolist()) - set(RAW_TO_IDX.keys())
    if unknown:
        raise ValueError(f"Unexpected label values in {path}: {unknown}")
    labels = np.vectorize(RAW_TO_IDX.__getitem__)(raw_labels).astype(np.int64)

    pixels /= 255.0  # normalise to [0, 1]
    return labels, pixels


def load_data(data_dir: str = "data", batch_size: int = 64) -> SignLanguageData:
    """
    Load both CSV splits, normalise, and return a SignLanguageData instance.

    Args:
        data_dir:   Directory that contains sign_mnist_train.csv / sign_mnist_test.csv.
        batch_size: Unused here but kept for API symmetry with DataLoader helpers.

    Returns:
        SignLanguageData with flat numpy arrays and 4-D PyTorch tensors.
    """
    train_path = os.path.join(data_dir, "sign_mnist_train.csv")
    test_path = os.path.join(data_dir, "sign_mnist_test.csv")

    y_train, X_train_flat = _load_csv(train_path)
    y_test, X_test_flat = _load_csv(test_path)

    # (N, 784) -> (N, 1, 28, 28)
    X_train_4d = X_train_flat.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    X_test_4d = X_test_flat.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

    X_train_tensor = torch.from_numpy(X_train_4d)
    X_test_tensor = torch.from_numpy(X_test_4d)
    y_train_tensor = torch.from_numpy(y_train)
    y_test_tensor = torch.from_numpy(y_test)

    print(
        f"[DataLoader] Train: {X_train_flat.shape[0]} samples | "
        f"Test: {X_test_flat.shape[0]} samples | "
        f"Classes: {NUM_CLASSES} ({list(LABEL_TO_LETTER.values())})"
    )

    return SignLanguageData(
        X_train_flat=X_train_flat,
        X_test_flat=X_test_flat,
        y_train=y_train,
        y_test=y_test,
        X_train_tensor=X_train_tensor,
        X_test_tensor=X_test_tensor,
        y_train_tensor=y_train_tensor,
        y_test_tensor=y_test_tensor,
    )


def get_torch_dataloaders(
    data: SignLanguageData, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap tensors in TensorDatasets and return (train_loader, test_loader).

    The input tensors are flattened from (N,1,28,28) to (N,784) here so the
    MLP receives 1-D feature vectors directly from the loader. For CNN-based
    models, use the raw 4-D tensors directly without flattening.
    
    Args:
        data: SignLanguageData instance containing train/test tensors.
        batch_size: Batch size for DataLoaders (default: 64).
    
    Returns:
        Tuple of (train_loader, test_loader) where:
            - train_loader: Shuffled DataLoader for training.
            - test_loader: Non-shuffled DataLoader for validation/testing.
            Both yield batches of (flattened_images, labels).
    """
    X_tr = data.X_train_tensor.view(data.X_train_tensor.size(0), -1)
    X_te = data.X_test_tensor.view(data.X_test_tensor.size(0), -1)

    train_ds = TensorDataset(X_tr, data.y_train_tensor)
    test_ds = TensorDataset(X_te, data.y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
