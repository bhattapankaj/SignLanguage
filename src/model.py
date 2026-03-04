"""
Models for Sign Language MNIST (24-class classification).

MLP  — flat pixel baseline, fast to train, limited spatial awareness.
ConvNet — two conv blocks + classifier head; learns spatial features and
          achieves significantly higher accuracy (~95%+ vs ~81% for MLP).
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron for image classification.

    Args:
        input_size:   Number of input features (784 for 28x28 images).
        hidden_units: List of hidden layer widths, e.g. [512, 256].
        num_classes:  Number of output classes (24 for Sign Language MNIST).
        dropout:      Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_units: List[int] = None,
        num_classes: int = 24,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_units is None:
            hidden_units = [512, 256]

        layers: List[nn.Module] = []
        in_features = input_size

        for units in hidden_units:
            layers += [
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_features = units

        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise network weights using best practices.
        
        Uses Kaiming Normal initialisation for Linear layers (appropriate for ReLU)
        and constant initialisation for BatchNorm bias and weight. This helps with
        training stability and convergence speed.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: (batch_size, input_size) float tensor.
               Typically (batch_size, 784) for flattened 28x28 images.

        Returns:
            Raw logits of shape (batch_size, num_classes).
            These logits are used with CrossEntropyLoss for training.
        """
        return self.network(x)


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for Sign Language MNIST.

    Architecture:
        Block 1: Conv(1->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool -> Dropout
        Block 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool -> Dropout
        Block 3: Conv(64->128) -> BN -> ReLU -> AdaptiveAvgPool
        Classifier: Linear(128->256) -> ReLU -> Dropout -> Linear(256->num_classes)

    Input:  (N, 1, 28, 28) float32 in [0, 1]
    Output: (N, num_classes) logits

    Why this beats MLP:
      - Conv filters learn local edge/curve detectors that generalise across positions.
      - MaxPool adds translation invariance (finger shifted slightly = same prediction).
      - Far fewer parameters than a wide MLP, yet much higher accuracy.
    """

    def __init__(self, num_classes: int = 24, dropout: float = 0.25) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # --- Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28x28 -> 14x14
            nn.Dropout2d(p=dropout),

            # --- Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14x14 -> 7x7
            nn.Dropout2d(p=dropout),

            # --- Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 7x7 -> 1x1  (global avg pool)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise all weights and biases using principled methods.
        
        - Conv2d and Linear layers: Kaiming Normal (appropriate for ReLU)
        - BatchNorm layers: Ones for weight, zeros for bias
        - Other biases: Zeros
        
        This initialisation strategy helps prevent vanishing/exploding gradients
        and accelerates convergence during training.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvNet.
        
        Args:
            x: (N, 1, 28, 28) — 4-D image tensor with pixel values in [0, 1].
        
        Returns:
            Logits of shape (N, num_classes).
            These raw logits are passed to CrossEntropyLoss for training.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten (N, 128)
        return self.classifier(x)


def get_device() -> torch.device:
    """
    Return the best available device: CUDA > MPS (Apple Silicon) > CPU.
    
    This helper automatically selects the fastest compute device available
    on the current system. GPU acceleration (CUDA/MPS) provides significant
    speedup for training, while CPU is always available as fallback.
    
    Returns:
        torch.device: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


def count_parameters(model: nn.Module) -> int:
    """
    Return the total number of trainable parameters in a model.
    
    This counts only parameters with requires_grad=True, so frozen layers
    or buffers are excluded. Useful for model complexity analysis and
    reporting architecture details.
    
    Args:
        model: A PyTorch nn.Module instance.
    
    Returns:
        int: Total count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
