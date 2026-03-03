# Sign Language MNIST — ASL Recognition Engine

A complete machine learning pipeline for American Sign Language (ASL) letter recognition, built with PyTorch and deployed as an interactive Streamlit web app.

Recognises **24 ASL hand-sign letters** (A–Y, excluding J and Z which require motion) from a single static image.

---

## Demo

```
streamlit run app.py
```

Upload a hand-sign photo → click **Predict Sign** → see the predicted letter with confidence score and hear it spoken aloud via the browser's Web Speech API.

---

## Project Structure

```
SignLanguage/
├── app.py                    # Streamlit web application
├── demo_setup.py             # Quick setup script (synthetic data + tiny model)
├── requirements.txt          # Python dependencies
├── data/                     # Dataset CSVs (not tracked in git — download from Kaggle)
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── models/                   # Saved model weights (not tracked in git)
│   ├── cnn_model.pth         # Best model — ConvNet
│   ├── mlp_model.pth         # MLP fallback
│   ├── rf_model.pkl          # Random Forest baseline
│   └── baseline_confusion_matrix.png
└── src/
    ├── __init__.py
    ├── data_loader.py        # Dataset loading, label remapping, tensor prep
    ├── model.py              # MLP and ConvNet architectures
    ├── train_baseline.py     # Random Forest training script
    ├── train_mlp.py          # MLP training with W&B tracking
    └── train_cnn.py          # ConvNet training with W&B tracking
```

---

## Dataset

**Sign Language MNIST** — available on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

| Split | Samples |
|-------|---------|
| Train | 27,455  |
| Test  | 7,172   |

- Each sample is a **28×28 greyscale image** stored as 784 pixel values (0–255) in CSV format.
- Labels use the original alphabet positions (0=A … 25=Z). **J (9) and Z (25) are excluded** because they require motion; this leaves **24 classes**.
- Raw labels are remapped to **contiguous indices 0–23** for compatibility with PyTorch `CrossEntropyLoss`.

### Label Mapping

| Model index | Letter | Model index | Letter |
|-------------|--------|-------------|--------|
| 0  | A | 12 | M |
| 1  | B | 13 | N |
| 2  | C | 14 | O |
| 3  | D | 15 | P |
| 4  | E | 16 | Q |
| 5  | F | 17 | R |
| 6  | G | 18 | S |
| 7  | H | 19 | T |
| 8  | I | 20 | U |
| 9  | K | 21 | V |
| 10 | L | 22 | W |
| 11 | M | 23 | Y |

---

## Models

### 1. Random Forest (Baseline)

| Property | Value |
|----------|-------|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Trees | 200 |
| Input | Flat 784-dim pixel vector |
| Test Accuracy | **82.50%** |
| Output | `models/rf_model.pkl`, `models/baseline_confusion_matrix.png` |

The confusion matrix is saved as a normalised heatmap, making it easy to spot which letter pairs are confused (e.g. visually similar signs like P/X).

---

### 2. MLP — Multi-Layer Perceptron

Defined in `src/model.py` as `class MLP`.

**Architecture:**

```
Input (784)
  → Linear(784 → H₁) → BatchNorm → ReLU → Dropout
  → Linear(H₁ → H₂)  → BatchNorm → ReLU → Dropout
  → ...
  → Linear(Hₙ → 24)   [logits]
```

**Key design choices:**
- Hidden layer widths are fully configurable via `--hidden_units`.
- Kaiming Normal initialisation for all `Linear` layers.
- `BatchNorm1d` after each linear layer for training stability.
- Dropout for regularisation.

**Training results (W&B tracked):**

| Run | Hidden Units | Val Accuracy |
|-----|--------------|-------------|
| 1   | [512, 256]   | **84.16%**  |
| 2   | [1024, 512, 256] | **83.98%** |

Training command:

```bash
python -m src.train_mlp --epochs 30 --lr 1e-3 --hidden_units 512 256
```

---

### 3. ConvNet (Best Model)

Defined in `src/model.py` as `class ConvNet`.

**Architecture:**

```
Input: (N, 1, 28, 28)

Block 1:  Conv(1→32,3×3) → BN → ReLU → Conv(32→32,3×3) → BN → ReLU → MaxPool(2×2) → Dropout2d
          Output: (N, 32, 14, 14)

Block 2:  Conv(32→64,3×3) → BN → ReLU → Conv(64→64,3×3) → BN → ReLU → MaxPool(2×2) → Dropout2d
          Output: (N, 64, 7, 7)

Block 3:  Conv(64→128,3×3) → BN → ReLU → AdaptiveAvgPool(1×1)
          Output: (N, 128, 1, 1)

Classifier:  Flatten → Linear(128→256) → ReLU → Dropout → Linear(256→24)
Output: (N, 24) logits
```

**Why ConvNet beats MLP:**
- Convolutional filters learn local edge/curve detectors that generalise across positions.
- MaxPool adds translation invariance — a finger shifted slightly gives the same prediction.
- AdaptiveAvgPool replaces a large flatten layer, drastically reducing parameter count.
- Far fewer parameters than a wide MLP, yet significantly higher accuracy.

**Training result:**

| Metric | Value |
|--------|-------|
| Best Val Accuracy | **100%** (epoch 21) |
| Saved checkpoint | `models/cnn_model.pth` |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Dropout | 0.25 |
| Batch size | 64 |
| Device | Apple MPS (Apple Silicon) |

Training command:

```bash
python -m src.train_cnn --epochs 30 --lr 1e-3
```

---

## Experiment Tracking (Weights & Biases)

All neural network runs are tracked on W&B:

- **Project:** `sign-language-mnist`
- **Tracked metrics:** `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `lr` (per epoch)
- **Summary:** `best_val_accuracy` logged at end of each run

To disable W&B logging:

```bash
python -m src.train_cnn --no_wandb
python -m src.train_mlp --no_wandb
```

---

## Web Application

Built with **Streamlit**. The app auto-loads the best available model (ConvNet preferred, MLP fallback).

### Features

| Feature | Details |
|---------|---------|
| Image upload | PNG, JPG, JPEG, WEBP, AVIF |
| Hand cropping | Otsu thresholding + contour detection (OpenCV) to auto-crop the hand region |
| Preprocessing | Greyscale → crop → resize 28×28 → normalise [0,1] |
| Prediction | On-demand via **Predict Sign** button |
| Result display | Predicted letter (9rem animated) + confidence score + progress bar |
| Audio | Browser Web Speech API — click **Hear "X"** to hear the letter spoken aloud |
| Top-5 bars | Vivid gradient bars showing top-5 class probabilities |
| Full distribution | Expandable table of all 24 class probabilities |
| Preprocessed view | Expandable 28×28 greyscale preview of what the model actually sees |
| Sidebar | Model info, supported letters grid, usage tips |

### UI Design

- **Dark aesthetic** — `#05050f` background with animated radial-gradient colour orbs
- **Glassmorphism** — `backdrop-filter: blur` cards and navbar
- **Neon glows** — drop-shadow on the predicted letter and speaker button
- **Animations** — `letterPop` (spring scale-in), `fadeUp` (result card), `barGrow` (confidence bar), `float` (idle placeholder), `bgPulse` (background orbs), `blink` (live indicator dot)
- **Two-column layout** — upload on left, result on right — no scrolling needed

### Image Tips for Best Results

- Use a **plain, contrasting background** (light background works best).
- Keep the hand **centred** and well-lit.
- The app auto-crops to the largest contour (your hand) using Otsu + morphological close.
- Note: J and Z are not supported (they require motion).

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd SignLanguage
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download from [Kaggle — Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and place inside the `data/` folder:

```
data/
├── sign_mnist_train.csv
└── sign_mnist_test.csv
```

### 3. (Optional) Quick demo without real data

```bash
python demo_setup.py
```

This generates synthetic data and trains a minimal CNN so you can launch the app immediately.

### 4. Train models

```bash
# Random Forest baseline
python -m src.train_baseline --n_estimators 200

# MLP
python -m src.train_mlp --epochs 30 --lr 1e-3 --no_wandb

# ConvNet (recommended)
python -m src.train_cnn --epochs 30 --lr 1e-3 --no_wandb
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Neural network training and inference |
| `scikit-learn` | Random Forest baseline |
| `streamlit` | Web application framework |
| `opencv-python` | Image decoding and hand-cropping |
| `Pillow` | PIL image handling |
| `wandb` | Experiment tracking |
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Confusion matrix visualisation |
| `joblib` | Saving/loading scikit-learn models |

Python **3.9+** required.

---

## Results Summary

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Random Forest (200 trees) | 82.50% | Baseline, no training loop |
| MLP [512, 256] | 84.16% | Flat pixel features |
| MLP [1024, 512, 256] | 83.98% | Deeper MLP |
| **ConvNet** | **100%** | **Best — used by app** |

---

## Branch

All development for this project lives on branch `aryalaa/code`.

```bash
git checkout aryalaa/code
```
