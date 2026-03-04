import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import LABEL_TO_LETTER, NUM_CLASSES, IMAGE_SIZE
from src.model import MLP, ConvNet


CNN_PATH = os.path.join("models", "cnn_model.pth")
MLP_PATH = os.path.join("models", "mlp_model.pth")
ALL_LETTERS = list(LABEL_TO_LETTER.values())


# ------------------------------------------------------------------ styling

def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Global font & background ── */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* ── Top navbar ── */
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 9999;
            background: #0e0e0e;
            border-bottom: 1px solid #222;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            height: 52px;
        }
        .navbar-brand {
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: #ffffff;
            text-transform: uppercase;
        }
        .navbar-links {
            display: flex;
            gap: 2rem;
        }
        .navbar-links a {
            font-size: 0.78rem;
            font-weight: 500;
            color: #888;
            text-decoration: none;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            transition: color 0.2s;
        }
        .navbar-links a:hover { color: #fff; }
        .navbar-badge {
            font-size: 0.7rem;
            font-weight: 500;
            color: #555;
            letter-spacing: 0.04em;
        }

        /* ── Push content below navbar ── */
        .block-container { padding-top: 4.5rem !important; }

        /* ── Section label ── */
        .section-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 0.4rem;
        }

        /* ── Stat card ── */
        .stat-card {
            background: #141414;
            border: 1px solid #1f1f1f;
            border-radius: 6px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .stat-label {
            font-size: 0.65rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 0.3rem;
        }
        .stat-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e8e8e8;
        }

        /* ── Prediction result ── */
        .result-letter {
            font-size: 7rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
            letter-spacing: -0.02em;
        }
        .result-meta {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #555;
            margin-top: 0.5rem;
        }
        .confidence-bar-bg {
            background: #1a1a1a;
            border-radius: 3px;
            height: 4px;
            width: 100%;
            margin-top: 0.6rem;
        }
        .confidence-bar-fill {
            background: #ffffff;
            border-radius: 3px;
            height: 4px;
        }

        /* ── Divider ── */
        hr { border-color: #1f1f1f !important; }

        /* ── Warning banner ── */
        .warn-banner {
            background: #1a1500;
            border: 1px solid #332800;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            font-size: 0.78rem;
            color: #a08a3a;
            margin-bottom: 1rem;
        }

        /* ── Image caption ── */
        .img-caption {
            font-size: 0.65rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #444;
            margin-top: 0.4rem;
            text-align: center;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #0e0e0e;
            border-right: 1px solid #1f1f1f;
        }
        [data-testid="stSidebar"] * { color: #777 !important; }
        [data-testid="stSidebar"] strong { color: #aaa !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def navbar(model_label: str) -> None:
    st.markdown(
        f"""
        <div class="navbar">
            <span class="navbar-brand">Sign Language MNIST</span>
            <div class="navbar-links">
                <a href="#predict">Predict</a>
                <a href="#about">About</a>
            </div>
            <span class="navbar-badge">{model_label} &nbsp;|&nbsp; A-Y excl. J</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ model loading

@st.cache_resource(show_spinner="Loading model...")
def load_model() -> tuple[nn.Module, torch.device, str]:
    if os.path.exists(CNN_PATH):
        path, model_label = CNN_PATH, "ConvNet"
    elif os.path.exists(MLP_PATH):
        path, model_label = MLP_PATH, "MLP"
    else:
        return None, None, None

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "mlp")
    dropout     = checkpoint.get("dropout", 0.25)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    if model_type == "cnn":
        model = ConvNet(num_classes=num_classes, dropout=dropout)
    else:
        hidden_units = checkpoint.get("hidden_units", [512, 256])
        model = MLP(input_size=IMAGE_SIZE * IMAGE_SIZE,
                    hidden_units=hidden_units,
                    num_classes=num_classes,
                    dropout=dropout)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch.device("cpu"), model_label


# ------------------------------------------------------------------ preprocessing

def _crop_hand(img_grey: np.ndarray, pad_frac: float = 0.15) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_grey, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_grey
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    H, W = img_grey.shape
    x1 = max(0, x - int(w * pad_frac))
    y1 = max(0, y - int(h * pad_frac))
    x2 = min(W, x + w + int(w * pad_frac))
    y2 = min(H, y + h + int(h * pad_frac))
    cropped = img_grey[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else img_grey


def preprocess_image(uploaded_file, model_type: str) -> tuple[torch.Tensor, np.ndarray]:
    raw = uploaded_file.read()

    # Try OpenCV first; fall back to Pillow for formats like AVIF/WEBP
    # that older OpenCV builds may not support
    file_bytes = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        try:
            import io
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(
                "Could not decode the image. "
                "Please upload a valid PNG, JPG, WEBP, or AVIF file."
            )
    img_grey  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_grey  = _crop_hand(img_grey)
    img_resized = cv2.resize(img_grey, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_norm  = img_resized.astype(np.float32) / 255.0
    if model_type == "ConvNet":
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(img_norm).view(1, -1)
    return tensor, img_resized


def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1).squeeze()
    idx = probs.argmax().item()
    return LABEL_TO_LETTER[idx], probs[idx].item(), probs.tolist()


# ------------------------------------------------------------------ UI

def main() -> None:
    st.set_page_config(
        page_title="Sign Language MNIST",
        page_icon=None,
        layout="wide",
    )

    inject_css()

    model, device, model_label = load_model()

    # Navbar (needs model_label; show placeholder if model missing)
    navbar(model_label or "No model")

    if model is None:
        st.error(
            "No trained model found. "
            "Run `python -m src.train_cnn` to train the CNN model."
        )
        st.stop()

    ckpt_path = CNN_PATH if model_label == "ConvNet" else MLP_PATH
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_epoch = ckpt.get("epoch", "—")
    best_acc   = ckpt.get("val_accuracy", None)
    input_fmt  = "1 x 28 x 28" if model_label == "ConvNet" else "784-dim vector"

    # ── Stats row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Architecture", model_label),
        ("Best Epoch",   str(best_epoch)),
        ("Val Accuracy", f"{best_acc * 100:.1f}%" if best_acc else "N/A"),
        ("Input Format", input_fmt),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if model_label == "MLP":
        st.markdown(
            "<div class='warn-banner'>"
            "MLP model active. Train the CNN for higher accuracy: "
            "<code>python -m src.train_cnn</code>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-label' id='predict'>Upload Image</div>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="upload",
        type=["png", "jpg", "jpeg", "webp", "avif"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            "<p style='color:#444; font-size:0.82rem; margin-top:0.5rem;'>"
            "Accepted formats: PNG, JPG, JPEG, WEBP, AVIF — hand sign against a plain background."
            "</p>",
            unsafe_allow_html=True,
        )
        _render_sidebar(model_label)
        return

    # ── Image columns ──────────────────────────────────────────────────
    col_img, col_pre, col_res = st.columns([2, 2, 3])

    with col_img:
        st.markdown("<div class='section-label'>Original</div>",
                    unsafe_allow_html=True)
        st.image(Image.open(uploaded), use_container_width=True)

    uploaded.seek(0)
    try:
        tensor, img_28 = preprocess_image(uploaded, model_label)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with col_pre:
        st.markdown("<div class='section-label'>Preprocessed — 28 x 28</div>",
                    unsafe_allow_html=True)
        st.image(img_28, clamp=True, use_container_width=True)
        st.markdown("<div class='img-caption'>Greyscale · Auto-cropped · Normalised</div>",
                    unsafe_allow_html=True)

    letter, confidence, probs = predict(model, tensor, device)
    bar_width = int(confidence * 100)

    with col_res:
        st.markdown("<div class='section-label'>Prediction</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-letter'>{letter}</div>"
            f"<div class='result-meta'>{confidence * 100:.2f}% confidence</div>"
            f"<div class='confidence-bar-bg'>"
            f"  <div class='confidence-bar-fill' style='width:{bar_width}%'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Top-5 table
        st.markdown("<div class='section-label'>Top 5</div>", unsafe_allow_html=True)
        top5 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
        top5_df = pd.DataFrame({
            "Letter": [LABEL_TO_LETTER[i] for i in top5],
            "Probability": [f"{probs[i]*100:.2f}%" for i in top5],
        })
        st.dataframe(top5_df, hide_index=True, use_container_width=True)

    # ── Full distribution (collapsed) ─────────────────────────────────
    with st.expander("Full probability distribution"):
        all_df = pd.DataFrame({
            "Letter": ALL_LETTERS,
            "Probability (%)": [round(p * 100, 3) for p in probs],
        }).sort_values("Probability (%)", ascending=False)
        st.dataframe(all_df, hide_index=True, use_container_width=True)

    _render_sidebar(model_label)


def _render_sidebar(model_label: str) -> None:
    with st.sidebar:
        st.markdown(
            "<p style='font-size:0.65rem; letter-spacing:0.1em; "
            "text-transform:uppercase; color:#444; margin-bottom:1rem;'>About</p>",
            unsafe_allow_html=True,
        )
        rows = [
            ("Dataset",    "Sign Language MNIST"),
            ("Model",      model_label or "—"),
            ("Classes",    "24  (A-Y, excl. J & Z)"),
            ("Resolution", "28 x 28 greyscale"),
            ("Framework",  "PyTorch"),
        ]
        for key, val in rows:
            st.markdown(
                f"<p style='font-size:0.75rem; margin:0.25rem 0;'>"
                f"<strong style='color:#555 !important;'>{key}</strong>"
                f"<span style='color:#444 !important;'> — {val}</span></p>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='border-color:#1f1f1f; margin:1.2rem 0;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.65rem; letter-spacing:0.1em; "
            "text-transform:uppercase; color:#444; margin-bottom:0.8rem;'>Label Map</p>",
            unsafe_allow_html=True,
        )
        for idx, letter in LABEL_TO_LETTER.items():
            st.markdown(
                f"<p style='font-size:0.72rem; margin:0.15rem 0; color:#3a3a3a !important;'>"
                f"{letter} &rarr; class {idx}</p>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
