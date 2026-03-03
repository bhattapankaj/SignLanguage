"""
Streamlit deployment app for Sign Language translation.

Automatically loads the best available model:
  1. cnn_model.pth  (ConvNet — preferred, higher accuracy)
  2. mlp_model.pth  (MLP fallback)

Run:
    streamlit run app.py
"""

import io
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Global ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0a0a0f;
        }
        .block-container {
            padding-top: 5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        /* ── Navbar ── */
        .navbar {
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            z-index: 9999;
            background: rgba(10, 10, 15, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(139, 92, 246, 0.15);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2.5rem;
            height: 56px;
        }
        .navbar-brand {
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .navbar-pill {
            font-size: 0.7rem;
            font-weight: 600;
            color: #a78bfa;
            background: rgba(139, 92, 246, 0.12);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            letter-spacing: 0.05em;
        }

        /* ── Hero ── */
        .hero {
            text-align: center;
            padding: 2.5rem 1rem 2rem;
        }
        .hero-badge {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #a78bfa;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.25);
            border-radius: 20px;
            padding: 0.3rem 1rem;
            margin-bottom: 1.2rem;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.8rem;
        }
        .hero-sub {
            font-size: 1rem;
            color: #6b7280;
            max-width: 500px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ── Stat cards ── */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }
        .stat-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            transition: border-color 0.2s, background 0.2s;
        }
        .stat-card:hover {
            border-color: rgba(139, 92, 246, 0.3);
            background: rgba(139, 92, 246, 0.05);
        }
        .stat-icon {
            font-size: 1.4rem;
            margin-bottom: 0.4rem;
        }
        .stat-label {
            font-size: 0.62rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #4b5563;
            margin-bottom: 0.3rem;
        }
        .stat-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #e5e7eb;
        }
        .stat-value.accent { color: #a78bfa; }

        /* ── Upload zone ── */
        .upload-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .upload-hint {
            font-size: 0.8rem;
            color: #374151;
            margin-top: 0.5rem;
        }

        /* ── Panel (image + result boxes) ── */
        .panel {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 1.5rem;
            height: 100%;
        }
        .panel-label {
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #4b5563;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .panel-caption {
            font-size: 0.65rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #374151;
            margin-top: 0.6rem;
            text-align: center;
        }

        /* ── Result ── */
        .result-wrapper {
            background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(96,165,250,0.05));
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 16px;
            padding: 2rem 1.5rem;
            text-align: center;
        }
        .result-letter {
            font-size: 8rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .result-label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #6b7280;
            margin-top: 0.2rem;
        }
        .confidence-pct {
            font-size: 2rem;
            font-weight: 700;
            color: #a78bfa;
            margin-top: 1rem;
        }
        .confidence-label {
            font-size: 0.65rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #4b5563;
        }
        .conf-bar-bg {
            background: rgba(255,255,255,0.05);
            border-radius: 99px;
            height: 6px;
            width: 100%;
            margin-top: 0.6rem;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 6px;
            border-radius: 99px;
            background: linear-gradient(90deg, #7c3aed, #a78bfa, #60a5fa);
            transition: width 0.6s ease;
        }

        /* ── Top-5 bars ── */
        .top5-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.5rem;
        }
        .top5-letter {
            font-size: 0.85rem;
            font-weight: 700;
            color: #e5e7eb;
            width: 1.2rem;
            text-align: center;
        }
        .top5-bar-bg {
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 99px;
            height: 5px;
            overflow: hidden;
        }
        .top5-bar-fill {
            height: 5px;
            border-radius: 99px;
        }
        .top5-pct {
            font-size: 0.72rem;
            color: #6b7280;
            width: 3.5rem;
            text-align: right;
        }
        .top5-winner .top5-letter { color: #a78bfa; }
        .top5-winner .top5-pct    { color: #a78bfa; }

        /* ── Section header ── */
        .section-header {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #4b5563;
            margin-bottom: 0.8rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        /* ── Warning banner ── */
        .warn-banner {
            background: rgba(234,179,8,0.08);
            border: 1px solid rgba(234,179,8,0.2);
            border-radius: 10px;
            padding: 0.7rem 1rem;
            font-size: 0.78rem;
            color: #ca8a04;
            margin-bottom: 1.5rem;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #0a0a0f !important;
            border-right: 1px solid rgba(255,255,255,0.05) !important;
        }
        .sidebar-title {
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #a78bfa;
            margin-bottom: 1rem;
        }
        .sidebar-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .sidebar-key {
            font-size: 0.72rem;
            color: #4b5563;
        }
        .sidebar-val {
            font-size: 0.72rem;
            font-weight: 600;
            color: #9ca3af;
        }
        .letter-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 0.3rem;
            margin-top: 0.6rem;
        }
        .letter-chip {
            background: rgba(139,92,246,0.08);
            border: 1px solid rgba(139,92,246,0.15);
            border-radius: 6px;
            text-align: center;
            padding: 0.25rem 0;
            font-size: 0.75rem;
            font-weight: 600;
            color: #7c3aed;
        }

        /* ── Divider ── */
        hr { border-color: rgba(255,255,255,0.05) !important; }

        /* Streamlit file uploader tweaks */
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.02) !important;
            border: 1.5px dashed rgba(139,92,246,0.3) !important;
            border-radius: 14px !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(139,92,246,0.6) !important;
            background: rgba(139,92,246,0.04) !important;
        }

        /* ── Predict button ── */
        div[data-testid="stButton"] > button {
            width: 100%;
            padding: 0.85rem 2rem;
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: #ffffff;
            background: linear-gradient(135deg, #7c3aed, #6d28d9);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s, box-shadow 0.2s;
            box-shadow: 0 4px 24px rgba(124,58,237,0.35);
        }
        div[data-testid="stButton"] > button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 6px 32px rgba(124,58,237,0.5);
        }
        div[data-testid="stButton"] > button:active {
            transform: translateY(0px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def navbar(model_label: str) -> None:
    st.markdown(
        f"""
        <div class="navbar">
            <span class="navbar-brand">✦ Sign Language AI</span>
            <span class="navbar-pill">{model_label} · 24 classes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ model loading

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if os.path.exists(CNN_PATH):
        path, model_label = CNN_PATH, "ConvNet"
    elif os.path.exists(MLP_PATH):
        path, model_label = MLP_PATH, "MLP"
    else:
        return None, None, None

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type  = checkpoint.get("model_type", "mlp")
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


def preprocess_image(uploaded_file, model_type: str):
    raw = uploaded_file.read()
    file_bytes = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        try:
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError("Could not decode image. Please upload PNG, JPG, WEBP, or AVIF.")

    img_grey   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_grey   = _crop_hand(img_grey)
    img_resized = cv2.resize(img_grey, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_norm   = img_resized.astype(np.float32) / 255.0

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


# ------------------------------------------------------------------ UI helpers

def _top5_bars(probs: list) -> None:
    top5 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    max_p = probs[top5[0]]
    colors = ["#a78bfa", "#818cf8", "#60a5fa", "#38bdf8", "#34d399"]
    html = ""
    for rank, idx in enumerate(top5):
        letter = LABEL_TO_LETTER[idx]
        pct    = probs[idx] * 100
        bar_w  = int(pct / max_p * 100) if max_p > 0 else 0
        winner = "top5-winner" if rank == 0 else ""
        html += f"""
        <div class="top5-row {winner}">
            <span class="top5-letter">{letter}</span>
            <div class="top5-bar-bg">
                <div class="top5-bar-fill" style="width:{bar_w}%;background:{colors[rank]};"></div>
            </div>
            <span class="top5-pct">{pct:.1f}%</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_sidebar(model_label: str, best_acc) -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Model Info</div>", unsafe_allow_html=True)
        rows = [
            ("Dataset",    "Sign Language MNIST"),
            ("Model",      model_label or "—"),
            ("Accuracy",   f"{best_acc*100:.1f}%" if best_acc else "N/A"),
            ("Classes",    "24  (A–Y, excl. J & Z)"),
            ("Input",      "28 × 28 greyscale"),
            ("Framework",  "PyTorch"),
        ]
        html = ""
        for k, v in rows:
            html += f"""<div class='sidebar-row'>
                <span class='sidebar-key'>{k}</span>
                <span class='sidebar-val'>{v}</span>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-title'>Supported Letters</div>", unsafe_allow_html=True)
        chips = "".join(f"<div class='letter-chip'>{l}</div>" for l in ALL_LETTERS)
        st.markdown(f"<div class='letter-grid'>{chips}</div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.62rem;color:#374151;line-height:1.6;'>"
            "For best results: plain background, hand centered, good lighting."
            "</p>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(
        page_title="Sign Language AI",
        page_icon="✦",
        layout="wide",
    )

    inject_css()
    model, device, model_label = load_model()
    navbar(model_label or "No model")

    if model is None:
        st.error("No trained model found. Run `python -m src.train_cnn` first.")
        st.stop()

    ckpt_path = CNN_PATH if model_label == "ConvNet" else MLP_PATH
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_epoch = ckpt.get("epoch", "—")
    best_acc   = ckpt.get("val_accuracy", None)
    input_fmt  = "1 × 28 × 28" if model_label == "ConvNet" else "784-dim vector"

    # ── Hero ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">✦ AI-Powered Sign Recognition</div>
            <div class="hero-title">Read the Signs</div>
            <p class="hero-sub">Upload a photo of an ASL hand sign and get an instant letter prediction powered by a trained ConvNet.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stats row ────────────────────────────────────────────────────────
    icons  = ["🧠", "🏆", "📐", "⚡"]
    labels = ["Architecture", "Best Epoch", "Val Accuracy", "Input Format"]
    values = [model_label, str(best_epoch),
              f"{best_acc*100:.1f}%" if best_acc else "N/A", input_fmt]
    accents = [False, False, True, False]

    cols = st.columns(4)
    for col, icon, label, value, is_accent in zip(cols, icons, labels, values, accents):
        accent_cls = "accent" if is_accent else ""
        col.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-icon'>{icon}</div>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value {accent_cls}'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if model_label == "MLP":
        st.markdown(
            "<div class='warn-banner'>⚠ MLP model active — train the CNN for higher accuracy: "
            "<code>python -m src.train_cnn</code></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='upload-label'>📁 &nbsp;Drop your hand sign image</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        label="upload",
        type=["png", "jpg", "jpeg", "webp", "avif"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            "<p class='upload-hint'>PNG · JPG · WEBP · AVIF — hand sign against a plain background</p>",
            unsafe_allow_html=True,
        )
        _render_sidebar(model_label, best_acc)
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Preview + button (always shown after upload) ──────────────────────
    prev_col, btn_col = st.columns([3, 1], gap="medium")
    with prev_col:
        st.markdown("<div class='panel-label'>📷 &nbsp;Image ready — click Predict to analyse</div>",
                    unsafe_allow_html=True)
        st.image(Image.open(uploaded), width=220)

    with btn_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run = st.button("✦  Predict Sign", use_container_width=True)

    if not run:
        _render_sidebar(model_label, best_acc)
        return

    # ── Run prediction ────────────────────────────────────────────────────
    uploaded.seek(0)
    try:
        tensor, img_28 = preprocess_image(uploaded, model_label)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    letter, confidence, probs = predict(model, tensor, device)
    bar_w = int(confidence * 100)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Three-column result layout ────────────────────────────────────────
    col_orig, col_pre, col_res = st.columns([2, 2, 3], gap="medium")

    with col_orig:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>📷 &nbsp;Original</div>", unsafe_allow_html=True)
        uploaded.seek(0)
        st.image(Image.open(uploaded), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pre:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>🔬 &nbsp;Preprocessed — 28 × 28</div>", unsafe_allow_html=True)
        st.image(img_28, clamp=True, use_container_width=True)
        st.markdown(
            "<div class='panel-caption'>Greyscale · Auto-cropped · Normalised</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        st.markdown(
            f"""
            <div class="result-wrapper">
                <div class="result-label">Predicted Letter</div>
                <div class="result-letter">{letter}</div>
                <div class="confidence-pct">{confidence*100:.1f}%</div>
                <div class="confidence-label">Confidence</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{bar_w}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Top 5 Predictions</div>", unsafe_allow_html=True)
        _top5_bars(probs)

    # ── Full distribution ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View full probability distribution"):
        all_df = pd.DataFrame({
            "Letter": ALL_LETTERS,
            "Probability (%)": [round(p * 100, 3) for p in probs],
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        st.dataframe(all_df, hide_index=True, use_container_width=True)

    _render_sidebar(model_label, best_acc)


if __name__ == "__main__":
    main()
