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
import streamlit.components.v1 as components
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import LABEL_TO_LETTER, NUM_CLASSES, IMAGE_SIZE
from src.model import MLP, ConvNet


CNN_PATH = os.path.join("models", "cnn_model.pth")
MLP_PATH = os.path.join("models", "mlp_model.pth")
ALL_LETTERS = list(LABEL_TO_LETTER.values())


# ─────────────────────────────────────────────────────────── CSS
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── reset / base ── */
    #MainMenu, footer, header { visibility: hidden; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── animated background ── */
    .stApp {
        background: #05050f;
        overflow-x: hidden;
    }
    .stApp::before {
        content: '';
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background:
            radial-gradient(ellipse 60% 50% at 15% 20%,  rgba(124,58,237,0.18) 0%, transparent 70%),
            radial-gradient(ellipse 50% 40% at 85% 75%,  rgba(6,182,212,0.14)  0%, transparent 70%),
            radial-gradient(ellipse 40% 35% at 60% 10%,  rgba(236,72,153,0.10) 0%, transparent 70%),
            radial-gradient(ellipse 55% 45% at 30% 80%,  rgba(16,185,129,0.08) 0%, transparent 70%);
        animation: bgPulse 12s ease-in-out infinite alternate;
    }
    @keyframes bgPulse {
        0%   { opacity: 0.7; transform: scale(1); }
        100% { opacity: 1;   transform: scale(1.06); }
    }

    .block-container {
        position: relative; z-index: 1;
        padding-top: 4.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1260px;
    }

    /* ── navbar ── */
    .navbar {
        position: fixed; top: 0; left: 0;
        width: 100%; z-index: 9999;
        background: rgba(5,5,15,0.75);
        backdrop-filter: blur(24px) saturate(180%);
        border-bottom: 1px solid rgba(139,92,246,0.2);
        display: flex; align-items: center;
        justify-content: space-between;
        padding: 0 2.5rem; height: 56px;
    }
    .navbar-brand {
        font-size: 1.05rem; font-weight: 900; letter-spacing: -0.01em;
        background: linear-gradient(90deg, #f0abfc, #a78bfa, #67e8f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .navbar-right { display: flex; align-items: center; gap: 0.7rem; }
    .navbar-pill {
        font-size: 0.68rem; font-weight: 600; color: #a78bfa;
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 99px; padding: 0.25rem 0.85rem;
    }
    .navbar-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981, 0 0 16px rgba(16,185,129,0.5);
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

    /* ── hero ── */
    .hero-wrap {
        padding: 1.4rem 0 1.2rem;
        display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap;
    }
    .hero-badge {
        font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #f0abfc;
        background: rgba(240,171,252,0.1);
        border: 1px solid rgba(240,171,252,0.25);
        border-radius: 99px; padding: 0.28rem 0.9rem;
        white-space: nowrap;
    }
    .hero-title {
        font-size: 2rem; font-weight: 900; line-height: 1;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #ffffff 0%, #f0abfc 40%, #67e8f9 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub { font-size: 0.78rem; color: #6b7280; margin: 0; }

    /* ── glass card mixin ── */
    .glass {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px) saturate(160%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }

    /* ── upload zone ── */
    .upload-label {
        font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #6b7280; margin-bottom: 0.6rem;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02) !important;
        border: 2px dashed rgba(167,139,250,0.35) !important;
        border-radius: 16px !important;
        transition: border-color 0.3s, background 0.3s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(167,139,250,0.75) !important;
        background: rgba(167,139,250,0.05) !important;
    }
    .upload-hint {
        font-size: 0.73rem; color: #374151; margin-top: 0.4rem;
    }

    /* ── preview card ── */
    .preview-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 0.9rem;
        margin-top: 0.9rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .preview-tag {
        display: inline-block;
        font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #10b981;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 99px; padding: 0.18rem 0.65rem;
        margin-bottom: 0.55rem;
    }

    /* ── predict button ── */
    div[data-testid="stButton"] > button {
        width: 100%; padding: 0.82rem 1.5rem;
        font-size: 0.92rem; font-weight: 800;
        letter-spacing: 0.04em; color: #fff;
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
        background-size: 200% 200%;
        border: none; border-radius: 14px; cursor: pointer;
        transition: transform 0.15s, box-shadow 0.25s, background-position 0.4s;
        box-shadow: 0 4px 24px rgba(168,85,247,0.45), 0 1px 0 rgba(255,255,255,0.1) inset;
        animation: gradShift 4s ease infinite;
    }
    @keyframes gradShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 36px rgba(168,85,247,0.65), 0 1px 0 rgba(255,255,255,0.15) inset;
    }
    div[data-testid="stButton"] > button:active { transform: translateY(0); }

    /* ── placeholder ── */
    .placeholder {
        min-height: 360px;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        border: 2px dashed rgba(255,255,255,0.06);
        border-radius: 24px; text-align: center;
        background: rgba(255,255,255,0.015);
    }
    .placeholder-emoji { font-size: 3rem; margin-bottom: 0.8rem;
        animation: float 3s ease-in-out infinite; }
    @keyframes float {
        0%,100% { transform: translateY(0); }
        50%      { transform: translateY(-8px); }
    }
    .placeholder-title {
        font-size: 1rem; font-weight: 700; color: #374151;
        margin-bottom: 0.3rem;
    }
    .placeholder-sub { font-size: 0.75rem; color: #1f2937; }

    /* ── result card ── */
    .result-card {
        background: linear-gradient(145deg,
            rgba(124,58,237,0.15) 0%,
            rgba(168,85,247,0.08) 40%,
            rgba(6,182,212,0.08) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 24px; padding: 2rem 1.8rem;
        box-shadow:
            0 0 0 1px rgba(167,139,250,0.1),
            0 16px 48px rgba(0,0,0,0.4),
            inset 0 1px 0 rgba(255,255,255,0.06);
        animation: fadeUp 0.4s cubic-bezier(0.16,1,0.3,1);
    }
    @keyframes fadeUp {
        from { opacity:0; transform: translateY(20px); }
        to   { opacity:1; transform: translateY(0); }
    }
    .result-eyebrow {
        font-size: 0.6rem; font-weight: 700; letter-spacing: 0.16em;
        text-transform: uppercase; color: #6b7280;
    }
    .result-letter {
        font-size: 9rem; font-weight: 900; line-height: 0.9;
        letter-spacing: -0.05em; margin: 0.1rem 0 0.2rem;
        background: linear-gradient(135deg, #ffffff 0%, #e879f9 45%, #67e8f9 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 30px rgba(232,121,249,0.4));
        animation: letterPop 0.5s cubic-bezier(0.34,1.56,0.64,1);
    }
    @keyframes letterPop {
        from { transform: scale(0.6); opacity: 0; }
        to   { transform: scale(1);   opacity: 1; }
    }
    .result-conf-num {
        font-size: 2.2rem; font-weight: 800; line-height: 1;
        background: linear-gradient(90deg, #a78bfa, #67e8f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .result-conf-label {
        font-size: 0.58rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #4b5563;
    }
    .conf-track {
        background: rgba(255,255,255,0.06);
        border-radius: 99px; height: 7px;
        overflow: hidden; margin-top: 0.6rem;
    }
    .conf-fill {
        height: 7px; border-radius: 99px;
        background: linear-gradient(90deg, #7c3aed, #a855f7, #06b6d4);
        box-shadow: 0 0 12px rgba(168,85,247,0.6);
        animation: barGrow 0.7s cubic-bezier(0.16,1,0.3,1);
        transform-origin: left;
    }
    @keyframes barGrow { from{transform:scaleX(0)} to{transform:scaleX(1)} }

    /* ── section header ── */
    .section-hdr {
        font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #374151;
        margin: 1.1rem 0 0.55rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .section-hdr::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(255,255,255,0.07), transparent);
    }

    /* ── top-5 bars ── */
    .t5-row {
        display: flex; align-items: center; gap: 0.7rem;
        margin-bottom: 0.45rem;
    }
    .t5-letter {
        font-size: 0.82rem; font-weight: 800;
        color: #9ca3af; width: 1.1rem; text-align: center;
    }
    .t5-track {
        flex: 1; background: rgba(255,255,255,0.05);
        border-radius: 99px; height: 5px; overflow: hidden;
    }
    .t5-fill { height: 5px; border-radius: 99px; }
    .t5-pct { font-size: 0.7rem; color: #6b7280; width: 3rem; text-align: right; }
    .t5-winner .t5-letter { color: #e879f9; font-size: 0.9rem; }
    .t5-winner .t5-pct    { color: #a78bfa; font-weight: 700; }

    /* ── warn ── */
    .warn {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.2);
        border-radius: 12px; padding: 0.55rem 1rem;
        font-size: 0.75rem; color: #d97706;
        margin-bottom: 1rem;
    }

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(5,5,15,0.9) !important;
        border-right: 1px solid rgba(139,92,246,0.12) !important;
    }
    .sb-title {
        font-size: 0.58rem; font-weight: 800; letter-spacing: 0.16em;
        text-transform: uppercase;
        background: linear-gradient(90deg,#a78bfa,#67e8f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.8rem;
    }
    .sb-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.38rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .sb-key { font-size: 0.7rem; color: #374151; }
    .sb-val { font-size: 0.7rem; font-weight: 700; color: #9ca3af; }
    .chip-grid {
        display: grid; grid-template-columns: repeat(6,1fr);
        gap: 0.28rem; margin-top: 0.55rem;
    }
    .chip {
        background: linear-gradient(135deg,rgba(124,58,237,0.15),rgba(6,182,212,0.1));
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 7px; text-align: center;
        padding: 0.24rem 0; font-size: 0.72rem;
        font-weight: 700; color: #a78bfa;
        transition: border-color 0.2s, background 0.2s;
    }
    .chip:hover {
        border-color: rgba(167,139,250,0.5);
        background: rgba(167,139,250,0.2);
    }
    hr { border-color: rgba(255,255,255,0.05) !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────── navbar
def navbar(model_label: str) -> None:
    st.markdown(f"""
    <div class="navbar">
        <span class="navbar-brand">✦ SignLang AI</span>
        <div class="navbar-right">
            <span class="navbar-pill">{model_label} · 24 classes</span>
            <div class="navbar-dot" title="Model loaded"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────── model
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if os.path.exists(CNN_PATH):
        path, label = CNN_PATH, "ConvNet"
    elif os.path.exists(MLP_PATH):
        path, label = MLP_PATH, "MLP"
    else:
        return None, None, None

    ckpt        = torch.load(path, map_location="cpu", weights_only=False)
    model_type  = ckpt.get("model_type", "mlp")
    dropout     = ckpt.get("dropout", 0.25)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)

    if model_type == "cnn":
        model = ConvNet(num_classes=num_classes, dropout=dropout)
    else:
        model = MLP(input_size=IMAGE_SIZE * IMAGE_SIZE,
                    hidden_units=ckpt.get("hidden_units", [512, 256]),
                    num_classes=num_classes, dropout=dropout)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, torch.device("cpu"), label


# ─────────────────────────────────────────────────────────────── image
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
    cropped = img_grey[
        max(0, y - int(h * pad_frac)): min(H, y + h + int(h * pad_frac)),
        max(0, x - int(w * pad_frac)): min(W, x + w + int(w * pad_frac)),
    ]
    return cropped if cropped.size > 0 else img_grey


def preprocess_image(raw: bytes, model_type: str):
    file_bytes = np.frombuffer(raw, dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        pil    = Image.open(io.BytesIO(raw)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img_grey    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_grey    = _crop_hand(img_grey)
    img_resized = cv2.resize(img_grey, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_norm    = img_resized.astype(np.float32) / 255.0
    if model_type == "ConvNet":
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(img_norm).view(1, -1)
    return tensor, img_resized


def run_predict(model: nn.Module, tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1).squeeze()
    idx = probs.argmax().item()
    return LABEL_TO_LETTER[idx], probs[idx].item(), probs.tolist()


# ─────────────────────────────────────────────────────────────── UI helpers
def speaker_button(letter: str) -> None:
    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap');
    body {{ margin:0; padding:0; background:transparent; }}
    .spk {{
        display: inline-flex; align-items: center; gap: 0.55rem;
        padding: 0.55rem 1.3rem;
        font-size: 0.85rem; font-weight: 700;
        font-family: 'Inter', sans-serif;
        color: #e879f9; cursor: pointer;
        background: linear-gradient(135deg,rgba(232,121,249,0.12),rgba(103,232,249,0.08));
        border: 1px solid rgba(232,121,249,0.35);
        border-radius: 12px;
        box-shadow: 0 0 16px rgba(232,121,249,0.15);
        transition: all 0.2s;
    }}
    .spk:hover {{
        background: linear-gradient(135deg,rgba(232,121,249,0.22),rgba(103,232,249,0.15));
        border-color: rgba(232,121,249,0.65);
        box-shadow: 0 0 24px rgba(232,121,249,0.35);
        transform: translateY(-2px);
    }}
    .spk:active {{ transform: translateY(0); }}
    </style>
    <button class="spk" onclick="
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance('{letter}');
        u.rate=0.85; u.pitch=1.0; u.volume=1.0;
        window.speechSynthesis.speak(u);
    ">🔊 &nbsp;Hear &ldquo;{letter}&rdquo;</button>
    """, height=54)


def top5_bars(probs: list) -> None:
    top5   = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    max_p  = probs[top5[0]]
    # vivid gradient stops for each rank
    fills  = [
        "linear-gradient(90deg,#7c3aed,#e879f9)",
        "linear-gradient(90deg,#2563eb,#06b6d4)",
        "linear-gradient(90deg,#059669,#34d399)",
        "linear-gradient(90deg,#d97706,#fbbf24)",
        "linear-gradient(90deg,#dc2626,#f87171)",
    ]
    html = ""
    for rank, idx in enumerate(top5):
        letter = LABEL_TO_LETTER[idx]
        pct    = probs[idx] * 100
        bar_w  = int(pct / max_p * 100) if max_p > 0 else 0
        winner = "t5-winner" if rank == 0 else ""
        html += f"""
        <div class="t5-row {winner}">
            <span class="t5-letter">{letter}</span>
            <div class="t5-track">
                <div class="t5-fill" style="width:{bar_w}%;background:{fills[rank]};"></div>
            </div>
            <span class="t5-pct">{pct:.1f}%</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar(model_label: str, best_acc) -> None:
    with st.sidebar:
        st.markdown("<div class='sb-title'>Model Info</div>", unsafe_allow_html=True)
        rows = [
            ("Dataset",   "Sign Language MNIST"),
            ("Model",     model_label or "—"),
            ("Accuracy",  f"{best_acc*100:.1f}%" if best_acc else "N/A"),
            ("Classes",   "24  (A–Y, excl. J & Z)"),
            ("Input",     "28 × 28 greyscale"),
            ("Framework", "PyTorch"),
        ]
        html = "".join(
            f"<div class='sb-row'><span class='sb-key'>{k}</span>"
            f"<span class='sb-val'>{v}</span></div>"
            for k, v in rows
        )
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sb-title'>Supported Letters</div>", unsafe_allow_html=True)
        chips = "".join(f"<div class='chip'>{l}</div>" for l in ALL_LETTERS)
        st.markdown(f"<div class='chip-grid'>{chips}</div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.62rem;color:#1f2937;line-height:1.7;'>"
            "💡 Best results with a plain background, hand centred in frame, and good lighting."
            "</p>", unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────── main
def main() -> None:
    st.set_page_config(page_title="SignLang AI", page_icon="🤟", layout="wide")
    inject_css()

    model, device, model_label = load_model()
    navbar(model_label or "No model")

    if model is None:
        st.error("No trained model found. Run `python -m src.train_cnn` first.")
        st.stop()

    ckpt_path = CNN_PATH if model_label == "ConvNet" else MLP_PATH
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_acc  = ckpt.get("val_accuracy", None)

    # ── hero ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">✦ ASL Recognition</div>
        <div class="hero-title">Read the Signs</div>
        <p class="hero-sub">Upload a hand sign → Predict → hear the letter spoken aloud.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_label == "MLP":
        st.markdown(
            "<div class='warn'>⚠ MLP active — train the CNN for best accuracy: "
            "<code>python -m src.train_cnn</code></div>",
            unsafe_allow_html=True,
        )

    # ── two-column layout ─────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: upload ─────────────────────────────────────────────────────
    with left:
        st.markdown("<div class='upload-label'>📁 &nbsp;Drop your hand sign image</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "upload", type=["png", "jpg", "jpeg", "webp", "avif"],
            label_visibility="collapsed",
        )

        if uploaded is None:
            st.markdown(
                "<p class='upload-hint'>PNG · JPG · WEBP · AVIF — hand sign on a plain background</p>",
                unsafe_allow_html=True,
            )
        else:
            raw = uploaded.read()

            st.markdown("<div class='preview-card'>", unsafe_allow_html=True)
            st.markdown("<span class='preview-tag'>✓ Image loaded</span>", unsafe_allow_html=True)
            st.image(Image.open(io.BytesIO(raw)), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🤟  Predict Sign", use_container_width=True):
                try:
                    tensor, img_28 = preprocess_image(raw, model_label)
                    letter, conf, probs = run_predict(model, tensor, device)
                    st.session_state["result"] = {
                        "letter": letter, "conf": conf,
                        "probs": probs, "img_28": img_28,
                    }
                except ValueError as e:
                    st.error(str(e))

    # ── RIGHT: result ─────────────────────────────────────────────────────
    with right:
        res = st.session_state.get("result")

        if res is None:
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-emoji">🤟</div>
                <div class="placeholder-title">Waiting for a prediction</div>
                <div class="placeholder-sub">Upload an image and hit Predict</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            letter = res["letter"]
            conf   = res["conf"]
            probs  = res["probs"]
            img_28 = res["img_28"]
            bar_w  = int(conf * 100)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-eyebrow">Predicted Letter</div>
                <div class="result-letter">{letter}</div>
                <div class="result-conf-num">{conf*100:.1f}%</div>
                <div class="result-conf-label">Confidence Score</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{bar_w}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            speaker_button(letter)

            st.markdown("<div class='section-hdr'>Top 5 Predictions</div>",
                        unsafe_allow_html=True)
            top5_bars(probs)

            with st.expander("🔬 Show preprocessed 28 × 28 input"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(img_28, clamp=True, use_container_width=True)
                with c2:
                    st.markdown(
                        "<p style='font-size:0.72rem;color:#4b5563;margin-top:0.4rem;line-height:1.7;'>"
                        "Greyscale · Auto-cropped to hand · Resized 28×28 · Normalised [0,1]"
                        "</p>", unsafe_allow_html=True,
                    )

            with st.expander("📊 Full probability distribution"):
                all_df = pd.DataFrame({
                    "Letter": ALL_LETTERS,
                    "Probability (%)": [round(p * 100, 3) for p in probs],
                }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
                st.dataframe(all_df, hide_index=True, use_container_width=True)

    render_sidebar(model_label, best_acc)


if __name__ == "__main__":
    main()
