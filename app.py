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


# ------------------------------------------------------------------ CSS

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    #MainMenu, footer, header { visibility: hidden; }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #07070f;
    }
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px;
    }

    /* ── Navbar ── */
    .navbar {
        position: fixed; top: 0; left: 0;
        width: 100%; z-index: 9999;
        background: rgba(7,7,15,0.88);
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(139,92,246,0.12);
        display: flex; align-items: center;
        justify-content: space-between;
        padding: 0 2.5rem; height: 54px;
    }
    .navbar-brand {
        font-size: 1rem; font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .navbar-pill {
        font-size: 0.68rem; font-weight: 600; color: #a78bfa;
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 20px; padding: 0.22rem 0.8rem;
    }

    /* ── Compact hero ── */
    .hero-row {
        display: flex; align-items: center; gap: 1rem;
        padding: 1.2rem 0 1rem;
    }
    .hero-badge {
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #a78bfa;
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.22);
        border-radius: 20px; padding: 0.22rem 0.8rem;
        white-space: nowrap;
    }
    .hero-title {
        font-size: 1.8rem; font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg,#fff 0%,#a78bfa 60%,#60a5fa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1;
    }
    .hero-sub {
        font-size: 0.8rem; color: #4b5563; margin: 0;
    }

    /* ── Upload panel ── */
    .panel-label {
        font-size: 0.62rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #4b5563; margin-bottom: 0.6rem;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1.5px dashed rgba(139,92,246,0.3) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139,92,246,0.6) !important;
        background: rgba(139,92,246,0.04) !important;
    }
    .upload-hint {
        font-size: 0.75rem; color: #374151; margin-top: 0.4rem;
    }

    /* ── Predict button ── */
    div[data-testid="stButton"] > button {
        width: 100%; padding: 0.75rem 1.5rem;
        font-size: 0.9rem; font-weight: 700;
        letter-spacing: 0.05em; color: #fff;
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        border: none; border-radius: 12px; cursor: pointer;
        transition: opacity 0.2s, transform 0.1s, box-shadow 0.2s;
        box-shadow: 0 4px 20px rgba(124,58,237,0.35);
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.9; transform: translateY(-1px);
        box-shadow: 0 6px 28px rgba(124,58,237,0.5);
    }
    div[data-testid="stButton"] > button:active { transform: translateY(0); }

    /* ── Image preview ── */
    .preview-box {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 0.8rem;
        margin-top: 0.8rem;
    }
    .preview-caption {
        font-size: 0.6rem; letter-spacing: 0.08em;
        text-transform: uppercase; color: #374151;
        text-align: center; margin-top: 0.4rem;
    }

    /* ── Result card ── */
    .result-card {
        background: linear-gradient(135deg,rgba(124,58,237,0.1),rgba(96,165,250,0.06));
        border: 1px solid rgba(139,92,246,0.22);
        border-radius: 20px; padding: 1.6rem 1.4rem;
    }
    .result-header {
        font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: #6b7280; margin-bottom: 0.2rem;
    }
    .result-letter {
        font-size: 7rem; font-weight: 800; line-height: 1;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg,#fff,#a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .result-conf {
        font-size: 1.6rem; font-weight: 700; color: #a78bfa;
        margin-top: 0.2rem;
    }
    .result-conf-label {
        font-size: 0.6rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: #4b5563;
    }
    .conf-bar-bg {
        background: rgba(255,255,255,0.05);
        border-radius: 99px; height: 5px; overflow: hidden;
        margin-top: 0.5rem;
    }
    .conf-bar-fill {
        height: 5px; border-radius: 99px;
        background: linear-gradient(90deg,#7c3aed,#a78bfa,#60a5fa);
    }

    /* ── Speaker button (HTML) ── */
    .speaker-btn {
        display: inline-flex; align-items: center; gap: 0.5rem;
        margin-top: 1rem; padding: 0.55rem 1.2rem;
        font-size: 0.82rem; font-weight: 700;
        color: #a78bfa; cursor: pointer;
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 10px; transition: all 0.2s;
        font-family: 'Inter', sans-serif;
    }
    .speaker-btn:hover {
        background: rgba(139,92,246,0.2);
        border-color: rgba(139,92,246,0.6);
        transform: translateY(-1px);
    }

    /* ── Top-5 bars ── */
    .section-header {
        font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #374151; margin: 1rem 0 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .top5-row {
        display: flex; align-items: center;
        gap: 0.6rem; margin-bottom: 0.4rem;
    }
    .top5-letter {
        font-size: 0.82rem; font-weight: 700;
        color: #e5e7eb; width: 1.1rem; text-align: center;
    }
    .top5-bar-bg {
        flex: 1; background: rgba(255,255,255,0.05);
        border-radius: 99px; height: 4px; overflow: hidden;
    }
    .top5-bar-fill { height: 4px; border-radius: 99px; }
    .top5-pct {
        font-size: 0.7rem; color: #6b7280;
        width: 3.2rem; text-align: right;
    }
    .top5-winner .top5-letter,
    .top5-winner .top5-pct { color: #a78bfa; }

    /* ── Placeholder ── */
    .placeholder {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        height: 100%; min-height: 320px;
        border: 1.5px dashed rgba(255,255,255,0.06);
        border-radius: 20px; text-align: center;
        color: #374151;
    }
    .placeholder-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
    .placeholder-text { font-size: 0.8rem; }

    /* ── Warn ── */
    .warn-banner {
        background: rgba(234,179,8,0.07);
        border: 1px solid rgba(234,179,8,0.18);
        border-radius: 10px; padding: 0.5rem 0.9rem;
        font-size: 0.75rem; color: #ca8a04; margin-bottom: 0.8rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #07070f !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    .sidebar-title {
        font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: #a78bfa; margin-bottom: 0.8rem;
    }
    .sidebar-row {
        display: flex; justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .sidebar-key { font-size: 0.7rem; color: #4b5563; }
    .sidebar-val { font-size: 0.7rem; font-weight: 600; color: #9ca3af; }
    .letter-grid {
        display: grid; grid-template-columns: repeat(6,1fr);
        gap: 0.25rem; margin-top: 0.5rem;
    }
    .letter-chip {
        background: rgba(139,92,246,0.08);
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 6px; text-align: center;
        padding: 0.22rem 0; font-size: 0.72rem;
        font-weight: 600; color: #7c3aed;
    }
    hr { border-color: rgba(255,255,255,0.05) !important; }
    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ navbar

def navbar(model_label: str) -> None:
    st.markdown(f"""
    <div class="navbar">
        <span class="navbar-brand">✦ Sign Language AI</span>
        <span class="navbar-pill">{model_label} · 24 classes · A–Y excl. J&Z</span>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ model

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


# ------------------------------------------------------------------ image processing

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
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img_grey   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_grey   = _crop_hand(img_grey)
    img_resized = cv2.resize(img_grey, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_norm   = img_resized.astype(np.float32) / 255.0
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


# ------------------------------------------------------------------ UI helpers

def speaker_button(letter: str) -> None:
    """Pure-HTML button that speaks the letter client-side — no Streamlit rerun."""
    components.html(f"""
    <style>
    .spk {{
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.5rem 1.2rem;
        font-size: 0.85rem; font-weight: 700;
        color: #a78bfa; cursor: pointer;
        background: rgba(139,92,246,0.1);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        transition: background 0.2s, transform 0.15s;
    }}
    .spk:hover {{ background: rgba(139,92,246,0.22); transform: translateY(-1px); }}
    .spk:active {{ transform: translateY(0); }}
    </style>
    <button class="spk" onclick="
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance('{letter}');
        u.rate = 0.85; u.pitch = 1.0; u.volume = 1.0;
        window.speechSynthesis.speak(u);
    ">🔊 &nbsp;Hear &ldquo;{letter}&rdquo;</button>
    """, height=52)


def top5_bars(probs: list) -> None:
    top5   = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    max_p  = probs[top5[0]]
    colors = ["#a78bfa", "#818cf8", "#60a5fa", "#38bdf8", "#34d399"]
    html   = ""
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


def render_sidebar(model_label: str, best_acc) -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Model Info</div>", unsafe_allow_html=True)
        rows = [
            ("Dataset",   "Sign Language MNIST"),
            ("Model",     model_label or "—"),
            ("Accuracy",  f"{best_acc*100:.1f}%" if best_acc else "N/A"),
            ("Classes",   "24  (A–Y, excl. J & Z)"),
            ("Input",     "28 × 28 greyscale"),
            ("Framework", "PyTorch"),
        ]
        html = "".join(f"<div class='sidebar-row'><span class='sidebar-key'>{k}</span>"
                       f"<span class='sidebar-val'>{v}</span></div>" for k, v in rows)
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-title'>Supported Letters</div>", unsafe_allow_html=True)
        chips = "".join(f"<div class='letter-chip'>{l}</div>" for l in ALL_LETTERS)
        st.markdown(f"<div class='letter-grid'>{chips}</div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.62rem;color:#374151;line-height:1.6;'>"
            "Best results: plain background · hand centred · even lighting."
            "</p>", unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="Sign Language AI", page_icon="✦", layout="wide")
    inject_css()

    model, device, model_label = load_model()
    navbar(model_label or "No model")

    if model is None:
        st.error("No trained model found. Run `python -m src.train_cnn` first.")
        st.stop()

    ckpt_path  = CNN_PATH if model_label == "ConvNet" else MLP_PATH
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_acc   = ckpt.get("val_accuracy", None)

    # ── Compact hero ────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-row">
        <div class="hero-badge">✦ ASL Recognition</div>
        <div class="hero-title">Read the Signs</div>
        <p class="hero-sub">Upload a hand sign → click Predict → hear & see the letter.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_label == "MLP":
        st.markdown(
            "<div class='warn-banner'>⚠ MLP active — train CNN for best accuracy: "
            "<code>python -m src.train_cnn</code></div>", unsafe_allow_html=True)

    # ── Two-column layout: left = upload, right = result ─────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: upload + predict ───────────────────────────────────────────
    with left:
        st.markdown("<div class='panel-label'>📁 &nbsp;Upload your hand sign</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="upload", type=["png", "jpg", "jpeg", "webp", "avif"],
            label_visibility="collapsed",
        )

        if uploaded is None:
            st.markdown(
                "<p class='upload-hint'>PNG · JPG · WEBP · AVIF</p>",
                unsafe_allow_html=True)
        else:
            # Show thumbnail preview
            raw = uploaded.read()
            st.markdown("<div class='preview-box'>", unsafe_allow_html=True)
            st.image(Image.open(io.BytesIO(raw)), use_container_width=True)
            st.markdown("<div class='preview-caption'>Ready to predict</div></div>",
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✦  Predict Sign", use_container_width=True):
                try:
                    tensor, img_28 = preprocess_image(raw, model_label)
                    letter, conf, probs = run_predict(model, tensor, device)
                    st.session_state["result"] = {
                        "letter": letter, "conf": conf,
                        "probs": probs, "img_28": img_28,
                        "raw": raw,
                    }
                except ValueError as e:
                    st.error(str(e))

    # ── RIGHT: result panel ───────────────────────────────────────────────
    with right:
        res = st.session_state.get("result")

        if res is None:
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-icon">🤟</div>
                <div class="placeholder-text">Your prediction will appear here</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            letter = res["letter"]
            conf   = res["conf"]
            probs  = res["probs"]
            img_28 = res["img_28"]
            raw    = res["raw"]
            bar_w  = int(conf * 100)

            # ── Result card ──
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">Predicted Letter</div>
                <div class="result-letter">{letter}</div>
                <div class="result-conf">{conf*100:.1f}%</div>
                <div class="result-conf-label">Confidence</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{bar_w}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Speaker button (pure HTML — no page reload) ──
            speaker_button(letter)

            # ── Top-5 ──
            st.markdown("<div class='section-header'>Top 5 Predictions</div>",
                        unsafe_allow_html=True)
            top5_bars(probs)

            # ── Preprocessed thumbnail ──
            with st.expander("Show preprocessed 28×28 image"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(img_28, clamp=True, use_container_width=True)
                with c2:
                    st.markdown(
                        "<p style='font-size:0.72rem;color:#4b5563;margin-top:0.5rem;'>"
                        "Greyscale · Auto-cropped · Resized to 28×28 · Normalised to [0,1]"
                        "</p>", unsafe_allow_html=True)

            # ── Full distribution ──
            with st.expander("Full probability distribution"):
                all_df = pd.DataFrame({
                    "Letter": ALL_LETTERS,
                    "Probability (%)": [round(p * 100, 3) for p in probs],
                }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
                st.dataframe(all_df, hide_index=True, use_container_width=True)

    render_sidebar(model_label, best_acc)


if __name__ == "__main__":
    main()
