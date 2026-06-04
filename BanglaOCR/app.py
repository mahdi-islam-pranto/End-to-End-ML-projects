# ============================================================
# Bangla Word Recognizer — Streamlit front-end
# Draws a word → segments characters → predicts each one
# ============================================================
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ── Paths ────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODEL_PATH  = ROOT / "bangla_ocr_model.keras"
LABELS_PATH = ROOT / "labels.json"

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="বাংলা OCR",
    page_icon="✍️",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
}
.bangla-text {
    font-family: 'Noto Sans Bengali', sans-serif;
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #f0f0f0;
    background: #1a1a2e;
    border: 1px solid #e94560;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 0.5rem 0;
}
.char-card {
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
    margin: 4px;
}
.char-label {
    font-family: 'Noto Sans Bengali', sans-serif;
    font-size: 1.6rem;
    color: #e94560;
    font-weight: 600;
}
.conf-text {
    font-size: 0.72rem;
    color: #a0a0b0;
    margin-top: 4px;
}
.section-header {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a0a0b0;
    margin-bottom: 0.4rem;
}
.stButton > button {
    background: #e94560 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: #c73652 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resource loader ───────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_labels():
    model  = tf.keras.models.load_model(str(MODEL_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)          # {"0": "ক", "1": "খ", ...}
    # Convert to list indexed by int: labels[0] = "ক"
    labels = [label_map[str(i)] for i in range(len(label_map))]
    return model, labels

model, labels = load_model_and_labels()
IMG_SIZE = (64, 64)


# ── Preprocessing helpers ────────────────────────────────────
def rgba_to_gray_numpy(rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA canvas image to grayscale uint8 numpy array."""
    img = Image.fromarray(rgba.astype("uint8"), mode="RGBA").convert("L")
    return np.array(img)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Binarize: canvas uses white stroke on black bg.
    Returns a binary image where strokes = 255, background = 0.
    Uses Otsu thresholding for robustness.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def segment_characters(binary: np.ndarray) -> list[np.ndarray]:
    """
    Segment individual characters from a binary word image.

    Strategy:
      1. Find connected components (handles gaps in strokes).
      2. Group overlapping/nearby bounding boxes (handles broken strokes,
         matras, and diacritics that sit above/below a character).
      3. Sort left-to-right (Bangla is written left-to-right).

    Returns a list of cropped grayscale character images (uint8).
    """
    # --- find all connected components ---
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA
    boxes = []
    for i in range(1, num_labels):          # skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:                        # skip tiny noise blobs
            continue
        boxes.append([x, y, x + w, y + h])

    if not boxes:
        return []

    # --- merge overlapping / nearby boxes (handles matra + body) ---
    # Sort by x so we process left-to-right
    boxes.sort(key=lambda b: b[0])

    # Horizontal merge gap: if two boxes are within this many pixels
    # horizontally they belong to the same character
    H_GAP = max(8, binary.shape[1] // 30)

    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        # Check horizontal overlap or closeness
        if box[0] <= prev[2] + H_GAP:
            # Merge: extend the previous box
            merged[-1] = [
                min(prev[0], box[0]),
                min(prev[1], box[1]),
                max(prev[2], box[2]),
                max(prev[3], box[3]),
            ]
        else:
            merged.append(box)

    # --- extract and return each character crop ---
    PAD = 6     # pixels of padding around each crop
    h_img, w_img = binary.shape
    char_imgs = []
    for x1, y1, x2, y2 in merged:
        x1p = max(0, x1 - PAD)
        y1p = max(0, y1 - PAD)
        x2p = min(w_img, x2 + PAD)
        y2p = min(h_img, y2 + PAD)
        crop = binary[y1p:y2p, x1p:x2p]
        char_imgs.append(crop)

    return char_imgs


def preprocess_char(char_img: np.ndarray) -> np.ndarray:
    """
    Resize a single character crop to IMG_SIZE, normalize to [0,1],
    and return shape (1, 64, 64, 1) ready for model.predict().

    The model was trained on: white bg, dark stroke, grayscale, /255 normalized.
    Canvas gives us: black bg, white stroke — so we invert first.
    """
    # Invert: white stroke on black → black stroke on white (matches training data)
    inverted = cv2.bitwise_not(char_img)
    # Resize to model input size
    resized  = cv2.resize(inverted, IMG_SIZE, interpolation=cv2.INTER_AREA)
    # Normalize
    arr      = resized.astype("float32") / 255.0
    # Add batch + channel dims
    return arr[np.newaxis, :, :, np.newaxis]


def predict_character(char_img: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Predict a single character image.
    Returns (predicted_label, confidence, full_probs_array).
    """
    x     = preprocess_char(char_img)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), probs


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
st.markdown("## ✍️  বাংলা Word Recognizer")
st.caption(
    "Draw a Bangla word on the canvas (white stroke, black background). "
    "The app segments it into characters and predicts each one."
)

col_left, col_right = st.columns([1.1, 1], gap="large")

# ── Left column: canvas ──────────────────────────────────────
with col_left:
    st.markdown('<p class="section-header">Draw here</p>', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=220,
        width=560,
        drawing_mode="freedraw",
        key="bangla-canvas",
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        predict_clicked = st.button("🔍  Predict word", use_container_width=True)
    with btn_col2:
        # Rerun clears the canvas state
        clear_clicked = st.button("🗑️  Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

    st.caption(
        "💡 Tips: use thick strokes · write characters with small gaps between them · "
        "avoid touching adjacent characters"
    )

# ── Right column: results ────────────────────────────────────
with col_right:
    st.markdown('<p class="section-header">Recognition result</p>', unsafe_allow_html=True)

    if predict_clicked and canvas_result.image_data is not None:

        gray   = rgba_to_gray_numpy(canvas_result.image_data)
        binary = binarize(gray)

        # Check canvas isn't empty
        if binary.sum() < 500:
            st.warning("Canvas looks empty — draw a word first.")
        else:
            char_imgs = segment_characters(binary)

            if not char_imgs:
                st.error("Could not detect any characters. Try drawing with thicker strokes.")
            else:
                # Predict each character
                predictions = []
                for img in char_imgs:
                    label, conf, probs = predict_character(img)
                    predictions.append((label, conf, probs))

                # ── Assembled word ───────────────────────────────
                word = "".join(p[0] for p in predictions)
                st.markdown(
                    f'<div class="bangla-text">{word}</div>',
                    unsafe_allow_html=True,
                )
                avg_conf = np.mean([p[1] for p in predictions]) * 100
                st.caption(f"Average confidence: {avg_conf:.1f}%  ·  {len(predictions)} character(s) detected")

                st.divider()

                # ── Per-character breakdown ──────────────────────
                st.markdown('<p class="section-header">Character breakdown</p>', unsafe_allow_html=True)

                cols = st.columns(min(len(predictions), 6))
                for i, (lbl, conf, probs) in enumerate(predictions):
                    col = cols[i % len(cols)]
                    with col:
                        # Show the segmented character image
                        display_img = cv2.resize(char_imgs[i], (80, 80), interpolation=cv2.INTER_NEAREST)
                        st.image(display_img, width=80)

                        st.markdown(
                            f'<div class="char-card">'
                            f'  <div class="char-label">{lbl}</div>'
                            f'  <div class="conf-text">{conf*100:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.divider()

                # ── Confidence table ─────────────────────────────
                with st.expander("Show full confidence scores"):
                    for i, (lbl, conf, probs) in enumerate(predictions):
                        st.markdown(
                            f'<p class="section-header">Character {i+1}: {lbl}</p>',
                            unsafe_allow_html=True,
                        )
                        top5_idx  = np.argsort(probs)[::-1][:5]
                        top5_data = {
                            "Character": [labels[j] for j in top5_idx],
                            "Confidence": [f"{probs[j]*100:.2f}%" for j in top5_idx],
                        }
                        st.table(top5_data)
    else:
        st.info("Draw a Bangla word on the left and press **Predict word**.")


# ── Sidebar: model info ───────────────────────────────────────
with st.sidebar:
    st.header("Model info")
    # st.write(f"**Classes:** `{len(labels)}`")
    st.write(f"**Input size:** `{IMG_SIZE[0]}×{IMG_SIZE[1]} px`")
    # st.write(f"**Architecture:** CNN + BiLSTM")
    st.write(f"**TensorFlow:** `{tf.__version__}`")
    st.divider()
    st.caption(
        "Labels loaded from `labels.json`. "
        "Model loaded from `bangla_ocr_model.keras`."
    )
    st.divider()
    st.markdown("**Known limitations**")
    st.caption(
        "• Conjunct consonants (যুক্তবর্ণ) may segment incorrectly\n"
        "• Matra (মাত্রা) may split from its base character\n"
        "• Model trained on isolated characters, not cursive words"
    )