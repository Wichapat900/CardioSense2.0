"""
CardioSense — Beat-Level Inference Module
----------------------------------------
Loads the trained ECG beat model and provides
functions to run predictions on ECG beats.

Classes:
0 = Normal
1 = PVC
2 = PAC
"""

import numpy as np
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/beat_model.keras"

# Load model once
model = load_model(MODEL_PATH)

# =========================
# PREPROCESSING
# =========================
def preprocess_beats(beats: np.ndarray) -> np.ndarray:
    """
    Normalize ECG beats per beat and add channel dimension.

    Parameters:
        beats (np.ndarray): shape (N, 360)

    Returns:
        np.ndarray: shape (N, 360, 1)
    """
    beats = beats.astype(np.float32)

    beats = (beats - beats.mean(axis=1, keepdims=True)) / \
            (beats.std(axis=1, keepdims=True) + 1e-8)

    return beats[..., np.newaxis]

# =========================
# INFERENCE
# =========================
def predict_beats(beats: np.ndarray):
    """
    Run beat-level inference.

    Parameters:
        beats (np.ndarray): shape (N, 360)

    Returns:
        preds (np.ndarray): predicted class per beat
        probs (np.ndarray): class probabilities per beat
    """
    beats = preprocess_beats(beats)

    probs = model.predict(beats, verbose=0)
    preds = np.argmax(probs, axis=1)

    return preds, probs