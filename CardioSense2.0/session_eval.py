"""
CardioSense — Session-Level Evaluation Module
---------------------------------------------
Aggregates beat-level predictions into a
clinically meaningful session-level decision.

Input:
- Multiple ECG beats (e.g. 30 beats)

Output:
- Normal / Abnormal
- Confidence (abnormal burden)
- Abnormal beat count
"""

import numpy as np
from inference import predict_beats

# =========================
# CONFIG
# =========================
SESSION_BEATS = 30
ABNORMAL_THRESHOLD = 0.20  # 20% abnormal beats

CLASS_NAMES = {
    0: "Normal",
    1: "PVC",
    2: "PAC"
}

# =========================
# SESSION EVALUATION
# =========================
def evaluate_session(beats: np.ndarray):
    """
    Evaluate a session of ECG beats.

    Parameters:
        beats (np.ndarray): shape (SESSION_BEATS, 360)

    Returns:
        dict with:
            - result: "Normal" or "Abnormal"
            - confidence: abnormal ratio
            - abnormal_beats: count
            - beat_predictions: list of beat labels
    """
    assert beats.shape[0] == SESSION_BEATS, \
        f"Expected {SESSION_BEATS} beats, got {beats.shape[0]}"

    preds, probs = predict_beats(beats)

    # Abnormal = anything not Normal
    abnormal_mask = preds != 0
    abnormal_count = int(np.sum(abnormal_mask))
    abnormal_ratio = abnormal_count / SESSION_BEATS

    # Final session decision
    result = "Abnormal" if abnormal_ratio >= ABNORMAL_THRESHOLD else "Normal"

    beat_predictions = [CLASS_NAMES[p] for p in preds]

    return {import numpy as np

SESSION_BEATS = 30

def evaluate_session(beats, model):
    beats = (beats - beats.mean(axis=1, keepdims=True)) / (
        beats.std(axis=1, keepdims=True) + 1e-8
    )
    beats = beats[..., np.newaxis]

    probs = model.predict(beats)
    preds = np.argmax(probs, axis=1)

    abnormal = (preds != 0).astype(int)
    abnormal_ratio = abnormal.sum() / SESSION_BEATS

    result = "Abnormal" if abnormal_ratio >= 0.1 else "Normal"

    return {
        "result": result,
        "confidence": abnormal_ratio,
        "abnormal_beats": abnormal.sum(),
        "beat_predictions": preds.tolist()
    }
        "result": result,
        "confidence": abnormal_ratio,
        "abnormal_beats": abnormal_count,
        "beat_predictions": beat_predictions
    }
