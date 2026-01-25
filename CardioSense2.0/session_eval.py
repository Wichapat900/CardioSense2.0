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

    if beats.shape[0] != SESSION_BEATS:
        raise ValueError(
            f"Expected {SESSION_BEATS} beats, got {beats.shape[0]}"
        )

    # Beat-level inference
    preds, probs = predict_beats(beats)

    # Abnormal = anything not Normal
    abnormal_mask = preds != 0
    abnormal_count = int(np.sum(abnormal_mask))
    abnormal_ratio = abnormal_count / SESSION_BEATS

    # Final session decision
    result = "Abnormal" if abnormal_ratio >= ABNORMAL_THRESHOLD else "Normal"

    beat_predictions = [CLASS_NAMES[int(p)] for p in preds]

    return {
        "result": result,
        "confidence": float(abnormal_ratio),
        "abnormal_beats": abnormal_count,
        "beat_predictions": beat_predictions
    }
