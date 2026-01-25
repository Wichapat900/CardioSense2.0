"""
CardioSense — Streamlit Demo App
--------------------------------
Demo application for ECG arrhythmia screening
using a trained beat-level model and session-level logic.
"""

import streamlit as st
import numpy as np
import os
from tensorflow import keras
from session_eval import evaluate_session, SESSION_BEATS

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="CardioSense",
    page_icon="🫀",
    layout="centered"
)

# =========================
# LOAD MODEL (ONCE)
# =========================
MODEL_PATH = os.path.join("models", "beat_model.keras")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# TITLE
# =========================
st.title("🫀 CardioSense")
st.subheader("AI-based ECG Arrhythmia Screening Demo")

st.markdown(
    """
This demo analyzes **30 ECG beats** and determines whether the session
is **Normal** or **Abnormal** based on AI predictions.

⚠️ This is a **screening tool**, not a medical diagnosis.
"""
)

# =========================
# INPUT METHOD
# =========================
st.header("Step 1: Provide ECG Beats")

option = st.radio(
    "Choose input method:",
    ("Use demo MIT-BIH sample", "Upload .npy file")
)

beats = None

# -------------------------
# DEMO MODE
# -------------------------
if option == "Use demo MIT-BIH sample":
    st.info("Using real demo ECG beats derived from MIT-BIH.")

    demo_type = st.selectbox(
        "Choose demo ECG rhythm:",
        ("Normal", "PVC", "PAC")
    )

    demo_path = {
        "Normal": "demo_data/normal.npy",
        "PVC": "demo_data/pvc.npy",
        "PAC": "demo_data/pac.npy"
    }[demo_type]

    beats = np.load(demo_path)

    if beats.shape != (SESSION_BEATS, 360):
        st.error(f"Demo data shape invalid: {beats.shape}")
        beats = None

# -------------------------
# UPLOAD MODE
# -------------------------
else:
    uploaded_file = st.file_uploader(
        "Upload a NumPy file with shape (30, 360)",
        type=["npy"]
    )

    if uploaded_file is not None:
        beats = np.load(uploaded_file)

        if beats.shape != (SESSION_BEATS, 360):
            st.error(f"Expected shape (30, 360), got {beats.shape}")
            beats = None

# =========================
# RUN ANALYSIS
# =========================
st.header("Step 2: Run Analysis")

if st.button("Analyze ECG Session"):

    if beats is None:
        st.warning("Please provide valid ECG beats first.")
    else:
        with st.spinner("Analyzing ECG session..."):
            result = evaluate_session(beats, model)

        st.success("Analysis complete")

        # =========================
        # RESULTS
        # =========================
        st.header("Results")

        if result["result"] == "Normal":
            st.markdown("### 🟢 **Normal Rhythm Detected**")
        else:
            st.markdown("### 🔴 **Abnormal Rhythm Detected**")

        st.metric(
            label="Abnormal Beat Burden",
            value=f"{result['confidence']*100:.1f}%"
        )

        st.metric(
            label="Abnormal Beats",
            value=f"{result['abnormal_beats']} / {SESSION_BEATS}"
        )

        with st.expander("Show beat-level predictions"):
            st.write(result["beat_predictions"])

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "CardioSense — Educational AI Project | Not for clinical use"
)
