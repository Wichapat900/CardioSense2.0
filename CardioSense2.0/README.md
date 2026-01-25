🫀 CardioSense

AI-Powered ECG Arrhythmia Detection System

📌 Overview

CardioSense is an AI-based ECG analysis system designed to detect cardiac arrhythmias automatically from ECG signals.
The goal of this project is to help identify abnormal heart rhythms that often go unnoticed, especially when continuous hospital monitoring is not available.

This project combines:

Real ECG data (MIT-BIH Arrhythmia Database)

Deep Learning (1D CNN)

Session-level analysis (30-beat aggregation)

Web deployment using Streamlit

🎯 Problem Statement

Many patients experience intermittent arrhythmias that:

Do not occur during short hospital ECG tests

Go undetected for long periods

Require continuous monitoring, which is expensive and inaccessible

CardioSense aims to provide:

Automatic detection

Timestamped abnormal events

A foundation for portable or wearable ECG monitoring systems

🧠 Features

✅ Beat-level ECG classification

✅ Binary detection: Normal vs Abnormal

✅ Session-level decision using 30 beats

✅ Trained on real ECG signals

✅ Web-based interface (Streamlit)

🧪 Dataset

MIT-BIH Arrhythmia Database

ECG beats extracted and labeled into:

Normal

PVC (Premature Ventricular Contraction)

PAC (Premature Atrial Contraction)

Raw MIT-BIH files are not included in this repository.
Only processed beat data (.npy) is used for training and inference.

🏗️ Model Architecture

1D Convolutional Neural Network (CNN)

Input: ECG beat (360 samples)

Layers:

Conv1D + MaxPooling

Dense + Dropout

Softmax output
