# 🚗💥 Crash Detection System

> A deep learning pipeline that classifies dashcam / surveillance footage as **CRASH** or **NORMAL** in real time — with a Flask web app supporting both local video uploads and YouTube URLs.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Web App Usage](#web-app-usage)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Road traffic accidents are a leading cause of fatalities worldwide. This project builds an end-to-end video classification system that automatically detects crash events from traffic footage using deep learning.

Three architectures were explored and compared:

| Model | Description |
|---|---|
| **CNN + LSTM** | Custom spatial-temporal model; CNN extracts per-frame features, LSTM captures motion dynamics |
| **MobileNetV2 (Tuned)** | Transfer learning with MobileNetV2 backbone fine-tuned on the crash dataset |
| **Preprocessing Pipeline** | Clip-based sampling (3 clips × 16 frames) + resize to 112×112, normalised to [0, 1] |

The best-performing model is served via a Flask API that accepts video file uploads or YouTube links and returns a **CRASH / NORMAL** prediction with a confidence score.

---

## Demo

```
Input  : dashcam video (upload or YouTube URL)
Output : { "prediction": "CRASH", "confidence": 91.4, "clips_analysed": 3 }
```

The web interface lets you drag-and-drop a video or paste a YouTube link and get an instant prediction.

---

## Architecture

### CNN + LSTM Pipeline

```
Video
  │
  ▼
Clip Sampler  ──►  3 clips × 16 frames  (evenly spaced across the video)
  │
  ▼
Per-Frame CNN  ──►  Spatial features per frame  (112 × 112 × 3 → feature vector)
  │
  ▼
LSTM  ──►  Temporal reasoning across the 16-frame sequence
  │
  ▼
Dense + Sigmoid  ──►  P(CRASH)  →  threshold 0.5  →  CRASH / NORMAL
```

### MobileNetV2 (Transfer Learning)

- **Backbone**: MobileNetV2 pretrained on ImageNet, top layers removed
- **Fine-tuning**: Last N layers unfrozen and retrained on crash dataset
- **Head**: GlobalAveragePooling → Dense(256, ReLU) → Dropout → Dense(1, Sigmoid)

---

## Dataset

| Split | Crash Videos | Normal Videos | Total |
|---|---|---|---|
| Full Dataset | 1,500 | 3,000 | 4,500 |

> 📥 **Download**: [Google Drive](https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhgsXWtjODA2MR-F)

Videos are preprocessed into **4D tensors** `(num_clips, seq_len, height, width, channels)` using the `Clipping_Preprocessing.ipynb` notebook before training.

**Preprocessing config:**

```python
SEQ_LEN       = 16     # frames per clip
IMG_SIZE      = 112    # height & width after resize
CLIPS_PER_VID = 3      # clips sampled per video
```

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| CNN + LSTM | — | Baseline temporal model |
| MobileNetV2 (Tuned) | — | Transfer learning with fine-tuning |

> ℹ️ Fill in your final validation accuracy/F1 scores here before publishing.

---

## Project Structure

```
Crash-Detection-System/
│
├── notebooks/
│   ├── Clipping_Preprocessing.ipynb   # Video → tensor preprocessing pipeline
│   ├── CNN+LSTM.ipynb                 # CNN-LSTM training & evaluation
│   └── MobileNetV2Tuned.ipynb         # MobileNetV2 fine-tuning & evaluation
│
├── templates/
│   └── index.html                     # Flask frontend (drag-and-drop + YouTube)
│
├── app.py                             # Flask inference server
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md
```

> ⚠️ **Model file** (`ccd_cnn_lstm_model.keras`) is **not included** in this repo due to size.  
> Download it from [Google Drive](https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhgsXWtjODA2MR-F) and place it in the **root directory** before running the app.

---

## Getting Started

### Prerequisites

- Python 3.9+
- `conda` or `venv` recommended

### 1. Clone the repo

```bash
git clone https://github.com/Koshal-30/Crash-Detection-System.git
cd Crash-Detection-System
```

### 2. Create a virtual environment

```bash
conda create -n crash-detection python=3.11 -y
conda activate crash-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

Download `ccd_cnn_lstm_model.keras` from [Google Drive](https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhgsXWtjODA2MR-F) and place it in the root of the project:

```
Crash-Detection-System/
└── ccd_cnn_lstm_model.keras   ← here
```

You can also set a custom path via environment variable:

```bash
export MODEL_PATH=/path/to/your/model.keras
```

### 5. Run the Flask app

```bash
python app.py
```

Open your browser at `http://localhost:5000`.

---

## Web App Usage

The app exposes two ways to analyse a video:

**Option A — Upload a local video file**

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` (max 200 MB)

**Option B — Paste a YouTube URL**

Requires `yt-dlp` (included in `requirements.txt`). The video is downloaded temporarily, analysed, and deleted immediately.

### API

```
POST /predict
Content-Type: multipart/form-data

Body (file upload):   video=<file>
Body (YouTube URL):   video_url=<url>
```

**Example response:**

```json
{
  "prediction":     "CRASH",
  "confidence":     91.4,
  "raw_score":      0.9139,
  "clips_analysed": 3,
  "clip_scores":    [0.8821, 0.9204, 0.9392]
}
```

---

## Roadmap

- [x] Preprocessing pipeline (clip sampling + normalisation)
- [x] CNN + LSTM classifier
- [x] MobileNetV2 fine-tuned classifier
- [x] Flask inference API with YouTube support
- [ ] Add model performance metrics (confusion matrix, ROC curve) to README
- [ ] Experiment with 3D-CNN (C3D / SlowFast)
- [ ] Deploy to cloud (Render / HuggingFace Spaces)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)

---

## License

This project is licensed under the MIT License.
