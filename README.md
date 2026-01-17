# Crash-Detection-System
📁 Dataset & Folder Structure

The dataset used for this project is available here:
🔗 Google Drive:
https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhgsXWtjODA2MR-F

The dataset consists of 1500 crash videos and 3000 normal driving videos, which have been preprocessed into 4D tensors suitable for deep learning models.

📂 Directory Structure

Car-Crash-Detection/
│

├── videos/

│   ├── Normal/                  # Normal driving videos

│   │   ├── 000001.mp4
│   │   ├── 000002.mp4
│   │   ├── ...
│   │   └── 003000.mp4
│   │
│   ├── Crash/                   # Crash / accident videos
│   │   ├── 000001.mp4
│   │   ├── 000002.mp4
│   │   ├── ...
│   │   └── 001500.mp4
│

🧠 Model Overview

The first architecture implemented in this project is a CNN–LSTM based model:

CNN for spatial feature extraction from video frames

LSTM for learning temporal dependencies across frame sequences

This combination enables effective classification of crash vs non-crash events from video data.
