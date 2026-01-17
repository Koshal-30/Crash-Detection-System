# Crash-Detection-System
The dataset used for this project can be found here:https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F
Preprocessed the 1500 Crash Videos and 3000 Normal Videos into 4-Dimensional Tensors that can be plugged into Deep Learning Models
The first architecture we are going to use is a combination of Convolutional Neural Networks and Long Short Term Memory.


The Folder Structure needs to be in the following manner:
Car Crash Detection 
├── videos
│   ├── Normal               # normal driving videos
│   │   ├── 000001.mp4
│   │   ├── ...
│   │   └── 003000.mp4
│   ├── Crash                # crash accident videos
│   │   ├── 000001.mp4
│   │   ├── ...
│   │   └── 001500.mp4
│   └── Crash-1500.txt       # annotation file for crash accident
