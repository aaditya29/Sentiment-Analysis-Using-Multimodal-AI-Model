# Model Explanation

## What Is Model About?

We are going to do `Sentiment Analysis of Video` using Multimodal EmotionLines Dataset (MELD). <br>
MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from <b>Friends TV series.</b><br>
Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.

### Representing Video as Tensors for Machine Learning Models

# Representing Video as Tensors for Machine Learning Models

## Introduction

In modern AI and machine learning applications, videos are represented as **tensors** to enable processing by deep learning models. This document provides a thorough understanding of how videos are structured as tensors, the mathematical principles involved, and practical implementation using Python.

## 1. Understanding Video as a Data Structure

A video is a sequence of images (frames) played over time. It can be thought of as a 3D signal evolving across time:

- **Width (W)**: Number of pixels in each row.
- **Height (H)**: Number of pixels in each column.
- **Channels (C)**: Number of color channels (e.g., RGB has 3 channels, grayscale has 1).
- **Time (T)**: Number of frames in the video sequence.

Thus, a video can be represented as a **4D tensor**:
$[
V \in \mathbb{R}^{T \times C \times H \times W}
]$
where:

- $(T)$ = Number of frames
- $(C)$ = Number of channels
- $(H)$ = Frame height (pixels)
- $(W)$ = Frame width (pixels)

## 2. Mathematical Representation

Each pixel in an image has intensity values across channels, making a single frame a **3D tensor** $(C*H*W)$ Stacking these over **time** ($(T)$), we get the **4D tensor**:
$[
V[i, c, h, w] \in \mathbb{R}
]$
where:

- $(i)$ indexes the frame in time.
- $(c)$ indexes the channel (Red, Green, Blue for RGB).
- $(h, w)$ specify the spatial coordinates.

### Alternative Representations

1. **Batch Representation** (Batch of Videos):
   $V \in R^{B*T*C*H*W}$

   where $(B)$ is the batch size (number of videos processed simultaneously).

2. **Flattened Representation** (Used in Transformers):
   $V \in R^{T*(C*H*W)}$
   where each frame is converted into a flattened feature vector.

### 3. Practical Implementation

### Loading a Video as a Tensor

We use `torchvision` and `opencv` to read video frames and convert them into a PyTorch tensor.

```python
import cv2
import torch
import torchvision.transforms as transforms

# Load video using OpenCV
cap = cv2.VideoCapture('video.mp4')
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frames.append(frame)

cap.release()

# Convert list of frames to a tensor
video_tensor = torch.tensor(frames, dtype=torch.float32)  # Shape: (T, H, W, C)
video_tensor = video_tensor.permute(0, 3, 1, 2)  # Shape: (T, C, H, W)
```

### Normalizing the Video Tensor

Normalization is crucial for stable training in deep learning models:

```python
video_tensor /= 255.0  # Normalize pixel values to [0,1]
```
