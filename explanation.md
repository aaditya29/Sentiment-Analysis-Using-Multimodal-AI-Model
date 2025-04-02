# Model Explanation

## What Is Model About?

We are going to do `Sentiment Analysis of Video` using Multimodal EmotionLines Dataset (MELD). <br>
MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from <b>Friends TV series.</b><br>
Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.

### Representing Video as Tensors for Machine Learning Models

In modern AI and machine learning applications, videos are represented as **tensors** to enable processing by deep learning models. This document provides a thorough understanding of how videos are structured as tensors, the mathematical principles involved, and practical implementation using Python.

#### Understanding Video as a Data Structure

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

#### Mathematical Representation

Each pixel in an image has intensity values across channels, making a single frame a **3D tensor** $(C*H*W)$ Stacking these over **time** ($(T)$), we get the **4D tensor**:
$[
V[i, c, h, w] \in \mathbb{R}
]$
where:

- $(i)$ indexes the frame in time.
- $(c)$ indexes the channel (Red, Green, Blue for RGB).
- $(h, w)$ specify the spatial coordinates.

#### Alternative Representations

1. **Batch Representation** (Batch of Videos):
   $V \in R^{B*T*C*H*W}$

   where $(B)$ is the batch size (number of videos processed simultaneously).

2. **Flattened Representation** (Used in Transformers):
   $V \in R^{T*(C*H*W)}$
   where each frame is converted into a flattened feature vector.

#### Practical Implementation

> Loading a Video as a Tensor

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

> Normalizing the Video Tensor

Normalization is crucial for stable training in deep learning models:

```python
video_tensor /= 255.0  # Normalize pixel values to [0,1]
```

### Tokenization with BERT Tokenizer

Tokenization is a fundamental preprocessing step in Natural Language Processing (NLP). The **BERT Tokenizer** is specifically designed for **Bidirectional Encoder Representations from Transformers (BERT)**, handling tokenization in a way that preserves subword information while efficiently processing large vocabularies.

#### Understanding Tokenization in NLP

Tokenization is the process of converting raw text into meaningful units (tokens) that a model can process. Traditional tokenization approaches include:

- **Word-based tokenization**: Splitting text into words (e.g., `"Hello world" → ["Hello", "world"]`).
- **Character-based tokenization**: Splitting text into characters (e.g., `"Hello" → ["H", "e", "l", "l", "o"]`).
- **Subword tokenization (used in BERT)**: Breaking words into smaller units based on frequency (`"playing" → ["play", "##ing"]`).

BERT uses **WordPiece Tokenization**, which combines word-level and subword-level tokenization.

---

#### Mathematical Foundations of WordPiece Tokenization

BERT Tokenizer is based on **WordPiece Tokenization**, which is designed to:

- Efficiently represent large vocabularies with a limited number of tokens.
- Preserve the meaning of frequent words while breaking rare words into subwords.

> Vocabulary Construction

The goal is to build a vocabulary $( V)$ of size $( N )$ by iteratively merging character sequences.

1. **Start with Characters**: The initial vocabulary consists of all unique characters in the corpus.
2. **Compute Bigram Frequencies**: Calculate the frequency of adjacent character pairs.
3. **Merge Most Frequent Pair**: Combine the most frequent character pair into a new token.
4. **Repeat Until Vocabulary Size \( N \) is Reached**.

Formally, given a word **W**, the probability of splitting it into subwords $( S*1, S_2, ..., S_k )$ is:<br>
$[
P(W) = \prod*{i=1}^{k} P(S*i | S_1, ..., S*{i-1})
]$ <br>
where **merging pairs optimizes this probability**.

> Token Representation

Each word **w** is tokenized into a sequence of subwords **s**:
$[
T(w) = (s_1, s_2, ..., s_k)
]$
where **unseen words** are decomposed into known subwords with special prefixes (e.g., `##ing`).

---

#### Practical Implementation of BERT Tokenizer

> Installing and Importing BERT Tokenizer

We use the `transformers` library from Hugging Face:

```python
!pip install transformers
```

```python
from transformers import BertTokenizer

# Load Pretrained BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

> 3.2 Tokenizing a Sentence

```python
sentence = "Tokenization with BERT is powerful!"
tokens = tokenizer.tokenize(sentence)
print(tokens)
```

**Output:**

```bash
['token', '##ization', 'with', 'bert', 'is', 'powerful', '!']
```

- `token` and `##ization` show subword splitting.
- `bert` remains unchanged as it's in the vocabulary.
- `!` is a separate token.

> Converting Tokens to Input IDs

```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
```

This maps tokens to unique integers used as input for BERT.

> Encoding a Full Sentence for BERT

BERT requires additional processing:

```python
encoding = tokenizer(sentence, return_tensors='pt')
print(encoding)
```

This returns a **PyTorch tensor** containing:

- `input_ids`: Tokenized sentence as IDs.
- `token_type_ids`: Distinguishes sentences in a pair.
- `attention_mask`: Masks padding tokens.

---

#### Special Tokens in BERT Tokenization

BERT uses special tokens:

- `[CLS]`: Represents the entire sentence (used for classification).
- `[SEP]`: Separates sentences in a pair.
- `[PAD]`: Used for padding shorter sentences.
- `[UNK]`: Represents unknown words.

Example:

```python
sentence_pair = ["How are you?", "I am fine."]
encoding = tokenizer(sentence_pair, padding=True, truncation=True, return_tensors='pt')
print(encoding)
```

### Representing Audio In Tensors

Audio data plays a crucial role in various AI/ML applications, including **speech recognition, music generation, and environmental sound classification**. However, raw audio signals must be transformed into a numerical representation before being processed by machine learning models. This document provides an in-depth explanation of how audio is represented in tensors, covering **intuitive explanations, mathematical foundations, and practical implementations**.

---

#### Understanding Audio Data

Audio is a continuous signal that represents variations in **air pressure over time**. It is captured digitally through **sampling and quantization**.

> Sampling and Quantization

- **Sampling**: Converts a continuous waveform into discrete time steps.
  - Defined by the **sampling rate (𝑓_s)** in Hertz (Hz), e.g., **44.1 kHz** (CD quality audio) means 44,100 samples per second.
- **Quantization**: Converts each sample into a finite set of values, typically stored as **16-bit integers** (or floating point in deep learning models).

Mathematically, a sampled audio signal is represented as:
$[
x[n] = A \cdot \sin(2\pi f n / f_s)
]$
where:

- $( x[n])$ is the sampled signal at time index $( n ).$
- $( A )$ is the amplitude
- $( f )$ is the frequency in Hz
- $( f_s )$ is the sampling rate

Example: A **440 Hz** sine wave (A4 musical note) sampled at **16 kHz** results in **16,000 samples per second**.

---

#### Representing Audio as Tensors

Since ML models work with numerical arrays (tensors), we must convert raw audio into a suitable tensor representation. Common formats include:

> Raw Waveform Representation

Audio can be represented as a **1D tensor**:

```python
import torchaudio
import torch

# Load an example audio file
audio_waveform, sample_rate = torchaudio.load("example.wav")
print(audio_waveform.shape)  # Shape: (Channels, Samples)
```

For **stereo audio**, the shape is `(2, N)` (2 channels, N samples). For **mono audio**, the shape is `(1, N)`.

> Spectrogram Representation

A more structured way to represent audio is through the **spectrogram**, which shows frequency content over time using the Short-Time Fourier Transform (STFT).

Mathematically, STFT is computed as:
$[
STFT(x[n]) = \sum\_{m=-\infty}^{\infty} x[m] w[n - m] e^{-j2\pi f m / f_s}
]$
where:

- $( x[m] )$ is the time-domain signal.
- $( w[n] )$ is a windowing function (e.g., Hamming window)
- $( e^{-j2\pi f m / f_s} )$ represents the Fourier transform

#### Code Example

```python
import torchaudio.transforms as T

# Convert waveform to spectrogram
spectrogram = T.Spectrogram(n_fft=1024, win_length=512, hop_length=256)(audio_waveform)
print(spectrogram.shape)  # Shape: (Channels, Frequency_bins, Time_frames)
```

This converts audio into a **2D tensor** (frequency vs. time).

> Mel-Spectrogram Representation

A **Mel-Spectrogram** applies the **Mel scale**, which better matches human perception.

```python
mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80)(audio_waveform)
print(mel_spectrogram.shape)  # Shape: (Channels, Mel_bins, Time_frames)
```

> MFCC (Mel-Frequency Cepstral Coefficients)

MFCCs are features used in **speech processing**, derived from the Mel-spectrogram.

```python
mfcc = T.MFCC(sample_rate=sample_rate, n_mfcc=13)(audio_waveform)
print(mfcc.shape)  # Shape: (Channels, MFCC_coefficients, Time_frames)
```

This produces a **2D tensor**, reducing dimensionality while keeping important frequency characteristics.

---

#### Choosing the Right Representation

| Representation  | Description                             | Tensor Shape                     |
| --------------- | --------------------------------------- | -------------------------------- |
| Raw Waveform    | Simple but high-dimensional             | `(1, N)`                         |
| Spectrogram     | Captures frequency content over time    | `(Channels, Freq_bins, Time)`    |
| Mel-Spectrogram | Human-perceptual frequency scale        | `(Channels, Mel_bins, Time)`     |
| MFCC            | Compact representation for speech tasks | `(Channels, Coefficients, Time)` |

---

### Training from Scratch vs. Transfer Learning

| Approach                  | Pros                                                                                       | Cons                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| **Training from Scratch** | ✅ Full control: optimization for a specific task <br> ✅ Smaller model                    | ❌ Requires more data <br> ❌ Needs more training time <br> ❌ Might underperform |
| **Transfer Learning**     | ✅ Better initial performance <br> ✅ Less data needed <br> ✅ Architecture already proven | ❌ Larger model size <br> ❌ May not fit the task                                 |

---

### Working with Different Modalities

#### Working with Text

### Without a Pre-Trained Model:

❌ No difference between a "bank with money" and "a riverbank".  
 Both are encoded the same way without context.

#### With a Pre-Trained Model:

✅ The output reflects the specific meaning of "bank".  
 More context is preserved.

---

### Working with Video

#### Without a Pre-Trained Model:

❌ Simple brightness changes over time (e.g., brightness goes up, then down).

#### With a Pre-Trained Model:

✅ Recognizes complex patterns like a person nodding their head while smiling.  
✅ Identifies hand waves from motion, smiles from facial features, and nods from sequences.

## Architecture of the Model

Here we are going to use three types of architecture:

1. Video encoder Resnet3D 18 layer.
2. Text encoder BERT.
3. Audio encoder raw spectrogram.

## Different Ways of Fusing Our Data Together

### Late Fusion Technique

Late fusion, also known as decision-level fusion, is a technique in machine learning where predictions from multiple modalities (e.g., video, text, audio) are combined at the final stage of the model pipeline. Instead of merging raw data or intermediate features, late fusion aggregates the outputs (e.g., probabilities, logits) of individual models trained on each modality.

#### How It Works

1. **Separate Models for Each Modality**: Each modality (e.g., video, text, audio) is processed independently using specialized models.
2. **Generate Predictions**: Each model outputs predictions, such as class probabilities or logits.
3. **Combine Predictions**: The predictions are combined using techniques like:

- Weighted averaging
- Majority voting
- Concatenation followed by a meta-classifier

#### Example Workflow

1. **Video Encoder**: Processes video frames and outputs probabilities for each class.
2. **Text Encoder**: Processes text data and outputs probabilities for each class.
3. **Audio Encoder**: Processes audio signals and outputs probabilities for each class.
4. **Fusion Layer**: Combines the outputs from all encoders to make the final prediction.

#### Pros of Late Fusion

- **Modality Independence**: Each modality is processed independently, allowing flexibility in model design.
- **Scalability**: New modalities can be added without retraining the entire system.
- **Interpretability**: Individual modality contributions can be analyzed by examining their predictions.
- **Robustness**: If one modality fails (e.g., missing data), others can still contribute to the final decision.

#### Cons of Late Fusion

- **Loss of Cross-Modality Interactions**: Late fusion does not capture interactions between modalities, which may limit performance in tasks requiring joint understanding.
- **Increased Complexity**: Requires separate models for each modality, increasing computational and memory requirements.
- **Suboptimal for Strong Correlations**: When modalities are highly correlated, early or intermediate fusion may perform better.

#### Applications of Late Fusion

- **Multimodal Sentiment Analysis**: Combining video, text, and audio predictions to determine sentiment.
- **Medical Diagnosis**: Aggregating outputs from imaging, clinical notes, and lab results.
- **Multimodal Action Recognition**: Combining predictions from video and audio for action classification.

#### Comparison with Other Fusion Techniques

| Fusion Technique        | Description                                                                    | Pros                                      | Cons                                              |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------- |
| **Early Fusion**        | Combines raw data or features from all modalities before feeding into a model. | Captures cross-modality interactions      | Requires aligned data; computationally expensive. |
| **Intermediate Fusion** | Combines intermediate features from modality-specific models.                  | Balances interaction and independence     | Requires careful feature alignment.               |
| **Late Fusion**         | Combines predictions from modality-specific models.                            | Modality independence; robust to failures | Loses cross-modality interactions.                |

#### Practical Implementation

```python
import numpy as np

# Example predictions from three modalities
video_preds = np.array([0.2, 0.5, 0.3])  # Video model probabilities
text_preds = np.array([0.1, 0.7, 0.2])   # Text model probabilities
audio_preds = np.array([0.3, 0.4, 0.3])  # Audio model probabilities

# Weighted averaging for late fusion
weights = [0.4, 0.4, 0.2]  # Weights for video, text, and audio
final_preds = weights[0] * video_preds + weights[1] * text_preds + weights[2] * audio_preds

print("Final Predictions:", final_preds)
```

### Early Fusion Technique

Early fusion, also known as feature-level fusion, is a technique in machine learning where raw data or features from multiple modalities (e.g., video, text, audio) are combined at the input stage of the model pipeline. This approach integrates information from all modalities before feeding it into a unified model, allowing the model to learn joint representations.

#### How It Works

1. **Feature Extraction**: Extract features from each modality (e.g., embeddings for text, spectrograms for audio, and frame tensors for video).
2. **Feature Concatenation**: Combine the features from all modalities into a single feature vector.
3. **Unified Model**: Feed the concatenated feature vector into a single model for training and prediction.

#### Example Workflow

1. **Video Features**: Extracted using a video encoder (e.g., ResNet3D).
2. **Text Features**: Extracted using a text encoder (e.g., BERT).
3. **Audio Features**: Extracted using an audio encoder (e.g., spectrogram-based CNN).
4. **Fusion Layer**: Concatenates all features into a single vector.
5. **Unified Model**: Processes the fused vector to make predictions.

#### Pros of Early Fusion

- **Cross-Modality Interactions**: Captures relationships and dependencies between modalities, which can improve performance for tasks requiring joint understanding.
- **Simplified Training**: A single model is trained on the fused features, reducing the need for separate models for each modality.
- **Compact Representation**: Combines all modalities into a single feature vector, simplifying downstream processing.

#### Cons of Early Fusion

- **Data Alignment**: Requires synchronized and aligned data from all modalities, which can be challenging in real-world scenarios.
- **High Dimensionality**: Concatenating features from multiple modalities can result in very high-dimensional input, increasing computational complexity.
- **Modality Dependency**: If one modality is missing or noisy, it can negatively impact the fused representation and overall performance.

#### Applications of Early Fusion

- **Multimodal Sentiment Analysis**: Combining text, audio, and video features to predict sentiment.
- **Action Recognition**: Using video and audio features to classify actions in videos.
- **Medical Diagnosis**: Integrating imaging, clinical notes, and lab results for disease prediction.

#### Comparison with Other Fusion Techniques

| Fusion Technique        | Description                                                                    | Pros                                      | Cons                                              |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------- |
| **Early Fusion**        | Combines raw data or features from all modalities before feeding into a model. | Captures cross-modality interactions      | Requires aligned data; computationally expensive. |
| **Intermediate Fusion** | Combines intermediate features from modality-specific models.                  | Balances interaction and independence     | Requires careful feature alignment.               |
| **Late Fusion**         | Combines predictions from modality-specific models.                            | Modality independence; robust to failures | Loses cross-modality interactions.                |

#### Practical Implementation

```python
import torch
import torch.nn as nn

# Example feature tensors from three modalities
video_features = torch.rand(1, 512)  # Video features (batch_size, feature_dim)
text_features = torch.rand(1, 768)   # Text features (batch_size, feature_dim)
audio_features = torch.rand(1, 256)  # Audio features (batch_size, feature_dim)

# Concatenate features for early fusion
fused_features = torch.cat((video_features, text_features, audio_features), dim=1)  # Shape: (1, 1536)

# Unified model
class UnifiedModel(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(UnifiedModel, self).__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.fc(x)

model = UnifiedModel(input_dim=1536, output_dim=3)  # Example: 3 output classes
output = model(fused_features)
print("Model Output:", output)
```

### Intermediate Fusion Technique

Intermediate fusion, also known as feature-level fusion at intermediate layers, is a technique in machine learning where features from multiple modalities (e.g., video, text, audio) are extracted independently and then combined at an intermediate stage of the model pipeline. This approach allows the model to learn modality-specific representations before integrating them for joint processing.

#### How It Works

1. **Modality-Specific Feature Extraction**: Each modality is processed independently using specialized encoders (e.g., ResNet for video, BERT for text, CNN for audio).
2. **Feature Fusion**: The extracted features are combined at an intermediate layer using techniques like concatenation, attention mechanisms, or cross-modal transformers.
3. **Unified Processing**: The fused features are passed through a shared model for final predictions.

#### Example Workflow

1. **Video Encoder**: Extracts features from video frames.
2. **Text Encoder**: Extracts features from text data.
3. **Audio Encoder**: Extracts features from audio signals.
4. **Fusion Layer**: Combines the features using a fusion mechanism (e.g., concatenation, attention).
5. **Shared Model**: Processes the fused features to make predictions.

#### Pros of Intermediate Fusion

- **Cross-Modality Interactions**: Captures relationships between modalities while preserving modality-specific features.
- **Flexibility**: Allows the use of specialized encoders for each modality, optimizing feature extraction.
- **Balanced Complexity**: Combines the benefits of early and late fusion, offering a trade-off between interaction and independence.

#### Cons of Intermediate Fusion

- **Alignment Challenges**: Requires careful alignment of features from different modalities, especially when they have different temporal or spatial resolutions.
- **Increased Complexity**: The fusion mechanism and shared model add computational overhead compared to late fusion.
- **Dependency on Encoders**: Performance depends on the quality of modality-specific encoders.

#### Applications of Intermediate Fusion

- **Multimodal Sentiment Analysis**: Combining text, audio, and video features at an intermediate layer for sentiment prediction.
- **Action Recognition**: Integrating video and audio features to classify actions in videos.
- **Medical Diagnosis**: Fusing imaging and clinical data for disease prediction.

#### Comparison with Other Fusion Techniques

| Fusion Technique        | Description                                                                    | Pros                                      | Cons                                              |
| ----------------------- | ------------------------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------- |
| **Early Fusion**        | Combines raw data or features from all modalities before feeding into a model. | Captures cross-modality interactions      | Requires aligned data; computationally expensive. |
| **Intermediate Fusion** | Combines intermediate features from modality-specific models.                  | Balances interaction and independence     | Requires careful feature alignment.               |
| **Late Fusion**         | Combines predictions from modality-specific models.                            | Modality independence; robust to failures | Loses cross-modality interactions.                |

#### Practical Implementation

```python
import torch
import torch.nn as nn

# Example feature tensors from three modalities
video_features = torch.rand(1, 512)  # Video features (batch_size, feature_dim)
text_features = torch.rand(1, 768)   # Text features (batch_size, feature_dim)
audio_features = torch.rand(1, 256)  # Audio features (batch_size, feature_dim)

# Fusion layer (e.g., concatenation)
fused_features = torch.cat((video_features, text_features, audio_features), dim=1)  # Shape: (1, 1536)

# Shared model
class SharedModel(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(SharedModel, self).__init__()
    self.fc1 = nn.Linear(input_dim, 512)
    self.fc2 = nn.Linear(512, output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    return self.fc2(x)

model = SharedModel(input_dim=1536, output_dim=3)  # Example: 3 output classes
output = model(fused_features)
print("Model Output:", output)
```

#### Advanced Fusion Mechanisms

1. **Attention Mechanisms**: Use attention layers to weigh the importance of each modality dynamically.
2. **Cross-Modal Transformers**: Employ transformers to model interactions between modalities explicitly.
3. **Graph Neural Networks (GNNs)**: Represent modalities as nodes in a graph and learn their relationships.

> For our model we will use Late Fusion Technique due to its simple implementation and robustness.

## Multimodal Archtecture Details

The model processes inputs from three different modalities:

- **Video Encoder**: ResNet3D 18-layer model to extract video features.
- **Text Encoder**: BERT model to extract textual features.
- **Audio Encoder**: Raw spectrogram processing to extract audio features.

#### Data Processing

Each encoder outputs a feature representation of size `[batch_size, 128]`, which are concatenated to form a unified representation of size `[batch_size, 384]`.

### Fusion Layer

A fusion layer learns relationships between the modalities, helping the model make predictions based on combined input features.

## Task-Specific Classification Heads

The model makes separate predictions for two tasks:

- **Emotion Classification**

  - Output shape: `[batch_size, 7]`
  - Predicts one of 7 emotions (e.g., joy, sadness, etc.)

- **Sentiment Classification**
  - Output shape: `[batch_size, 3]`
  - Predicts sentiment as negative, neutral, or positive.

## Steps Followed For Modelling

- Dataset Classification
- Encoders
- Fusion
- Training
- Evaluation
- Deployment
