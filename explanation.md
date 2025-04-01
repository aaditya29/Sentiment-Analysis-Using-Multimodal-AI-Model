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
