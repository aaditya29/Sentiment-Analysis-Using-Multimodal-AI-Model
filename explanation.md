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
