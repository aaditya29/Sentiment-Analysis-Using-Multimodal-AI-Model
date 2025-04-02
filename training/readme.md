# Training Our Data

We are going to train our data in the:

- csv data
- video

## Dataset Class Architecture

The Dataset Class is designed to load preprocessed samples efficiently. It extracts and processes text, video, and audio features from the given dataset, preparing it for use in training deep learning models.

### Overview

The Dataset Class integrates various data sources, including:

- **CSV files** containing metadata and labels
- **Video files** containing visual and audio features

It processes each data sample to extract and normalize relevant features, making them suitable for model training.

---

#### 1. CSV File Path

The dataset loads a CSV file, where each row contains:

- **Utterance text** (spoken content)
- **Speaker information**
- **Session and episode details**
- **Start and end time of video clips**
- **Emotion and sentiment labels**

Example CSV row:

```csv
ID, Utterance, Speaker, StartTime, EndTime, Emotion, Sentiment
1, "Hello, how are you?", SpeakerA, 00:12:34, 00:12:36, Joy, Positive
```

#### 2. Video File Path

Video files are used to extract both visual and audio features. The Dataset Class supports multiple video formats, such as:

- **.mp4**
- **.avi**
- **.m5**

Example file structure:

```
/videos
    ├── video1.mp4
    ├── video2.mp4
    ├── video3.m5
```

---

### Data Processing Pipeline

#### 1. **Loading CSV into the Class**

The CSV file is read into a class instance, mapping labels and metadata for easy access.

#### 2. **Storing Mappings**

Mappings are stored for:

- **Sentiment labels** (e.g., Neutral → 1, Positive → 2)
- **Emotion labels** (e.g., Anger → 0, Joy → 3, Sadness → 5)

#### 3. **Extracting Features**

Once a sample is selected, multiple feature extraction steps occur:

#### Text Features

- **Tokenize the utterance** from the dataset

#### Video Features

- **Locate the corresponding video file**
- **Extract frames** from the video
- **Normalize RGB values** for better processing
- **Store all frames as a tensor**

#### Audio Features

- **Extract audio from video**
- **Convert .mp4 to .wav**
- **Create Mel spectrogram** (efficient representation for deep learning models)
- **Normalize Mel spectrogram values**

#### Labels

- **Retrieve labels from the dataset row**
- **Map emotion and sentiment to numerical values**

#### 4. **Returning the Processed Data**

The final processed data sample is structured as follows:

```json
{
  "text": "Hello, how are you?",
  "video_frames": tensor_data,
  "audio_features": mel_spectrogram,
  "emotion_label": "sadness",
  "sentiment_label": "negative"
}
```

---

### Implementing Dataset Class in Python

#### Magic Methods

The class includes Python magic methods for easy interaction:

1. **`__len__`**: Returns the number of rows in the dataset.
   ```python
   len(my_dataset)  # Returns total number of rows in dataset
   ```
2. **`__getitem__`**: Retrieves a specific row as a processed sample.
   ```python
   my_dataset[0]  # Returns the first preprocessed sample
   ```

---

### Key Concepts

#### 1. Normalizing RGB Channels

- Standard activation functions work better with normalized values.

#### 2. Mel Spectrogram

- Works well with Convolutional Neural Networks (CNNs).
- More compact than waveform data, making it more efficient for training.

---
