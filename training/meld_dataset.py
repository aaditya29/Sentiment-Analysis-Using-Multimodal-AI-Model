from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.utils.data.dataloader
import numpy as np
import pandas as pd
import torchaudio
import subprocess
import torch
import cv2
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)  # read the csv file

        self.video_dir = video_dir  # directory containing the videos

        self.tokenizer = AutoTokenizer.from_pretrained(
            # loading the tokenizer which allows us to use the BERT model and convert the text to tokens
            "bert-base-uncased")

        # adding emotion labels to the tokenizer
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        # adding sentiment labels to the tokenizer
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }

    def _load_video_frames(self, video_path):
        # load the video frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []  # array to store the frames

        try:
            if not cap.isOpened():  # check if the video is opened
                raise ValueError(
                    f"Could not open video file at {video_path}")
            # Using Try and reading first frame to validate the video
            ret, frame = cap.read()  # read the first frame
            if not ret or frame is None:
                raise ValueError(f"Video not found at {video_path}")

            # Resetting index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # resetting the index to 0

            # reading the frame of first second only
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # resize the frame to 224x224
                frame = cv2.resize(frame, (224, 224))
                frame = frame/255.0  # normalizing the frame
                frames.append(frame)  # appending the frame to the list

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()  # releasing the video capture object

        if (len(frames) == 0):
            raise ValueError(f"No frames could be extracted")

        # Padding or truncating frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * \
                (30 - len(frames))  # padding with zeros
        else:
            frames = frames[:30]  # truncating to 30 frames

        """
        converting to tensor and changing the order of dimensions and returning the tensor
        here permute rearranges the original tensor according to the desired ordering and returns a new multidimensional rotated tensor.
        Before Permute: [frames, height, width, channels]
        After Permute: [frames, channels, height, width]
        """
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        # replacing the video extension with audio extension
        audio_path = video_path.replace('.mp4', '.wav')

        """
        Here `subprocess.run([...])` executes the specified command in a new process. 
        The command is passed as a list of strings, where each element corresponds to a part of the command.
        'ffmpeg' is the command-line tool being invoked. ffmpeg is a popular multimedia framework for processing video and audio files.
        '-i', video_path:-i specifies the input file, and video_path is the path to the video file from which audio will be extracted.
        'vn' flag tells ffmpeg to ignore the video stream and only process the audio stream.
        '-acodec', 'pcm_s16le' specifies the audio codec to use. pcm_s16le is a raw, uncompressed audio format with 16-bit samples and little-endian byte order.
        '-ar', '16000': this sets the audio sample rate to 16,000 Hz (16 kHz) which is commonly used for speech processing tasks.
        '-ac', '1': sets the number of audio channels to 1 (mono audio).
        audio_path: specifies the output file path where the extracted audio will be saved.

        Further

        check=True ensures that an exception (subprocess.CalledProcessError) is raised if the ffmpeg command fails (i.e., returns a non-zero exit code).
        stdout=subprocess.DEVNULL redirects the standard output (stdout) of the ffmpeg command to DEVNULL, effectively silencing any output from the command.
        stderr=subprocess.DEVNULL redirects the standard error (stderr) of the ffmpeg command to DEVNULL, silencing any error messages.
        """
        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',  # no video
                '-acodec', 'pcm_s16le',  # audio codec
                '-ar', '16000',  # audio sample rate
                '-ac', '1',  # number of audio channels
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(
                audio_path)  # loading the audio file

            if sample_rate != 16000:
                resampler = torch.transform.Resample(
                    sample_rate, 16000)  # resampling the audio
                # here we are resampling the audio by changing the sample rate
                waveform = resampler(waveform)

            # using mel_spectrogram to measure  represent a signal's loudness, or amplitude, as it varies over time at different frequencies.
            # creating the mel spectrogram with the given parameters where sample_rate is 16000,
            # n_mels(number of mel filterbanks) is 64, n_fft is 1024 and
            # hop_length(Length of hop between short time fourier transform STFT window) is 512
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512

            )

            # applying the mel spectrogram to the waveform
            mel_spec = mel_spectrogram(waveform)

            # normalizing the mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)  # padding the mel spectrogram
                # padding the mel spectrogram where 0 is the left padding and padding is the right padding
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                # truncating the mel spectrogram to 300
                mel_spec = mel_spec[:, :, :300]
            return mel_spec  # returning the mel spectrogram

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")

        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)  # remove the audio file after processing

    def __len__(self):
        return len(self.data)  # return the length of the dataset

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):  # checking if the index is a tensor
            idx = idx.item()  # converting the tensor to an integer
        row = self.data.iloc[idx]  # get the row at the given index
        try:
            video_filename = f"""dia{row['Dialogue_ID']}_utt{
                row['Utterance_ID']}.mp4"""  # get the video filename from the row with dialogue ID and utterance ID

            # get the path of the video
            path = os.path.join(self.video_dir, video_filename)
            video_path_exists = os.path.exists(
                path)  # check if the video exists

            if video_path_exists == False:
                raise FileNotFoundError(f"Video file at {path} not found")

            """
            Here 'self.tokenizer' is referencing the AutoTokenizer class from the transformers library.
            'utterance' is a column in the dataframe that contains the text data we want to tokenize.
            'padding' is set to 'max_length' to ensure all sequences are of the same length.
            'truncation' is set to True to truncate sequences longer than the max length.
            'max_length' is set to 128, which is the maximum length of the sequences.
            'return_tensors' is set to 'pt' to return PyTorch tensors.
            """
            text_inputs = self.tokenizer(row['Utterance'],
                                         padding='max_length', truncation=True, max_length=128,
                                         return_tensors='pt')

            video_frames = self._load_video_frames(
                path)  # load the video frames
            # print(video_frames)
            audio_features = self._extract_audio_features(path)
            # print(audio_features)

            # Mapping sentiment and emotion labels to the dataset
            # mapping the emotion label to the dataset
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            # mapping the sentiment label to the dataset
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }  # returning the dictionary containing the text inputs, video frames, audio features, emotion label and sentiment label
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None


"""
We are making collate_fn which is an optional argument passed to a DataLoader. 
It tells PyTorch how to combine a list of samples from  Dataset into a single batch.

By default, PyTorch tries to batch the samples using default_collate, which stacks tensors 
along the first dimension (i.e., creates a batch). But if data samples are:
- having different shapes
- if they are not all tensors
- require custom logic(like filtering out None valuesor padding)
then we need to define a custom collate function.
If we consider __getitem__() method returns one sample (like a dict of tensors for text/audio/video). 
The DataLoader collects multiple such samples then it sends them to collate_fn and then it receives
one big batch dict (batched tensors, padded as needed).
"""


def collate_fn(batch):
    # Filtering out the None values from the batch
    batch = list(filter(None, batch))
    # returning the collated batch
    return torch.utils.data.dataloader.default_collate(batch)


"""
making function to prepare the dataloaders where train_csv is the path to the train csv file,
train_video_dir is the path to the train video directory, dev_csv is the path to the dev csv file, 
dev_video_dir is the path to the dev video directory, test_csv is the path to the test csv file,
test_video_dir is the path to the test video directory and batch_size is the batch size
"""


def prepare_dataloaders(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
    # creating the train dataset
    train_dataset = MELDDataset(train_csv, train_video_dir)
    # creating the dev dataset
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    # creating the test dataset
    # creating the test dataset
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(
        # creating the train dataloader
        train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    dev_loader = DataLoader(
        # creating the dev dataloader
        dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # creating the test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/train/train_sent_emo.csv',
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/train/train_splits',
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_sent_emo.csv',
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_splits_complete',
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/test/test_sent_emo.csv',
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/test/output_repeated_splits_test')

    for batch in train_loader:  # iterating through the train dataloader
        print(batch['text_inputs'])  # printing the text inputs
        # printing the shape of the video frames
        print(batch['video_frames'].shape)
        # printing the shape of the audio features
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])  # printing the emotion label
        print(batch['sentiment_label'])  # printing the sentiment label
        break
