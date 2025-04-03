from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import numpy as np
import torch
import cv2


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

    def __len__(self):
        return len(self.data)  # return the length of the dataset

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # get the row at the given index
        video_filename = f"""dia{row['Dialogue_ID']}_utt{
            row['Utterance_ID']}.mp4"""  # get the video filename from the row with dialogue ID and utterance ID

        # get the path of the video
        path = os.path.join(self.video_dir, video_filename)
        video_path_exists = os.path.exists(path)  # check if the video exists

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

        video_frames = self._load_video_frames(path)  # load the video frames
        print(video_frames)


if __name__ == "__main__":
    meld = MELDDataset('/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_sent_emo.csv',
                       '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_splits_complete')
    print(meld[0])
