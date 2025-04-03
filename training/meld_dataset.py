from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os


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

    def __len__(self):
        return len(self.data)  # return the length of the dataset

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # get the row at the given index
        video_filename = f"""dia{row['Dialogue_ID']}_utt{
            row['Utterance_ID']}.mp4"""  # get the video filename from the row with dialogue ID and utterance ID

        # get the path of the video
        path = os.path.join(self.video_dir, video_filename)
        video_path = os.path.exists(path)  # check if the video exists

        if video_path == False:
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
        print(text_inputs)


if __name__ == "__main__":
    meld = MELDDataset('/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_sent_emo.csv',
                       '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_splits_complete')
    print(meld[0])
