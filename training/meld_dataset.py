from torch.utils.data import Dataset
import pandas as pd


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)  # read the csv file
        print(len(self.data))


if __name__ == "__main__":
    meld = MELDDataset('/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_sent_emo.csv',
                       '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/dev/dev_splits_complete')
