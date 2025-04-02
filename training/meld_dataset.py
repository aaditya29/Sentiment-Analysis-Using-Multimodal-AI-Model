from torch.utils.data import Dataset


class MELDDataset(Dataset):
    def __init__(self):
        print("Hello! I am a MELD dataset.")


if __name__ == "__main__":
    meld = MELDDataset()
