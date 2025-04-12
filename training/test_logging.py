from models import MultiModalSentimentModel, MultiModalTrainer
from torch.utils.data import DataLoader
from collections import namedtuple
import torch


def test_logging():
    # Creates a custom lightweight data structure (like a class) called Batch with 3 fields i.e. text_inputs: inputs for the text encoder video input features, audio_features foraudio input features
    Batch = namedtuple('Batch', ['text_inputs', 'video_frames',
                       'audio_features'])
    # Creates a mock batch of data with dummy values of `` text inputs, video frames, and audio features
    mock_batch = Batch(text_inputs={'input_ids': torch.ones(1), 'attention_mask': torch.ones(1)},
                       video_frames=torch.ones(1),
                       audio_features=torch.ones(1))
    # Creating a DataLoader with the mock batch. This simulates a data loader that would be used in training or validation
    mock_loader = DataLoader([mock_batch])
    # Instantiating the MultimodalSentimentModel and MultimodalTrainer classes
    model = MultiModalSentimentModel()
    # The MultimodalTrainer is initialized with the model and the mock data loaders for training and validation
    trainer = MultiModalTrainer(model, mock_loader, mock_loader)
    # The trainer is used to log metrics during training and validation
    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
    }
    # Logging training losses
    trainer.log_metrics(train_losses, phase="train")
    # Logging validation losses and metrics
    val_losses = {
        'total': 1.5,
        'emotion': 0.5,
        'sentiment': 1.0
    }
    # Logging validation losses
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95
    }
    # Logging validation metrics
    trainer.log_metrics(val_losses, val_metrics, phase="val")


if __name__ == "__main__":
    test_logging()
