from models import MultiModalSentimentModel, MultiModalTrainer
from meld_dataset import prepare_dataloaders

from tqdm import tqdm
import torchaudio
import argparse
import torch
import json
import sys
import os

# forAWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get(
    'SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get(
    'SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get(
    'SM_CHANNEL_TEST', "/opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def main():

    # printing the available audio backends
    print("Available audio backends: ")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tracking initial GPU memory if availlable
    if torch.cuda.is_availlable():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

        train_loader, val_loader, test_loader = prepare_dataloaders(
            train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
            train_video_dir=os.path.join(args.train_dir, 'train_splits'),
            dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
            dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
            test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
            test_video_dir=os.path.join(
                args.test_dir, 'output_repeated_splits_test'),
            batch_size=args.batch_size
        )

        print(f"""Training DSV path: {os.path.join(
            args.train_dir, 'train_sent_emo.csv')}""")
        print(f"""Training video directory: {
            os.path.join(args.train_dir, 'train_splits')}""")
        model = MultiModalSentimentModel().to(device)
        trainer = MultiModalTrainer(model, train_loader, val_loader)
        best_val_loss = float('inf')  # best validation loss

        metrics_data = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }

        for epoch in tqdm(range(args.epochs), desc="Epochs"):
            train_loss = trainer.train_epoch()
            val_loss, val_metrics = trainer.evaluate(val_loader)

            # Tracking metrics
            metrics_data["train_losses"].append(train_loss["total"])
            metrics_data["val_losses"].append(val_loss["total"])
            metrics_data["epochs"].append(epoch)

            # Logging metrics in SageMaker format
            print(json.dumps({
                "metrics": [
                    {"Name": "train:loss", "Value": train_loss["total"]},
                    {"Name": "validation:loss", "Value": val_loss["total"]},
                    {"Name": "validation:emotion_precision",
                        "Value": val_metrics["emotion_precision"]},
                    {"Name": "validation:emotion_accuracy",
                        "Value": val_metrics["emotion_accuracy"]},
                    {"Name": "validation:sentiment_precision",
                        "Value": val_metrics["sentiment_precision"]},
                    {"Name": "validation:sentiment_accuracy",
                        "Value": val_metrics["sentiment_accuracy"]},
                ]
            }))
