from models import MultimodalSentimentModel
import torch
import os


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Checking if GPU is available
    model = MultimodalSentimentModel().to(device)  # Load the model

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model file not found in path " + model_path)
