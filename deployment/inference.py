from models import MultimodalSentimentModel
import torch


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Checking if GPU is available
    model = MultimodalSentimentModel().to(device)  # Load the model
