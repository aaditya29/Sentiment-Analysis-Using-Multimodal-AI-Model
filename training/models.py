from sklearn.metrics import precision_score, accuracy_score
from meld_dataset import MELDDataset
from transformers import BertModel
from datetime import datetime
import torch.nn as nn
import torch
import os


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()  # accessing the parent class constructor using super()
