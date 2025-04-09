from sklearn.metrics import precision_score, accuracy_score
from torchvision import models as vision_models
from meld_dataset import MELDDataset
from transformers import BertModel
from datetime import datetime
import torch.nn as nn
import torch
import os


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # here we are loading the BERT model if it is not already loaded
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False  # freeze the BERT model parameters to non-traianable

        # projecting the BERT output to a lower dimension
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extracting BERT features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs.pooler_output  # the output of the last layer of BERT

        return self.projection(pooler_output)  # project to lower dimension


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # here we are loading the pretrained 3D ResNet model and installing it in the instance property called backbone.
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # setting all the parameters to not being able to trainable
        for param in self.backbone.parameters():
            param.requires_grad = False

        # getting the number of features in the last layer
        # accessing the number of input features of the model's final fully-connected layer (fc)
        # necessary to replace it with our own layer because we will want fewer or more compact features
        num_fts = self.backbone.fc.in_features
        """
        replacing the last fully-connected layer with a new one with of the pretrained ResNet of our own.
        1. nn.Linear(num_fts, 128): Projects the high-dimensional features into a 128-dimensional vector.
        2. nn.ReLU(): Applies a non-linear activation so the model can learn more complex relationships.

        3. nn.Dropout(0.2): Prevents overfitting by randomly setting 20% of neurons to zero during training.
        """
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    # Defining how our input data flows through the model
    def forward(self, x):
        # [batch_size, frames, channels, height, width]->[batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)  # transposing time and channel axes
        return self.backbone(x)
