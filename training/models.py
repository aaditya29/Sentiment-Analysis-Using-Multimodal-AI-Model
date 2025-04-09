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


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        """
        Defining a sequence of layers that extract temporal patterns from audio using 1D convolutions.
        1. nn.Conv1d(64, 64, kernel_size=3): Applies a 1D convolution with 64 input and output channels.
        2. nn.BatchNorm1d(64): Normalizes the output of the convolution to stabilize training.
        3. nn.ReLU(): Applies a non-linear activation function to introduce non-linearity.
        4. nn.MaxPool1d(2): Reduces the dimensionality of the output by taking the maximum value in each 2-element window.
        5. nn.Conv1d(64, 128, kernel_size=3): Applies another 1D convolution with 128 output channels.
        6. nn.BatchNorm1d(128): Normalizes the output of the second convolution.
        7. nn.ReLU(): Applies another non-linear activation function.
        8. nn.AdaptiveAvgPool1d(1): Reduces the output to a fixed size of 1 for each channel, regardless of the input size.
        """
        self.conv_layers = nn.Sequential(
            # writing lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # writing higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        """
        Here we project the 128D audio embedding into a final form.

        Linear(128, 128) maps features to same size.
        ReLU() allows non-linearity.
        Dropout(0.2) randomly zeroes 20% of neurons during training to avoid overfitting.
        """
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.squeeze(1)  # removing the channel dimension by squeezing it

        # features output: [batch_size, 128, 1]
        # passing the input through conv layers and pooling so that each example has 128 channels of learned features across time (shrunk to 1 step)
        features = self.conv_layers(x)
        # squeezing out the last dimension to the 1 time step so it's [batch_size, 128]
        return self.projection(features.squeeze(-1))
