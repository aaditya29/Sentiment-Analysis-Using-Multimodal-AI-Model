from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
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


class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # initializing the encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # building fusion layers
        """
        Here we are combining the features from all three modalities (text, video, and audio) into a single representation
        128 * 3 = 384 makes the 3 encoders each output 128-dim vectors.
        nn.Linear(384, 256) compresses this into a 256-dim fused representation.
        BatchNorm1d(256) normalizes values across the batch for stability.
        ReLU() adds non-linearity, allowing the model to learn complex relationships.
        Dropout(0.3) randomly drops 30% of values during training to avoid overfitting.
        """
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3))

        # Building classification heads of emotional and sentiment classification
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # sadness and anger
        )

        self.sentiment_classifer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # positive, negative and neutral
        )

    # Defining forward pass
    def forward(self, text_inputs, video_frames, audio_features):
        # extracting features from each modality
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(
            video_frames)  # declaring the video features
        audio_features = self.audio_encoder(
            audio_features)  # declaring the audio features

        # concatenating the features from all three modalities along the feature dimension [batch_size, 128*3]
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1)

        # passing the combined vector through the fusion layer
        fused_features = self.fusion_layer(combined_features)
        # Outputs shape: [batch_size, 7]
        emotion_output = self.emotion_classifier(fused_features)
        # Outputs shape: [batch_size, 3]
        sentiment_output = self.sentiment_classifer(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }


class MultiModalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model  # initializing the model
        self.train_loader = train_loader  # initializing the training data loader
        self.val_loader = val_loader  # initializing the validation data loader

        # Logging dataset size
        # getting the size of the training dataset
        train_size = len(train_loader.dataset)
        # getting the size of the validation dataset
        val_size = len(val_loader.dataset)
        print("\n Dataset sizes: ")
        print(f"Training samples: {train_size, }")
        print(f"Validation samples: {val_size, }")
        print(f"Batches per epoch: {len(train_loader):, }")

        timestamp = datetime.now().strftime(
            '%b%d_%H-%M-%S')  # getting the current timestamp
        # setting the base directory for TensorBoard logs
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        # creating a directory for the current run
        log_dir = f"{base_dir}/run_{timestamp}"
        # creating the directory if it doesn't exist
        self.writer = SummaryWriter(log_dir=log_dir)
        # initializing a counter for how many training steps have happened.
        self.global_step = 0

        # Very high: 1, high: 0.1-0.01, medium: 1e-1, low: 1e-4, very low: 1e-5
        """
        Here we are defining the Adam optimizer and the learning rate scheduler.
        We are defining different learning rates for different parts of the model.
        1. Text encoder: 8e-6 
        2. Video encoder: 8e-5
        3. Audio encoder: 8e-5
        4. Fusion layer: 5e-4
        5. Emotion classifier: 5e-4
        6. Sentiment classifier: 5e-4
        Text encoder gets a small learning rate (since it's probably a pretrained BERT-like model)
        Fusion layer and classifiers get higher rates (since they are trained from scratch)
        weight_decay helps prevent overfitting by slightly penalizing large weights.
        """
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)

        """
        Here we are defining the learning rate scheduler.
        The ReduceLROnPlateau scheduler reduces the learning rate when a metric has stopped improving.
        It monitors the validation loss and reduces the learning rate by a factor of 0.1 if the loss does not improve for 2 epochs.
        This helps the model to converge better and avoid overfitting.
        """
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2)

        self.current_train_losses = None  # initializing the current training losses
        """
        Here we are defining the loss functions for emotion and sentiment classification.
        CrossEntropyLoss is used for multi-class classification (emotions/sentiments)
        label_smoothing=0.05: helps regularize the model by softening the labels (prevents overconfidence)
        """
        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=self.emotion_weights)
        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=self.sentiment_weights)

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":  # logging training metrics
            self.current_train_losses = losses  # updating the current training losses
        else:  # validation metrics
            self.writer.add_scalar(
                'loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar(
                'loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar(
                'loss/emotion/val', losses['emotion'], self.global_step)

            self.writer.add_scalar(
                'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar(
                'loss/sentiment/val', losses['sentiment'], self.global_step)

        if metrics:
            elf.writer.add_scalar(
                f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step)
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step)
            self.writer.add_scalar(
                f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step)
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step)

    def train_epoch(self):
        self.model.train()  # switching the model to training mode
        # initializing the running loss dictionary to track the losses
        running_loss = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

        # iterating over the training data loader
        for batch in self.train_loader:
            # moving the batch data to the same device as the model
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']
                ['input_ids'].to(device),
                'attention_mask': batch['text_inputs']
                ['attention_mask'].to(device)
            }
            # moving the video frames, audio features, and labels to the same device as the model
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            # zero_grad() clears old gradients
            self.optimizer.zero_grad()

            # passing the inputs through the model
            outputs = self.model(text_inputs, video_frames, audio_features)

            # calculating the loss using the raw logits
            emotion_loss = self.emotion_criterion(
                outputs['emotions'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs['sentiments'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

            """
            .backward() computes gradients
            clip_grad_norm_() is used to prevent exploding gradients
            step() updates model weights
            """
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # tracking losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

            # logging the losses to TensorBoard
            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })
            self.global_step += 1  # incrementing the global step

        # dividing the running loss by the number of batches to get the average loss
        return {k: v/len(self.train_loader) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                outputs = self.model(text_inputs, video_frames, audio_features)

                emotion_loss = self.emotion_criterion(
                    outputs["emotions"], emotion_labels)
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiments"], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss

                all_emotion_preds.extend(
                    outputs["emotions"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs["sentiments"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

        avg_loss = {k: v/len(data_loader) for k, v in losses.items()}

        # Compute the precision and accuracy
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_accuracy = accuracy_score(
            all_emotion_labels, all_emotion_preds)
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_accuracy = accuracy_score(
            all_sentiment_labels, all_sentiment_preds)

        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }, phase=phase)

        if phase == "val":
            self.scheduler.step(avg_loss['total'])

        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }


if __name__ == "__main__":
    dataset = MELDDataset(
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/train/train_sent_emo.csv',
        '/Users/adityamishra/Documents/AI-Sentiment-Analyser/dataset.Raw/train/train_splits')

    sample = dataset[0]  # getting the first sample

    model = MultiModalSentimentModel()  # initializing the model
    model.eval()  # setting the model to evaluation mode

    # dictionary containing result sample
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    # unsqueeze to add batch dimension
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    # loading the model weights
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
    }

    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }

    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.4f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Predictions for utterance: ")
