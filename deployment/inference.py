from models import MultimodalSentimentModel
from transformers import AutoTokenizer
import numpy as np
import torch
import cv2
import os


class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Padding or truncating frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


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
