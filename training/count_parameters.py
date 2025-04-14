from models import MultiModalSentimentModel


def count_parameters(model):
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }
