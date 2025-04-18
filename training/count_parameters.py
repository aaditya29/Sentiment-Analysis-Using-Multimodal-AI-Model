from models import MultiModalSentimentModel


# storing the parameter count for each named component in the model
def count_parameters(model):
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }

    # counting the total number of parameters in the model
    total_params = 0
    # iterating through each named parameter in the model
    for name, param in model.named_parameters():
        if param.requires_grad:  # check if the parameter is trainable
            param_count = param.numel()  # get the number of elements in the parameter tensor
            total_params += param_count  # add to the total parameter count

            # checking which component the parameter belongs to and adding to the respective count
            if 'text_encoder' in name:
                params_dict['text_encoder'] += param_count
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += param_count
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += param_count
            elif 'fusion_layer' in name:
                params_dict['fusion_layer'] += param_count
            elif 'emotion_classifier' in name:
                params_dict['emotion_classifier'] += param_count
            elif 'sentiment_classifier' in name:
                params_dict['sentiment_classifier'] += param_count

    # returning the dictionary with parameter counts for each component and the total parameter count
    return params_dict, total_params


if __name__ == "__main__":
    model = MultiModalSentimentModel()
    param_dics, total_params = count_parameters(model)

    print("Parameter count by component")
    for component, count in param_dics.items():
        print(f"{component:20s}: {count:,} parameters")

    print("\nTotal trainable parameters", f"{total_params:,}")
