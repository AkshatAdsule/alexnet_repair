import torch
from model_classes.alexnet import AlexNet
import sytorch as st


def alexnet(state_dict_path=None):
    model = AlexNet(10)
    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
    print(f"model instance: {model.__class__.__name__}")
    model = st.nn.from_torch(model)
    return model
