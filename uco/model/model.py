from segmentation_models_pytorch import *  # noqa
from efficientnet_pytorch import EfficientNet as EffNet


def EfficientNet(model_name, classes):
    return EffNet.from_pretrained(model_name, classes)
