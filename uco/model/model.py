from segmentation_models_pytorch import *  # noqa
from efficientnet_pytorch import EfficientNet as EffNet


def EfficientNet(encoder_name, classes):
    return EffNet.from_pretrained(encoder_name, classes)
