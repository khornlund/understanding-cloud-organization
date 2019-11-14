import timm
from segmentation_models_pytorch import *  # noqa
from efficientnet_pytorch import EfficientNet as EffNet


def EfficientNet(encoder_name, classes):
    return EffNet.from_pretrained(encoder_name, classes)


def TIMM(encoder_name, classes=4, pretrained=True):
    return timm.create_model(encoder_name, pretrained=pretrained, num_classes=classes)
