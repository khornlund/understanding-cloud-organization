import abc

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .process import IMAGE_ORIGINAL_HEIGHT, IMAGE_ORIGINAL_WIDTH  # noqa


# divisible by 2^5
IMAGE_ROUNDED_HEIGHT = 1376
IMAGE_ROUNDED_WIDTH  = 2080


class AugmentationFactoryBase(abc.ABC):

    def build(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class LightTransforms(AugmentationFactoryBase):

    # training distribution stats
    MEANS = [0.2606, 0.2786, 0.3266]
    STDS  = [0.0643, 0.0621, 0.0552]

    def __init__(self, height, width):
        self.H = height
        self.W = width

    def build_train(self):
        return A.Compose([
            A.RandomCrop(self.H, self.W),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])

    def build_test(self):
        return A.Compose([
            A.RandomCrop(IMAGE_ROUNDED_HEIGHT, IMAGE_ROUNDED_WIDTH),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])
