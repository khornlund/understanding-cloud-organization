import abc

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .process import IMAGE_ORIGINAL_HEIGHT, IMAGE_ORIGINAL_WIDTH  # noqa


# divisible by 2^5
IMAGE_ROUNDED_HEIGHT = 1376
IMAGE_ROUNDED_WIDTH  = 2080

VALID_IMG_SIZES = [
    (64, 96),
    (128, 192),
    (192, 288),
    (256, 384),
    (320, 480),
    (384, 576),
    (448, 672),
    (512, 768),
    (576, 864),
    (640, 960),
]


class AugmentationFactoryBase(abc.ABC):

    def build(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class NormalizeTransforms(AugmentationFactoryBase):

    # training distribution stats
    MEANS = [0.2606, 0.2786, 0.3266]
    STDS  = [0.0643, 0.0621, 0.0552]

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def build_train(self):
        return A.Compose([
            A.RandomCrop(self.height, self.width),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])

    def build_test(self):
        return A.Compose([
            A.RandomCrop(IMAGE_ROUNDED_HEIGHT, IMAGE_ROUNDED_WIDTH),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])


class LightRandomResizeTransforms(NormalizeTransforms):

    def __init__(self, height, width):
        super().__init__(height, width)

    def build_train(self):
        return A.Compose([
            A.RandomResizedCrop(self.height, self.width, scale=(0.5, 0.8), ratio=(1.354, 1.554)),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])

    def build_test(self):
        return A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])


class MediumResizeTransforms(NormalizeTransforms):

    def __init__(self, height, width):
        super().__init__(height, width)

    def build_train(self):
        return A.Compose([
            A.Flip(p=0.55),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0),
            A.Resize(self.height, self.width),
            A.RandomBrightness(),
            A.RandomContrast(),
            A.OneOf([
                A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32),
            ], p=0.5),
            A.OneOf([
                A.IAAPerspective(),
                A.IAAPiecewiseAffine()
            ]),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])

    def build_test(self):
        return A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])


class HeavyResizeTransforms(NormalizeTransforms):

    def __init__(self, height, width):
        super().__init__(height, width)

    def build_train(self):
        return A.Compose([
            A.Flip(p=0.55),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, border_mode=0),
            A.Resize(self.height, self.width),
            A.RandomBrightness(),
            A.RandomContrast(),
            A.OneOf([
                A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
            ], p=0.5),
            A.OneOf([
                A.IAAPerspective(),
                A.IAAPiecewiseAffine(),
                # A.GridDistortion(),
                # A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ]),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])

    def build_test(self):
        return A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])


class ResizeRandomCropTransforms(NormalizeTransforms):
    """
    Resizes to the next largest valid size and then random crops at the desired size.
    Used as a base to build other transforms.
    """

    def __init__(self, height, width):
        self.crop_h = height
        self.crop_w = width
        self.resize_h = height + 64
        self.resize_w = width + 64

    def build_train(self):
        return A.Compose([
            A.Resize(self.resize_h, self.resize_w),
            A.RandomCrop(self.crop_h, self.crop_w)
        ])

    def build_test(self):
        return A.Compose([
            A.Resize(self.resize_h, self.resize_w),
            A.Normalize(self.MEANS, self.STDS),
            ToTensorV2(),
        ])


class HeavyResizeRandomCropTransforms(ResizeRandomCropTransforms):

    def build_train(self):
        return A.Compose([
            super().build_train(),
            A.Flip(p=0.6),
            A.RandomBrightness(),
            A.RandomContrast(),
            A.OneOf([
                A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
            ], p=0.4),
            A.OneOf([
                A.IAAPerspective(),
                A.IAAPiecewiseAffine(),
            ], p=0.4),
        ])
