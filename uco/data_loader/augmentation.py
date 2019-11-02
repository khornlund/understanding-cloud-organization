import abc
import random

import albumentations as A
from albumentations.augmentations import functional as F
from albumentations.pytorch import ToTensorV2

from .process import IMAGE_ORIGINAL_HEIGHT, IMAGE_ORIGINAL_WIDTH  # noqa


# divisible by 2^5
IMAGE_ROUNDED_HEIGHT = 1376
IMAGE_ROUNDED_WIDTH = 2080

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
    STDS = [0.0643, 0.0621, 0.0552]

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def build_train(self):
        return A.Compose(
            [
                A.RandomCrop(self.height, self.width),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )

    def build_test(self):
        return A.Compose(
            [
                A.RandomCrop(IMAGE_ROUNDED_HEIGHT, IMAGE_ROUNDED_WIDTH),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class LightRandomResizeTransforms(NormalizeTransforms):
    def __init__(self, height, width):
        super().__init__(height, width)

    def build_train(self):
        return A.Compose(
            [
                A.RandomResizedCrop(
                    self.height, self.width, scale=(0.5, 0.8), ratio=(1.354, 1.554)
                ),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )

    def build_test(self):
        return A.Compose(
            [
                A.Resize(self.height, self.width),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class MediumResizeTransforms(NormalizeTransforms):
    def __init__(self, height, width):
        super().__init__(height, width)

    def build_train(self):
        return A.Compose(
            [
                A.Flip(p=0.55),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0),
                A.Resize(self.height, self.width),
                A.RandomBrightness(),
                A.RandomContrast(),
                A.OneOf(
                    [
                        A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                        A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
                        A.CoarseDropout(max_holes=8, max_height=32, max_width=32),
                    ],
                    p=0.5,
                ),
                A.OneOf([A.IAAPerspective(), A.IAAPiecewiseAffine()]),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )

    def build_test(self):
        return A.Compose(
            [
                A.Resize(self.height, self.width),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class HeavyResizeTransforms(NormalizeTransforms):
    def __init__(self, height, width):
        super().__init__(height, width)

    def build_train(self):
        return A.Compose(
            [
                A.Flip(p=0.55),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, border_mode=0),
                A.Resize(self.height, self.width),
                A.RandomBrightness(),
                A.RandomContrast(),
                A.OneOf(
                    [
                        # broken for masks
                        A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                        A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
                    ],
                    p=0.0,
                ),
                A.OneOf(
                    [
                        A.IAAPerspective(),
                        A.IAAPiecewiseAffine(),
                        A.GridDistortion(),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ],
                    p=0.5,
                ),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )

    def build_test(self):
        return A.Compose(
            [
                A.Resize(self.height, self.width),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class RandomResizeCropTransforms(NormalizeTransforms):
    """
    Used as a base to build other transforms.
    """

    def __init__(self, height, width, scale=(0.8, 1.0), ratio=(0.75, 1.33)):
        self.crop_h = height
        self.crop_w = width
        self.scale = scale
        self.ratio = ratio

    def build_train(self):
        return A.Compose(
            [A.RandomResizedCrop(self.crop_h, self.crop_w, self.scale, self.ratio)]
        )

    def build_test(self):
        return A.Compose(
            [
                A.Resize(self.crop_h, self.crop_w),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class HeavyResizeRandomCropTransforms(RandomResizeCropTransforms):
    def build_train(self):
        return A.Compose(
            [
                super().build_train(),
                A.Flip(p=0.6),
                A.RandomBrightness(),
                A.RandomContrast(),
                A.OneOf(
                    [
                        A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                        A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
                    ],
                    p=0.0,
                ),
                A.OneOf(
                    [
                        A.IAAPerspective(),
                        A.IAAPiecewiseAffine(),
                        A.GridDistortion(),
                        A.OpticalDistortion(),
                    ],
                    p=0.5,
                ),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


# -- custom augmentations -------------------------------------------------------------


class DualCoarseDropout(A.DualTransform):
    """
    Applies cutout to img and mask.
    """

    def __init__(
        self,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        assert 0 < self.min_holes <= self.max_holes
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width

    def apply(self, image, fill_value=0, holes=[], **params):
        print(f"Applying {len(holes)} holes to img")
        return F.cutout(image, holes, fill_value)

    def apply_to_mask(self, image, fill_value=0, holes=[], **params):
        print(f"Applying {len(holes)} holes to mask")
        return F.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
        )
