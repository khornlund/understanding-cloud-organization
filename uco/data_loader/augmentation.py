import abc
import random

import albumentations as A
from albumentations.augmentations import functional as F
from albumentations.pytorch import ToTensorV2

from .process import IMAGE_ORIGINAL_HEIGHT, IMAGE_ORIGINAL_WIDTH


# image sizes divisible by 32 with same ratio as 1400x2100
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


# -- Base Classes ---------------------------------------------------------------------


class AugmentationFactoryBase(abc.ABC):
    def build(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class NormalizeBase(AugmentationFactoryBase):

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
                A.RandomCrop(IMAGE_ORIGINAL_HEIGHT, IMAGE_ORIGINAL_WIDTH),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class RandomResizeCropBase(NormalizeBase):
    def __init__(
        self,
        height,
        width,
        scale_rrc=(0.7, 1),
        ratio=(0.75, 1.33),
        scale_ssr=0.4,
        rotate=10,
    ):
        self.h = height
        self.w = width
        self.scale_rrc = scale_rrc
        self.ratio = ratio
        self.scale_ssr = scale_ssr
        self.rotate = rotate

    def build_train(self):
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomResizedCrop(self.h, self.w, self.scale_rrc, self.ratio),
                        A.Compose(
                            [
                                A.Resize(self.h, self.w),
                                A.ShiftScaleRotate(
                                    scale_limit=self.scale_ssr,
                                    rotate_limit=self.rotate,
                                    border_mode=0,
                                ),
                            ]
                        ),
                    ],
                    p=1,
                )
            ]
        )

    def build_test(self):
        return A.Compose(
            [A.Resize(self.h, self.w), A.Normalize(self.MEANS, self.STDS), ToTensorV2()]
        )


def CutoutBase(height, width):
    return A.Compose(
        [
            A.OneOf(
                [
                    DualCoarseDropout(
                        max_holes=1,
                        min_height=height // 4,
                        max_height=height // 2,
                        min_width=width // 4,
                        max_width=width // 2,
                        p=1,
                    ),
                    DualCoarseDropout(
                        max_holes=4,
                        min_height=height // 8,
                        max_height=height // 4,
                        min_width=width // 8,
                        max_width=width // 4,
                        p=1,
                    ),
                ],
                p=0.2,
            )
        ]
    )


def DistortionBase():
    return A.Compose(
        [
            A.OneOf(
                [
                    A.IAAPerspective(),
                    A.IAAPiecewiseAffine(),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.50,
            )
        ]
    )


# -- Compositions ---------------------------------------------------------------------


class LightTransforms(RandomResizeCropBase):
    def build_train(self):
        return A.Compose(
            [
                super().build_train(),
                A.Flip(p=0.6),
                A.RandomBrightness(),
                A.RandomContrast(),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class CutoutTransforms(RandomResizeCropBase):
    def build_train(self):
        return A.Compose(
            [
                super().build_train(),
                A.Flip(p=0.6),
                A.RandomBrightness(),
                A.RandomContrast(),
                CutoutBase(self.h, self.w),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class DistortionTransforms(RandomResizeCropBase):
    def build_train(self):
        return A.Compose(
            [
                super().build_train(),
                A.Flip(p=0.6),
                A.RandomBrightness(),
                A.RandomContrast(),
                DistortionBase(),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class CutoutDistortionTransforms(RandomResizeCropBase):
    def build_train(self):
        return A.Compose(
            [
                super().build_train(),
                A.Flip(p=0.6),
                A.RandomBrightness(),
                A.RandomContrast(),
                CutoutBase(self.h, self.w),
                DistortionBase(),
                A.Normalize(self.MEANS, self.STDS),
                ToTensorV2(),
            ]
        )


class HeavyResizeTransforms(NormalizeBase):
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
                        A.CoarseDropout(max_holes=2, max_height=128, max_width=128),
                        A.CoarseDropout(max_holes=4, max_height=64, max_width=64),
                    ],
                    p=0.5,
                ),
                DistortionBase(),
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
        return F.cutout(image, holes, fill_value)

    def apply_to_mask(self, image, fill_value=0, holes=[], **params):
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
