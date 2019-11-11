import time
from pathlib import Path
from copy import deepcopy

import numpy as np

from uco.runner import TrainingManager, InferenceManager
from uco.utils import setup_logger, setup_logging, load_config, seed_everything, Indexer


GPU = 11  # RTX 2080Ti


class EnsembleManager:
    def __init__(self, infer_config):
        self.infer_config = infer_config
        setup_logging(infer_config)
        self.logger = setup_logger(self, infer_config["verbose"])
        self.run_inference = infer_config.get("run_inference", False)

    def start(self, num_models):
        # randomiser = ConfigurationRandomiser("experiments/training-template.yml")
        model_checkpoint = None
        train_config = None
        for _ in range(num_models):
            try:
                # generate random training config
                seed = int(time.clock() * 1000)
                self.logger.info(f"Using seed: {seed}")
                seed_everything(seed)
                # train_config = randomiser.generate()

                config_filenames = list(Path("experiments/blessed").glob("*.yml"))
                train_config = load_config(np.random.choice(config_filenames))
                train_config["seed"] = seed

                self.logger.info(
                    f"Starting seed {train_config['seed']}: {train_config}"
                )

                # perform training
                checkpoint_dir = TrainingManager(train_config).run(None)

                # delete other checkpoints
                for f in checkpoint_dir.glob("checkpoint-epoch*.pth"):
                    if "model_best" in f.name:
                        self.logger.warning(f"Search found {f}")
                        continue
                    self.logger.info(f"Deleting {f}")
                    f.unlink()

                # Log details for run
                Indexer.index(checkpoint_dir.parent)

                if self.run_inference:  # run inference using the best model
                    model_checkpoint = checkpoint_dir / "model_best.pth"
                    InferenceManager(self.infer_config).run(model_checkpoint)
            except Exception as ex:
                self.logger.critical(f"Caught exception: {ex}")
                self.logger.critical(f"Model checkpoint: {model_checkpoint}")
                self.logger.critical(f"Config: {train_config}")


class ConfigurationRandomiser:
    @property
    def base(self):
        return deepcopy(self._base)

    def __init__(self, base_config):
        self._base = load_config(base_config)

    def generate(self):
        config = self.base
        for each in [SeedOptions, LossOptions, ModelOptions, OptimizerOptions]:
            config = each.update(config)
        return config


class ConfigOptionBase:
    @classmethod
    def select(cls):
        return np.random.choice(cls.options())

    @classmethod
    def options(cls):
        raise NotImplementedError

    @classmethod
    def update(cls, config):
        raise NotImplementedError


class SeedOptions(ConfigOptionBase):
    @classmethod
    def options(cls):
        return [int(time.clock() * 1000)]

    @classmethod
    def update(cls, config):
        config["seed"] = int(cls.select())
        return config


class LossOptions(ConfigOptionBase):
    @classmethod
    def options(cls):
        opts = []
        # bce_weight = np.random.uniform(0.83, 0.92)
        # opts.append(
        #     {
        #         "type": "BCELovaszLoss",
        #         "args": {"bce_weight": bce_weight, "lovasz_weight": 1 - bce_weight},
        #     }
        # )
        bce_weight = np.random.uniform(0.65, 0.75)
        opts.append(
            {
                "type": "BCEDiceLoss",
                "args": {"bce_weight": bce_weight, "dice_weight": 1 - bce_weight},
            }
        )
        # opts.append(
        #     {"type": "BCEDiceLoss", "args": {"bce_weight": 0.99, "dice_weight": 0.01}}
        # )
        return opts

    @classmethod
    def update(cls, config):
        config["loss"] = cls.select()
        return config


class OptimizerOptions(ConfigOptionBase):
    @classmethod
    def options(cls):
        return [
            {
                "optim": np.random.choice(
                    [
                        # "RAdam",
                        "QHAdamW"
                    ]
                ),
                # "encoder": np.random.uniform(5e-5, 9e-5),
                "encoder": np.random.uniform(5e-5, 6e-5),
                "decoder": np.random.uniform(3e-3, 4e-3),
            }
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["optimizer"]["type"] = str(option["optim"])
        config["optimizer"]["encoder"]["lr"] = float(option["encoder"])
        config["optimizer"]["decoder"]["lr"] = float(option["decoder"])

        # if config["loss"]["args"]["bce_weight"] > 0.98:
        #     config["optimizer"]["encoder"]["lr"] *= 1.5
        #     config["optimizer"]["decoder"]["lr"] *= 1.5
        return config


class ModelOptions(ConfigOptionBase):
    @classmethod
    def options(cls):
        dropout = float(np.random.uniform(0.10, 0.20))
        transforms = str(
            np.random.choice(
                [
                    # "CutoutTransforms",
                    # "CutoutImgOnlyTransforms",
                    # "DistortionTransforms",
                    # "CutoutDistortionTransforms",
                    "HeavyResizeTransforms"
                ]
            )
        )
        return (
            [
                # unet - efficientnet-b0
                {
                    "type": "Unet",
                    "args": {"encoder_name": "efficientnet-b0", "dropout": dropout},
                    "batch_size": 20,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                # unet - efficientnet-b2
                {
                    "type": "Unet",
                    "args": {"encoder_name": "efficientnet-b2", "dropout": dropout},
                    "batch_size": 24,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 256, "width": 384},
                    },
                },
                {
                    "type": "Unet",
                    "args": {"encoder_name": "efficientnet-b2", "dropout": dropout},
                    "batch_size": 16,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                # fpn - efficientnet-b2
                # {
                #     "type": "FPN",
                #     "args": {
                #         "encoder_name": "efficientnet-b2",
                #         "dropout": dropout,
                #         "decoder_merge_policy": "cat",
                #     },
                #     "batch_size": 16,
                #     "augmentation": {
                #         "type": transforms,
                #         "args": {"height": 320, "width": 480},
                #     },
                # },
            ]
            if GPU == 11
            else [
                # unet - efficientnet-b5
                {
                    "type": "Unet",
                    "args": {
                        "encoder_name": "efficientnet-b5",
                        # "dropout": dropout
                    },
                    "batch_size": 10,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                # unet - resnext101_32x8d
                {
                    "type": "Unet",
                    "args": {
                        "encoder_name": "resnext101_32x8d",
                        # "dropout": dropout
                    },
                    "batch_size": 12,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                {
                    "type": "Unet",
                    "args": {
                        "encoder_name": "resnext101_32x8d",
                        # "dropout": dropout,
                        "encoder_weights": "instagram",
                    },
                    "batch_size": 12,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                # fpn - efficientnet-b5
                {
                    "type": "FPN",
                    "args": {
                        "encoder_name": "efficientnet-b5",
                        "dropout": dropout,
                        "decoder_merge_policy": "cat",
                    },
                    "batch_size": 12,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                # fpn - resnext101_32x8d
                {
                    "type": "FPN",
                    "args": {
                        "encoder_name": "resnext101_32x8d",
                        "dropout": dropout,
                        "decoder_merge_policy": "cat",
                    },
                    "batch_size": 14,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                {
                    "type": "FPN",
                    "args": {
                        "encoder_name": "resnext101_32x8d",
                        "dropout": dropout,
                        "decoder_merge_policy": "cat",
                        "encoder_weights": "instagram",
                    },
                    "batch_size": 14,
                    "augmentation": {
                        "type": transforms,
                        "args": {"height": 320, "width": 480},
                    },
                },
                # fpn - se_resnext101_32x4d
            ]
        )

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["arch"]["type"] = option["type"]
        config["arch"]["args"].update(option["args"])
        config["data_loader"]["args"]["batch_size"] = option["batch_size"]
        config["augmentation"] = option["augmentation"]
        return config
