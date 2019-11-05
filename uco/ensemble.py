import time
from copy import deepcopy

import numpy as np

from uco.runner import TrainingManager, InferenceManager
from uco.utils import setup_logger, setup_logging, load_config, seed_everything, Indexer


class EnsembleManager:
    def __init__(self, infer_config):
        self.infer_config = infer_config
        setup_logging(infer_config)
        self.logger = setup_logger(self, infer_config["verbose"])
        self.run_inference = infer_config.get("run_inference", False)

    def start(self, num_models):
        randomiser = ConfigurationRandomiser("experiments/training-template.yml")
        model_checkpoint = None
        train_config = None
        for _ in range(num_models):
            try:
                # generate random training config
                seed = int(time.clock() * 1000)
                self.logger.info(f"Using seed: {seed}")
                seed_everything(seed)
                train_config = randomiser.generate()

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
        bce_weights = [0.65, 0.70, 0.75]
        opts = []
        for bce_weight in bce_weights:
            opts.append(
                {
                    "type": "BCEDiceLoss",
                    "args": {"bce_weight": bce_weight, "dice_weight": 1 - bce_weight},
                }
            )
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
                "optim": np.random.choice(["RAdam", "QHAdamW"]),
                "encoder": np.random.choice([5e-5, 7e-5, 9e-5]),
                "decoder": np.random.choice([3e-3, 4e-3]),
            }
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["optimizer"]["type"] = str(option["optim"])
        config["optimizer"]["encoder"]["lr"] = float(option["encoder"])
        config["optimizer"]["decoder"]["lr"] = float(option["decoder"])
        return config


class ModelOptions(ConfigOptionBase):
    @classmethod
    def options(cls):
        dropout = float(np.random.choice([0.10, 0.15, 0.20]))
        transforms = str(
            np.random.choice(
                [
                    # "LightTransforms",
                    "CutoutTransforms",
                    "DistortionTransforms",
                    "CutoutDistortionTransforms",
                    # "HeavyResizeTransforms",
                ]
            )
        )
        return [
            # unet - inceptionv4
            {
                "type": "Unet",
                "args": {"encoder_name": "inceptionv4", "dropout": dropout},
                "batch_size": 24,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 256, "width": 384},
                },
            },
            {
                "type": "Unet",
                "args": {"encoder_name": "inceptionv4", "dropout": dropout},
                "batch_size": 16,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 320, "width": 480},
                },
            },
            # unet - inceptionresnetv2
            {
                "type": "Unet",
                "args": {"encoder_name": "inceptionresnetv2", "dropout": dropout},
                "batch_size": 20,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 256, "width": 384},
                },
            },
            # unet - efficientnet-b0
            {
                "type": "Unet",
                "args": {"encoder_name": "efficientnet-b0", "dropout": dropout},
                "batch_size": 32,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 256, "width": 384},
                },
            },
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
            {
                "type": "FPN",
                "args": {
                    "encoder_name": "efficientnet-b2",
                    "dropout": dropout,
                    "decoder_merge_policy": "cat",
                },
                "batch_size": 28,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 256, "width": 384},
                },
            },
            {
                "type": "FPN",
                "args": {
                    "encoder_name": "efficientnet-b2",
                    "dropout": dropout,
                    "decoder_merge_policy": "cat",
                },
                "batch_size": 16,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 320, "width": 480},
                },
            },
            # fpn - efficientnet-b0
            {
                "type": "FPN",
                "args": {
                    "encoder_name": "efficientnet-b0",
                    "dropout": dropout,
                    "decoder_merge_policy": "cat",
                },
                "batch_size": 24,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 320, "width": 480},
                },
            },
            {
                "type": "FPN",
                "args": {
                    "encoder_name": "efficientnet-b0",
                    "dropout": dropout,
                    "decoder_merge_policy": "cat",
                },
                "batch_size": 16,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 384, "width": 576},
                },
            },
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["arch"]["type"] = option["type"]
        config["arch"]["args"].update(option["args"])
        config["data_loader"]["args"]["batch_size"] = option["batch_size"]
        config["augmentation"] = option["augmentation"]
        return config
