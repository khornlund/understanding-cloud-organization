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
        train_cfg = infer_config["training"]
        self.randomiser = globals()[train_cfg["randomiser"]](train_cfg["template"])

    def start(self, num_models):
        model_checkpoint = None
        train_config = None
        for _ in range(num_models):
            try:
                # generate random training config
                seed = int(time.clock() * 1000)
                self.logger.info(f"Using seed: {seed}")
                seed_everything(seed)
                train_config = self.randomiser.generate()

                self.logger.info(
                    f"Starting seed {train_config['seed']}: {train_config}"
                )

                # perform training
                checkpoint_dir = TrainingManager(train_config).run(None)

                # delete other checkpoints
                for f in checkpoint_dir.glob("checkpoint-epoch*.pth"):
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


# -- Segmentation Configurations ------------------------------------------------------


class ConfigurationRandomiserSegmentation:
    @property
    def base(self):
        return deepcopy(self._base)

    def __init__(self, base_config):
        self._base = load_config(base_config)

    def generate(self):
        config = self.base
        for each in [
            SeedOptions,
            LossOptionsSegmentation,
            ModelOptionsSegmentation,
            OptimizerOptionsSegmentation,
        ]:
            config = each.update(config)
        return config


class LossOptionsSegmentation(ConfigOptionBase):
    @classmethod
    def options(cls):
        opts = []
        bce_weight = np.random.uniform(0.83, 0.92)
        opts.append(
            {
                "type": "BCELovaszLoss",
                "args": {"bce_weight": bce_weight, "lovasz_weight": 1 - bce_weight},
            }
        )
        bce_weight = np.random.uniform(0.65, 0.75)
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


class OptimizerOptionsSegmentation(ConfigOptionBase):
    @classmethod
    def options(cls):
        encoder_lr = np.random.uniform(5e-5, 9e-5)
        decoder_lr = np.random.uniform(3e-3, 5e-3)
        return [
            {
                "optim": "QHAdamW",
                "args": {"nus": (0.7, 1.0)},
                "encoder_lr": encoder_lr,
                "decoder_lr": decoder_lr,
            },
            {
                "optim": "RAdam",
                "args": {},
                "encoder_lr": encoder_lr,
                "decoder_lr": decoder_lr,
            },
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["optimizer"]["type"] = str(option["optim"])
        config["optimizer"]["args"].update(**option["args"])
        config["optimizer"]["encoder"]["lr"] = float(option["encoder_lr"])
        config["optimizer"]["decoder"]["lr"] = float(option["decoder_lr"])
        return config


class ModelOptionsSegmentation(ConfigOptionBase):
    @classmethod
    def options(cls):
        dropout = float(np.random.uniform(0.10, 0.20))
        transforms = str(
            np.random.choice(
                [
                    "CutoutTransforms",
                    "DistortionTransforms",
                    "CutoutDistortionTransforms",
                    "HeavyResizeTransforms",
                ]
            )
        )
        return [
            {
                "type": "FPN",
                "args": {"encoder_name": "efficientnet-b0", "dropout": dropout},
                "batch_size": 12,
                "decoder_merge_policy": "cat",
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 448, "width": 672},
                },
            },
            # add other configs as desired
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["arch"]["type"] = option["type"]
        config["arch"]["args"].update(option["args"])
        config["data_loader"]["args"]["batch_size"] = option["batch_size"]
        config["augmentation"] = option["augmentation"]
        return config


# -- Classification Configurations ----------------------------------------------------


class ConfigurationRandomiserClassification:
    @property
    def base(self):
        return deepcopy(self._base)

    def __init__(self, base_config):
        self._base = load_config(base_config)

    def generate(self):
        config = self.base
        for each in [
            SeedOptions,
            ModelOptionsClassification,
            OptimizerOptionsClassification,
        ]:
            config = each.update(config)
        return config


class OptimizerOptionsClassification(ConfigOptionBase):
    @classmethod
    def options(cls):
        return [
            {
                "optim": np.random.choice(["RAdam", "QHAdamW"]),
                "lr": np.random.uniform(5e-5, 1e-4),
            }
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["optimizer"]["type"] = str(option["optim"])
        config["optimizer"]["args"]["lr"] = float(option["lr"])
        return config


class ModelOptionsClassification(ConfigOptionBase):
    @classmethod
    def options(cls):
        transforms = str(
            np.random.choice(
                [
                    "CutoutTransforms",
                    "DistortionTransforms",
                    "CutoutDistortionTransforms",
                    "HeavyResizeTransforms",
                ]
            )
        )
        return [
            {
                "type": "EfficientNet",
                "args": {"encoder_name": "efficientnet-b2"},
                "batch_size": 16,
                "augmentation": {
                    "type": transforms,
                    "args": {"height": 384, "width": 576},
                },
            },
            # add other configs as desired
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config["arch"]["type"] = option["type"]
        config["arch"]["args"].update(option["args"])
        config["data_loader"]["args"]["batch_size"] = option["batch_size"]
        config["augmentation"] = option["augmentation"]
        return config
