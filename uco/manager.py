import numpy as np

from uco.runner import Runner
from uco.utils import (
    load_train_config,
    setup_logger,
    setup_logging,
    load_config,
    verbose_config_name
)


class EnsembleManager:

    def __init__(self, infer_config):
        self.infer_config = infer_config
        setup_logging(infer_config)
        self.logger = setup_logger(self, infer_config['verbose'])

    def start(self, num_models):
        randomiser = ConfigurationRandomiser('experiments/training-template.yml')
        for _ in range(num_models):
            try:
                # generate random training config
                train_config = randomiser.generate()
                train_config['name'] = verbose_config_name(train_config)

                # perform training
                checkpoint_dir = Runner(train_config).train(None)

                # run inference using the best model
                model_checkpoint = checkpoint_dir / 'model_best.pth'
                Runner(self.infer_config).predict(model_checkpoint)
            except Exception as ex:
                self.logger.warning(f'Caught exception: {ex}')

    def load_random_config(self):
        filename = np.random.choice(self.train_config_filenames)
        self.logger.info(f'Selected: "{filename}"')
        return load_train_config(filename)


class ConfigurationRandomiser:

    @property
    def base(self):
        return self._base.copy()

    def __init__(self, base_config):
        self._base = load_config(base_config)

    def generate(self):
        config = self.base
        for each in [
            SeedOptions,
            LossOptions,
            AnnealOptions,
            ModelOptions,
            LearningRateOptions
        ]:
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
        return [np.random.randint(0, 1e6)]

    @classmethod
    def update(cls, config):
        config['seed'] = int(cls.select())
        return config


class LossOptions(ConfigOptionBase):

    @classmethod
    def options(cls):
        bce_weights = [0.5, 0.6, 0.7, 0.8, 0.9]
        smooth_factors = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        opts = []
        for bce_weight in bce_weights:
            for smooth_factor in smooth_factors:
                opts.append({
                    'type': 'SmoothBCEDiceLoss',
                    'args': {
                        'bce_weight': bce_weight,
                        'dice_weight': 1 - bce_weight,
                        'smooth': smooth_factor,
                    }
                })
        return opts

    @classmethod
    def update(cls, config):
        config['loss'] = cls.select()
        return config


class AnnealOptions(ConfigOptionBase):

    @classmethod
    def options(cls):
        return [1, 2, 3, 4, 5]

    @classmethod
    def update(cls, config):
        config['lr_scheduler']['args']['start_anneal'] = int(cls.select())
        return config


class LearningRateOptions(ConfigOptionBase):

    @classmethod
    def options(cls):
        return [
            {
                'encoder': np.random.choice([1e-5, 3e-5, 5e-5, 7e-5, 9e-5]),
                'decoder': np.random.choice([1e-3, 3e-3, 5e-3]),
            }
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config['optimizer']['encoder']['lr'] = float(option['encoder'])
        config['optimizer']['decoder']['lr'] = float(option['decoder'])
        return config


class ModelOptions(ConfigOptionBase):

    @classmethod
    def options(cls):
        dropout = float(np.random.choice([0.00, 0.05, 0.10, 0.15, 0.20]))
        return [
            # unet - inceptionv4
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'inceptionv4',
                    'dropout': dropout,
                },
                'batch_size': 48,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 192,
                        'width': 228,
                    }
                }
            },
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'inceptionv4',
                    'dropout': dropout,
                },
                'batch_size': 24,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 256,
                        'width': 384,
                    }
                }
            },
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'inceptionv4',
                    'dropout': dropout,
                },
                'batch_size': 16,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 320,
                        'width': 480,
                    }
                }
            },
            # unet - inceptionresnetv2
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'inceptionresnetv2',
                    'dropout': dropout,
                },
                'batch_size': 40,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 192,
                        'width': 228,
                    }
                }
            },
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'inceptionresnetv2',
                    'dropout': dropout,
                },
                'batch_size': 20,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 256,
                        'width': 384,
                    }
                }
            },
            # unet - efficientnet-b0
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'efficientnet-b0',
                    'dropout': dropout,
                },
                'batch_size': 32,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 256,
                        'width': 384,
                    }
                }
            },
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'efficientnet-b0',
                    'dropout': dropout,
                },
                'batch_size': 20,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 320,
                        'width': 480,
                    }
                }
            },
            # unet - efficientnet-b2
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'efficientnet-b2',
                    'dropout': dropout,
                },
                'batch_size': 32,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 256,
                        'width': 384,
                    }
                }
            },
            {
                'type': 'Unet',
                'args': {
                    'encoder_name': 'efficientnet-b2',
                    'dropout': dropout,
                },
                'batch_size': 16,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 320,
                        'width': 480,
                    }
                }
            },
            # pspnet - efficientnet-b2
            {
                'type': 'PSPNet',
                'args': {
                    'encoder_name': 'efficientnet-b2',
                    'dropout': dropout,
                },
                'batch_size': 32,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 256,
                        'width': 384,
                    }
                }
            },
            {
                'type': 'PSPNet',
                'args': {
                    'encoder_name': 'efficientnet-b2',
                    'dropout': dropout,
                },
                'batch_size': 20,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 320,
                        'width': 480,
                    }
                }
            },
            # fpn - efficientnet-b2
            {
                'type': 'FPN',
                'args': {
                    'encoder_name': 'efficientnet-b2',
                    'dropout': dropout,
                },
                'batch_size': 28,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 256,
                        'width': 384,
                    }
                }
            },
            {
                'type': 'FPN',
                'args': {
                    'encoder_name': 'efficientnet-b2',
                    'dropout': dropout,
                },
                'batch_size': 16,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 320,
                        'width': 480,
                    }
                }
            },
            # fpn - efficientnet-b0
            {
                'type': 'FPN',
                'args': {
                    'encoder_name': 'efficientnet-b0',
                    'dropout': dropout,
                },
                'batch_size': 24,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 320,
                        'width': 480,
                    }
                }
            },
            {
                'type': 'FPN',
                'args': {
                    'encoder_name': 'efficientnet-b0',
                    'dropout': dropout,
                },
                'batch_size': 16,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 384,
                        'width': 576,
                    }
                }
            },
            {
                'type': 'FPN',
                'args': {
                    'encoder_name': 'efficientnet-b0',
                    'dropout': dropout,
                },
                'batch_size': 12,
                'augmentation': {
                    'type': 'HeavyResizeTransforms',
                    'args': {
                        'height': 448,
                        'width': 672,
                    }
                }
            },
        ]

    @classmethod
    def update(cls, config):
        option = cls.select()
        config['arch']['type'] = option['type']
        config['arch']['args'].update(option['args'])
        config['data_loader']['args']['batch_size'] = option['batch_size']
        config['augmentation'] = option['augmentation']
        return config
