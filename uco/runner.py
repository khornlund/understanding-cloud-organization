from pathlib import Path
from typing import Any, List, Tuple
from types import ModuleType

import torch
import torch.nn as nn
import ttach as tta
from tqdm import tqdm
from torchvision.utils import make_grid

import uco.model.optimizer as module_optimizer
import uco.model.scheduler as module_scheduler
import uco.data_loader.augmentation as module_aug
import uco.data_loader.data_loaders as module_data
import uco.model.loss as module_loss
import uco.model.metric as module_metric
import uco.model.model as module_arch
from uco.trainer import Trainer
from uco.utils import setup_logger, setup_logging, TensorboardWriter, seed_everything
from uco.h5 import HDF5PredictionWriter


class ManagerBase:
    def __init__(self, config: dict):
        setup_logging(config)
        seed_everything(config["seed"])
        self.logger = setup_logger(self, config["verbose"])
        self.cfg = config

    def get_instance(
        self, module: ModuleType, name: str, config: dict, *args: Any
    ) -> Any:
        """
        Helper to construct an instance of a class.

        Parameters
        ----------
        module : ModuleType
            Module containing the class to construct.
        name : str
            Name of class, as would be returned by ``.__class__.__name__``.
        config : dict
            Dictionary containing an 'args' item, which will be used as ``kwargs`` to
            construct the class instance.
        args : Any
            Positional arguments to be given before ``kwargs`` in ``config``.
        """
        ctor_name = config[name]["type"]
        self.logger.info(f"Building: {module.__name__}.{ctor_name}")
        return getattr(module, ctor_name)(*args, **config[name]["args"])

    def setup_device(
        self, model: nn.Module, target_devices: List[int]
    ) -> Tuple[torch.device, List[int]]:
        """
        setup GPU device if available, move model into configured device
        """
        available_devices = list(range(torch.cuda.device_count()))

        if not available_devices:
            self.logger.warning(
                "There's no GPU available on this machine. "
                "Training will be performed on CPU."
            )
            device = torch.device("cpu")
            model = model.to(device)
            return model, device

        if not target_devices:
            self.logger.info("No GPU selected. Training will be performed on CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            return model, device

        max_target_gpu = max(target_devices)
        max_available_gpu = max(available_devices)

        if max_target_gpu > max_available_gpu:
            msg = (
                f"Configuration requests GPU #{max_target_gpu} but only  "
                f"{max_available_gpu} available. Check the configuration and try again."
            )
            self.logger.critical(msg)
            raise Exception(msg)

        self.logger.info(
            f"Using devices {target_devices} of available devices {available_devices}"
        )
        device = torch.device(f"cuda:{target_devices[0]}")
        if len(target_devices) > 1:
            model = nn.DataParallel(model, device_ids=target_devices)
        else:
            model = model.to(device)
        return model, device


class TrainingManager(ManagerBase):
    """
    Top level class to construct objects for training.
    """

    def run(self, resume: str) -> None:
        cfg = self.cfg.copy()

        model = self.get_instance(module_arch, "arch", cfg)
        model, device = self.setup_device(model, cfg["target_devices"])
        torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

        param_groups = self.setup_param_groups(model, cfg["optimizer"])
        optimizer = self.get_instance(module_optimizer, "optimizer", cfg, param_groups)
        # optimizer = module_optimizer.Lookahead(optimizer)
        lr_scheduler = self.get_instance(
            module_scheduler, "lr_scheduler", cfg, optimizer
        )
        model, optimizer, start_epoch = self.resume_checkpoint(
            resume, model, optimizer, cfg
        )
        try:
            transforms = self.get_instance(module_aug, "augmentation", cfg)
        except:
            cfg["augmentation"]["type"] = "LightTransforms"
            transforms = self.get_instance(module_aug, "augmentation", cfg)

        data_loader = self.get_instance(module_data, "data_loader", cfg, transforms)
        valid_data_loader = data_loader.split_validation()

        self.logger.info("Getting loss and metric function handles")
        loss = self.get_instance(module_loss, "loss", cfg).to(device)
        metrics = [getattr(module_metric, met) for met in cfg["metrics"]]

        self.logger.info("Initialising trainer")
        trainer = Trainer(
            model,
            loss,
            metrics,
            optimizer,
            start_epoch=start_epoch,
            config=cfg,
            device=device,
            data_loader=data_loader,
            valid_data_loader=valid_data_loader,
            lr_scheduler=lr_scheduler,
        )

        checkpoint_dir = trainer.train()
        self.logger.info("Training completed.")
        return checkpoint_dir

    def setup_param_groups(self, model: nn.Module, config: dict) -> dict:
        """
        Helper to remove weight decay from bias parameters.
        """
        encoder_opts = config["encoder"]
        decoder_opts = config["decoder"]

        encoder_weight_params = []
        encoder_bias_params = []
        decoder_weight_params = []
        decoder_bias_params = []

        for name, param in model.encoder.named_parameters():
            if name.endswith("bias"):
                encoder_bias_params.append(param)
            else:
                encoder_weight_params.append(param)

        for name, param in model.decoder.named_parameters():
            if name.endswith("bias"):
                decoder_bias_params.append(param)
            else:
                decoder_weight_params.append(param)

        self.logger.info(f"Found {len(encoder_weight_params)} encoder weight params")
        self.logger.info(f"Found {len(encoder_bias_params)} encoder bias params")
        self.logger.info(f"Found {len(decoder_weight_params)} decoder weight params")
        self.logger.info(f"Found {len(decoder_bias_params)} decoder bias params")

        params = [
            {"params": encoder_weight_params, **encoder_opts},
            {"params": decoder_weight_params, **decoder_opts},
            {
                "params": encoder_bias_params,
                "lr": encoder_opts["lr"],
                "weight_decay": encoder_opts["weight_decay"],
            },
            {
                "params": decoder_bias_params,
                "lr": decoder_opts["lr"],
                "weight_decay": decoder_opts["weight_decay"],
            },
        ]
        return params

    def resume_checkpoint(self, resume_path, model, optimizer, config):
        """
        Resume from saved checkpoint.
        """
        if not resume_path:
            return model, optimizer, 0

        self.logger.info(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from "
                "that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(f'Checkpoint "{resume_path}" loaded')
        return model, optimizer, checkpoint["epoch"]


class InferenceManager(ManagerBase):
    """
    Top level class to perform inference.
    """

    SCORE_THRESHOLD = 0.610

    def run(self, model_checkpoint: str) -> None:
        cfg = self.cfg.copy()
        device = self.setup_device(cfg["device"])
        checkpoint = self.load_checkpoint(model_checkpoint, device)
        if not self.check_score(checkpoint):
            return
        train_cfg = checkpoint["config"]

        model = self.get_instance(module_arch, "arch", train_cfg)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

        tta_model = tta.SegmentationTTAWrapper(
            model,
            tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()]),
            merge_mode="mean",
        )

        transforms = self.get_instance(module_aug, "augmentation", train_cfg)
        data_loader = self.get_instance(module_data, "data_loader", cfg, transforms)

        timestamp = Path(model_checkpoint).parent.parent.name
        rw = HDF5PredictionWriter(filename=cfg["output"]["h5"], dataset=str(timestamp))
        writer_dir = Path(cfg["save_dir"]) / cfg["name"] / timestamp
        writer = TensorboardWriter(writer_dir, cfg["tensorboard"])
        self.logger.info("Performing inference")
        tta_model.eval()
        with torch.no_grad():
            for bidx, (f, data) in tqdm(enumerate(data_loader), total=len(data_loader)):
                data = data.to(device)
                output = torch.sigmoid(tta_model(data))
                if bidx % 10 == 0:
                    self.log_predictions_tensorboard(writer, bidx, data, output)
                output = output.cpu().numpy()
                rw.write(output)

        self.logger.info(rw.close())

    # -- helpers ---------------------------------------------------------------------

    def log_predictions_tensorboard(self, writer, bidx, data, output):
        writer.set_step(bidx, "inference")
        data, output = data.cpu(), output.cpu()
        data_grayscale = torch.mean(data, dim=1, keepdim=True)
        for c in range(4):
            image = torch.cat([data_grayscale, output[:, c : c + 1, :, :]], dim=0)
            writer.add_image(
                f"class_{c}",
                make_grid(image, nrow=data.size(0), normalize=True, scale_each=True),
            )

    def check_score(self, checkpoint):
        best_score = checkpoint["monitor_best"].item()
        self.logger.info(f"Best score: {best_score}")
        if best_score < self.SCORE_THRESHOLD:
            self.logger.warning(f"Skipping low scoring model")
            return False
        return True

    def load_checkpoint(self, path, device):
        """
        Load a saved checkpoint.
        """
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        return checkpoint

    def setup_device(self, device):
        if device is None or device == "cpu":
            return torch.device("cpu")
        else:
            return torch.device(device)
