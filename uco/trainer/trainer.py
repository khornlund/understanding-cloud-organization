import numpy as np
import torch
from torchvision.utils import make_grid  # noqa

from uco.base import TrainerBase, AverageMeter


class Trainer(TrainerBase):
    """
    Responsible for training loop and validation.
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        start_epoch,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
    ):
        super().__init__(model, loss, metrics, optimizer, start_epoch, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.bs)) * 32
        self.start_val_epoch = config["training"]["start_val_epoch"]
        self.unfreeze_encoder = config["training"]["unfreeze_encoder"]

        # self.logger.info("Freezing encoder weights")
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False

    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        Returns
        -------
        dict
            Dictionary containing results for the epoch.
        """
        # if self.unfreeze_encoder is not None and epoch >= self.unfreeze_encoder:
        #     self.logger.info("Unfreezing encoder weights")
        #     for p in self.model.encoder.parameters():
        #         p.requires_grad = True
        #     self.unfreeze_encoder = None

        self.loss.set_epoch(epoch)
        self.model.train()
        self.writer.set_step((epoch) * len(self.data_loader))
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:
                self.writer.add_scalar("LR/encoder", param_group["lr"])
            elif i == 1:
                self.writer.add_scalar("LR/decoder", param_group["lr"])
        # self.writer.add_scalar("weight/bce", self.loss.bce_weight)

        loss_mtrs = [AverageMeter(loss) for loss in ["loss", "bce", "dice", "lovasz"]]
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        # visualize_batch = np.random.choice(np.arange(len(self.data_loader) - 1))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss_dict = self.loss(output, target)
            loss = loss_dict["loss"]
            loss.backward()
            self.optimizer.step()

            for mtr in loss_mtrs:
                mtr.update(
                    loss_dict.get(mtr.name, torch.tensor([0])).item(), data.size(0)
                )

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch) * len(self.data_loader) + batch_idx)
                for mtr in loss_mtrs:
                    self.writer.add_scalar(f"batch/{mtr.name}", mtr.val)
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                    self.writer.add_scalar(f"batch/{mtr.name}", value)
                self._log_batch(
                    epoch,
                    batch_idx,
                    self.data_loader.bs,
                    len(self.data_loader),
                    loss.item(),
                )

            # if batch_idx == visualize_batch:
            #     self.writer.set_step((epoch) * len(self.data_loader) + batch_idx)
            #     # construct an image for each class that will have the data, output,
            #     # and target on each of 3 rows in tensorboard
            #     data, target = data.cpu(), target.cpu()
            #     output = torch.sigmoid(output.cpu())
            #     data_grayscale = torch.mean(data, dim=1, keepdim=True)
            #     for c in range(4):
            #         image = torch.cat([
            #             data_grayscale,
            #             output[:, c:c + 1, :, :],
            #             target[:, c:c + 1, :, :],
            #         ], dim=0)
            #         self.writer.add_image(
            #             f'class_{c}',
            #             make_grid(image,
            #             nrow=data.size(0), normalize=True, scale_each=True)
            #         )

        del data
        del target
        del output
        torch.cuda.empty_cache()

        for mtr in loss_mtrs:
            self.writer.add_scalar(f"epoch/{mtr.name}", mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(f"epoch/{mtr.name}", mtr.avg)

        log = {mtr.name: mtr.avg for mtr in loss_mtrs}
        log["metrics"]: [mtr.avg for mtr in metric_mtrs]

        if self.do_validation and (epoch % 2 == 0 or epoch >= self.start_val_epoch):
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _log_batch(self, epoch, batch_idx, batch_size, len_data, loss):
        n_samples = batch_size * len_data
        n_complete = batch_idx * batch_size
        percent = 100.0 * batch_idx / len_data
        msg = (
            f"Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] "
            f"Loss: {loss:.6f}"
        )
        self.logger.debug(msg)

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            for metric in self.metrics:
                value = metric(output, target)
                yield value

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validate after training an epoch

        Returns
        -------
        dict
            Contains keys 'val_loss' and 'val_metrics'.
        """
        self.model.eval()
        self.writer.set_step(epoch, "valid")
        loss_mtrs = [AverageMeter(loss) for loss in ["loss", "bce", "dice", "lovasz"]]
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        # visualize_batch = np.random.choice(np.arange(len(self.valid_data_loader) - 1))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss_dict = self.loss(output, target)

                for mtr in loss_mtrs:
                    mtr.update(
                        loss_dict.get(mtr.name, torch.tensor([0])).item(), data.size(0)
                    )
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

                # if batch_idx == visualize_batch:
                #     data, target = data.cpu(), target.cpu()
                #     output = torch.sigmoid(output.cpu())
                #     data_grayscale = torch.mean(data, dim=1, keepdim=True)
                #     for c in range(4):
                #         image = torch.cat([
                #             data_grayscale,
                #             output[:, c:c + 1, :, :],
                #             target[:, c:c + 1, :, :],
                #         ], dim=0)
                #         self.writer.add_image(
                #             f'class_{c}',
                #             make_grid(image,
                #             nrow=data.size(0), normalize=True, scale_each=True)
                #         )

        for mtr in loss_mtrs:
            self.writer.add_scalar(mtr.name, mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(mtr.name, mtr.avg)

        del data
        del target
        del output
        torch.cuda.empty_cache()

        log = {f"val_{mtr.name}": mtr.avg for mtr in loss_mtrs}
        log["val_metrics"] = [mtr.avg for mtr in metric_mtrs]
        return log
