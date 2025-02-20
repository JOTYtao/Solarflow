"""
Custom callbacks used for training
"""

from pytorch_lightning import Callback, LightningModule, Trainer
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
class NeptuneModelLogger(Callback):
    """
    Saves out the last and best models after each validation epoch.

    If the files don't exists, does nothing.

    Example::
        from pl_bolts.callbacks import NeptuneModelLogger
        trainer = Trainer(callbacks=[NeptuneModelLogger()])
    """

    def __init__(self, model_name: str) -> None:
        """
        Base initialization, nothing specific needed here
        """
        super().__init__()
        self.model_name = model_name

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Save the best and last model checkpoints to Neptune after each validation

        Args:
            trainer: PyTorchLightning trainer
            pl_module: LightningModule that is being trained

        Returns:
            None
        """
        try:
            last_ckpt_path = os.path.join(trainer.default_root_dir, "last.ckpt")

            trainer.logger.experiment[0][f"model_checkpoints/{self.model_name}/last.ckpt"].upload(last_ckpt_path)
        except Exception as e:
            print(
                f"No file to upload at {last_ckpt_path}. Error: {e}"
            )
            pass

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Save out the best and last model checkpoints at the end of trainer.fit to Neptune

        Args:
            trainer: PyTorchLightning Trainer
            pl_module: LightningModule being used for training

        Returns:
            None
        """
        try:
            best_ckpt_path = os.path.join(trainer.default_root_dir, "best.ckpt")
            trainer.logger.experiment[0][f"model_checkpoints/{self.model_name}/best.ckpt"].upload(best_ckpt_path)
        except Exception as e:
            print(
                f"No file to upload at {best_ckpt_path}. Error: {e}"
            )
            pass



class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step


    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            #grid = (grid + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag=tag,
                img_tensor=grid,
                global_step=pl_module.global_step
            )
    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            val = images[k]
            if val.ndim == 5:
                val = val.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()
            grid = torchvision.utils.make_grid(val, nrow=4)

            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
