from typing import Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from models.layers.loss import get_loss
from typing import Type
from models.Swin_LSTM.swin_LSTM import SwinLSTM
import numpy as np
import neptune.types
from torchmetrics import MeanMetric
import torchvision.transforms
from pathlib import Path
import torch
import torchvision
import pandas as pd
from diffusers import AutoencoderKL
import warnings
REGISTERED_MODELS = {}

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls

@register_model
class Swin_LSTMpl(pl.LightningModule):
    def __init__(
            self,
            img_size=16,
            patch_size=2,
            input_channels: int = 32,
            out_channels: int = 32,
            depths_downsample=[2, 6],
            depths_upsample = [6, 2],
            num_heads = [4, 8],          
            window_size = 4,   
            forecast_steps: int = 8,
            lr: float = 0.001,
            visualize: bool = False,
            loss: Union[str, torch.nn.Module] = "mse",
            # pretrained: bool = False,
            save_dir: str = None,
    ):
        super(Swin_LSTMpl, self).__init__()
        self.save_hyperparameters()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_steps = forecast_steps
        self.criterion = get_loss(loss)
        self.lr = lr
        self.visualize = visualize
        self.input_channels = input_channels
        self.output_channels = out_channels
        self.model = SwinLSTM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            embed_dim=out_channels,
            depths_downsample=depths_downsample,
            depths_upsample=depths_upsample,
            num_heads=num_heads,
            window_size=window_size
            )
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.save_hyperparameters()
        self.test_step_outputs = []
        self.test_loader = None
        self.depths_downsample = depths_downsample
        self.depths_upsample = depths_upsample

    @classmethod
    def from_config(cls, config):
        return Swin_LSTMpl(
            input_channels=config.get("in_channels"),
            img_size=config.get("img_size"),
            patch_size=config.get("patch_size"),
            depths_downsample=config.get("depths_downsample"),
            depths_upsample=config.get("depths_upsample"),
            num_heads=config.get("num_heads"),          
            window_size=config.get("window_size"),  
            out_channels=config.get("out_channels"),
            forecast_steps=config.get("forecast_steps"),
            lr=config.get("lr"),
            lr_scheduler_mode=config.get("lr_scheduler_mode"),
            visualize=config.get("visualize"),
            save_dir=config.get("save_dir")
        )

    def setup(self, stage='test'):
        if stage == 'test':
            self.test_loader = self.trainer.datamodule.test_dataloader()

    def forward(self, x):
        states_down = [None] * len(self.depths_downsample)
        states_up = [None] * len(self.depths_upsample)
        return self.model.forward(x, states_down, states_up)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
        
    def training_step(self, batch, batch_idx):

        images = batch["his_cal"].float()
        target = batch["target"].float()
        B, C, T, H, W = images.shape
        z_input = images.reshape(B, T, H, W)
        z_target = target.reshape(B, T, H, W)
        y_hat = self(z_input)
        loss = self.criterion(y_hat, z_target)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        frame_loss_dict = {}
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["his_cal"].float()
        target = batch["target"].float()
        B, C, T, H, W = images.shape
        z_input = images.reshape(B, T, H, W)
        z_target = target.reshape(B, T, H, W)
        y_hat = self(z_input)
        loss = self.criterion(y_hat, z_target)
        self.train_loss(loss)
        self.log("val/loss", self.train_loss, on_step=True, prog_bar=True)
        frame_loss_dict = {}
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            images = batch["his_cal"].float()
            target = batch["target"].float()
            B, C, T, H, W = images.shape
            z_input = images.reshape(B, T, H, W)
            z_target = target.reshape(B, T, H, W)
            y_hat = self(z_input)
            loss = self.criterion(y_hat, z_target)
            y_hat = y_hat.unsqueeze(1)  
            y_pred = y_hat.cpu()
            y_target = target.cpu()
        save_path = self.save_dir / "predictions"
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / f"pred_batch_{batch_idx}.npy", y_pred)
        np.save(save_path / f"target_batch_{batch_idx}.npy", y_target)
    
        del y_hat, y_target, y_pred
        torch.cuda.empty_cache()
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        output = {
            "loss": loss,
            "batch_idx": batch_idx
        }
        self.test_step_outputs.append(output)
        return output
    def on_test_epoch_end(self):
        pred_path = self.save_dir / "predictions"
        pred_files = sorted(pred_path.glob("pred_batch_*.npy"))
        target_files = sorted(pred_path.glob("target_batch_*.npy"))
        all_preds = []
        all_targets = []
        for pred_file, target_file in zip(pred_files, target_files):
            pred = np.load(pred_file)
            target = np.load(target_file)
            all_preds.append(pred)
            all_targets.append(target)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics_dict = {}
        print(f"Predictions shape after squeeze: {all_preds.shape}")
        print(f"Targets shape after squeeze: {all_targets.shape}") 
        for t in range(all_preds.shape[1]):
            pred_t = all_preds[:, t]  # (B, 1, H, W)
            target_t = all_targets[:, t]  # (B, 1, H, W)

            mae_t = np.mean(np.abs(pred_t - target_t))
            rmse_t = np.sqrt(np.mean((pred_t - target_t) ** 2))
            metrics_dict.update({
                f"test_mae_t{t}": float(mae_t),
                f"test_rmse_t{t}": float(rmse_t)
            })
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        metrics_dict.update({
            "test_mae_overall": float(mae),
            "test_rmse_overall": float(rmse)
        })
        save_dir = self.save_dir
        save_dir.mkdir(exist_ok=True, parents=True)

        np.save(save_dir / "predictions.npy", all_preds)
        np.save(save_dir / "targets.npy", all_targets)
    
        results_df = pd.DataFrame({
            "metric": list(metrics_dict.keys()),
            "value": list(metrics_dict.values())
        })
        results_df.to_csv(save_dir / "test_metrics.csv", index=False)
        for name, value in metrics_dict.items():
            self.log(name, value)
        self.test_step_outputs.clear()
        torch.cuda.empty_cache()
        return metrics_dict
    
    def visualize_step(self, x, y, y_hat, batch_idx, step="train"):
        if not self.visualize:
            return

        if len(x.shape) == 5:
            images = x[0].cpu().detach()
            for i, t in enumerate(images):
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
                image_grid = torchvision.transforms.ToPILImage()(image_grid)
                self.logger.experiment[f"{step}/Input_Image_Stack_Frame_{i}"].log(
                    neptune.types.File.as_image(image_grid)
                )

            images = y[0].cpu().detach()
            for i, t in enumerate(images):
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.output_channels)
                image_grid = torchvision.transforms.ToPILImage()(image_grid)
                self.logger.experiment[f"{step}/Target_Image_Stack_Frame_{i}"].log(
                    neptune.types.File.as_image(image_grid)
                )
            images = y_hat[0].cpu().detach()
            for i, t in enumerate(images):
                t = [torch.unsqueeze(img, dim=0) for img in t]
                image_grid = torchvision.utils.make_grid(t, nrow=self.output_channels)
                image_grid = torchvision.transforms.ToPILImage()(image_grid)
                self.logger.experiment[f"{step}/Generated_Stack_Frame_{i}"].log(
                    neptune.types.File.as_image(image_grid)
                )


