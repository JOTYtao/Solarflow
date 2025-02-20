from typing import Union

import antialiased_cnns
import numpy as np
import pytorch_lightning as pl
import torch
import neptune.types
import torchvision
from models.layers.loss import get_loss
from typing import Any, Type
from torchmetrics import MeanMetric
from pathlib import Path
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from models.utils import warmup_lambda
from models.Unet_layer.D3Unet import Unet3D
REGISTERED_MODELS = {}
def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls

class D3Unet_model(pl.LightningModule):
    def __init__(
        self,
        dim: int = 64,
        out_channels: int = 1,
        dim_mults=(1, 2, 4, 8),
        input_channels: int = 1,
        attn_heads=4,
        attn_dim_head=32,


        forecast_steps: int = 6,
        lr: float = 0.001,
        visualize: bool = False,
        loss: Union[str, torch.nn.Module] = "mse",
        lr_scheduler_mode: str = "plateau",
        max_epochs: int = 1000,
        save_dir: str = None
    ):
        super(D3Unet_model, self).__init__()
        self.save_hyperparameters()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_steps = forecast_steps
        self.criterion = get_loss(loss)
        self.lr = lr
        self.visualize = visualize
        self.input_channels = input_channels
        self.output_channels = out_channels
        self.model = Unet3D(
            dim=dim,
            out_dim=out_channels,
            dim_mults=dim_mults,
            channels=input_channels,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.save_hyperparameters()
        self.test_step_outputs = []
        self.test_loader = None
        self.lr_scheduler_mode = lr_scheduler_mode
        self.max_epochs = max_epochs
        self.all_preds = []
        self.all_targets = []

    @classmethod
    def from_config(cls, config):
        return D3Unet_model(
            dim=config.get("dim",64),
            input_channels=config.get("in_channels", 7),
            out_channels=config.get("out_channels", 1),
            dim_mults=config.get("dim_mults", (1, 2, 4, 8)),
            attn_heads=config.get("attn_heads", 4),
            attn_dim_head=config.get("attn_dim_head", 32),
            forecast_steps=config.get("forecast_steps", 6),
            lr=config.get("lr", 0.001),
            lr_scheduler_mode=config.get("lr_scheduler_mode"),
            visualize=config.get("visualize"),
            max_epochs=config.get("max_epochs"),
            save_dir=config.get("save_dir")
        )
    def setup(self, stage='test'):
        if stage == 'test':
            self.test_loader = self.trainer.datamodule.test_dataloader()

    # def forward(self, x, out_channels):
    #     B, C, T, H, W = x.shape
    #     x_reshaped = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    #     x_reshaped = x_reshaped.reshape(B * T, C, H, W)  # [B*T, C, H, W]
    #     output = self.model.forward(x_reshaped)  # [B*T, out_put channel, H, W]
    #     output = output.reshape(B, T, out_channels, H, W)  # [B, T, out_put channel, H, W]
    #     return output

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler_mode == 'plateau':
            # Warmup
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            accumulate_grad_batches = 2
            warmup_percentage = 0.05
            total_num_steps = (self.max_epochs * steps_per_epoch) // accumulate_grad_batches
            warmup_steps = int(warmup_percentage * total_num_steps)
            base_lr_ratio = 1.0e-3
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=warmup_lambda(warmup_steps, base_lr_ratio)
            )

            # ReduceLROnPlateau
            plateau_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                min_lr=1e-6,
                verbose=True
            )
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, plateau_scheduler],
                milestones=[warmup_steps]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 3,
                }
            }
        else:
            raise NotImplementedError(f"Unknown lr_scheduler_mode: {self.lr_scheduler_mode}")


    def training_step(self, batch, batch_idx):
        #x, y = batch
        x = batch["his_cal"].float()
        y = batch["target"].float()
        y_hat = self(x)
        if self.visualize:
            if np.random.random() < 0.01:
                self.visualize_step(x, y, y_hat, batch_idx)
        loss = self.criterion(y_hat, y)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(
                y_hat[:, :, f, :, :],
                y[:, :, f, :, :]
            ).item()
            frame_loss_dict[f"train/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["his_cal"].float()
        y = batch["target"].float()
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.val_loss(val_loss)

        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(
                y_hat[:, :, f, :, :],
                y[:, :, f, :, :]
            ).item()
            frame_loss_dict[f"val/frame_{f}_loss"] = frame_loss
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            x = batch["his_cal"].float()  # [B, 1, input_len, H, W]
            y = batch["target"].float()
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.all_preds.append(y_hat.cpu().numpy())  # [B, forecast_len, C, H, W]
            self.all_targets.append(y.cpu().numpy())  # [B, forecast_len, C, H, W]

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_idx": batch_idx}

    def on_test_epoch_end(self):
        all_preds = np.concatenate(self.all_preds, axis=0)  # [Total_B, forecast_len, C, H, W]
        all_targets = np.concatenate(self.all_targets, axis=0)  # [Total_B, forecast_len, C, H, W]
        self.all_preds.clear()
        self.all_targets.clear()
        metrics_dict = {}
        for t in range(all_preds.shape[1]):
            pred_t = all_preds[:, t]  # [Total_B, C, H, W]
            target_t = all_targets[:, t]  # [Total_B, C, H, W]
            mae_t = np.mean(np.abs(pred_t - target_t))
            rmse_t = np.sqrt(np.mean((pred_t - target_t) ** 2))
            metrics_dict[f"test_mae_t{t}"] = float(mae_t)
            metrics_dict[f"test_rmse_t{t}"] = float(rmse_t)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        metrics_dict["test_mae_overall"] = float(mae)
        metrics_dict["test_rmse_overall"] = float(rmse)
        save_dir = Path(r"E:\research\my_code\solar_flow\results\D3Unet")
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "predictions.npy", all_preds)
        np.save(save_dir / "target.npy", all_targets)
        results_df = pd.DataFrame({
            "metric": list(metrics_dict.keys()),
            "value": list(metrics_dict.values())
        })
        results_df.to_csv(save_dir / "test_metrics.csv", index=False)
        for name, value in metrics_dict.items():
            self.log(name, value)
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



