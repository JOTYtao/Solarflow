from typing import Any, Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from models.layers.loss import get_loss
from typing import Any, Type
from models.layers.ConvLSTM import ConvLSTMCell
import numpy as np
import neptune.types
from torchmetrics import MeanMetric
import torchvision.transforms
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from models.utils import warmup_lambda

REGISTERED_MODELS = {}


def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls


@register_model
class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(
            self,
            hidden_dim: int = 64,
            input_channels: int = 7,
            out_channels: int = 1,
            forecast_steps: int = 6,
            lr: float = 0.001,
            visualize: bool = False,
            loss: Union[str, torch.nn.Module] = "mse",
            # pretrained: bool = False,
            conv_type: str = "standard",
            save_dir: str = None,
            max_epochs: int = 1000,
            lr_scheduler_mode: str = "plateau",
    ):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.save_hyperparameters()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_steps = forecast_steps
        self.criterion = get_loss(loss)
        self.lr = lr
        self.visualize = visualize
        self.input_channels = input_channels
        self.output_channels = out_channels
        self.model = ConvLSTM(input_channels, hidden_dim, out_channels, conv_type=conv_type)
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
        return EncoderDecoderConvLSTM(
            input_channels=config.get("in_channels"),
            hidden_dim=config.get("hidden_dim"),
            out_channels=config.get("out_channels"),
            forecast_steps=config.get("forecast_steps"),
            lr=config.get("lr"),
            lr_scheduler_mode=config.get("lr_scheduler_mode"),
            conv_type=config.get("conv_type"),
            visualize=config.get("visualize"),
            max_epochs=config.get("max_epochs"),
            save_dir=config.get("save_dir")
        )

    def setup(self, stage='test'):
        if stage == 'test':
            self.test_loader = self.trainer.datamodule.test_dataloader()

    def forward(self, x, future_seq=0, hidden_state=None):
        return self.model.forward(x, future_seq, hidden_state)

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
                    "frequency": 1,
                }
            }
        else:
            raise NotImplementedError(f"Unknown lr_scheduler_mode: {self.lr_scheduler_mode}")

    def training_step(self, batch, batch_idx):
        # x, y = batch
        his_sis = batch["his_cal"].float()  # [B, 1, input_len, H, W]
        spatial_coords = batch["spatial_coordinates"].float()  # [B, 4, input_len, H, W]
        time_coords = batch["time_coordinates"].float()  # [B, 4, input_len, H, W]
        y = batch["target"].float()
        x = torch.cat([his_sis, spatial_coords, time_coords], dim=1)
        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)

        y_hat = self(x, self.forecast_steps)
        y_hat = torch.permute(y_hat, dims=(0, 2, 1, 3, 4))
        # Generally only care about the center x crop, so the model can take into account the clouds in the area without
        # being penalized for that, but for now, just do general MSE loss, also only care about first 12 channels
        # the logger you used (in this case tensorboard)
        if self.visualize:
            if np.random.random() < 0.01:
                self.visualize_step(x, y, y_hat, batch_idx)
        loss = self.criterion(y_hat, y)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        frame_loss_dict = {}
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :, :], y[:, f, :, :, :]).item()
            frame_loss_dict[f"train/frame_{f}_loss"] = frame_loss
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        his_sis = batch["his_cal"].float()  # [B, 1, input_len, H, W]
        spatial_coords = batch["spatial_coordinates"].float()  # [B, 4, input_len, H, W]
        time_coords = batch["time_coordinates"].float()  # [B, 4, input_len, H, W]
        y = batch["target"].float()
        x = torch.cat([his_sis, spatial_coords, time_coords], dim=1)
        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)
        y_hat = self(x, self.forecast_steps)
        y_hat = torch.permute(y_hat, dims=(0, 2, 1, 3, 4))
        val_loss = self.criterion(y_hat, y)
        self.val_loss(val_loss)
        # Save out loss per frame as well
        frame_loss_dict = {}
        # y_hat = torch.moveaxis(y_hat, 2, 1)
        for f in range(self.forecast_steps):
            frame_loss = self.criterion(y_hat[:, f, :, :, :], y[:, f, :, :, :]).item()
            frame_loss_dict[f"val/frame_{f}_loss"] = frame_loss
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(frame_loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            his_sis = batch["his_cal"].float()  # [B, 1, input_len, H, W]
            spatial_coords = batch["spatial_coordinates"].float()  # [B, 4, input_len, H, W]
            time_coords = batch["time_coordinates"].float()  # [B, 4, input_len, H, W]
            y = batch["target"].float()
            x = torch.cat([his_sis, spatial_coords, time_coords], dim=1)
            x = x.permute(0, 2, 1, 3, 4)
            y = y.permute(0, 2, 1, 3, 4)
            y_hat = self(x, self.forecast_steps)
            y_hat = torch.permute(y_hat,
                                  dims=(0, 2, 1, 3, 4))  # [B, C, forecast_len, H, W] -> [B, forecast_len, C, H, W]
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

        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.save_dir / "predictions.npy", all_preds)
        np.save(self.save_dir / "target.npy", all_targets)
        results_df = pd.DataFrame({
            "metric": list(metrics_dict.keys()),
            "value": list(metrics_dict.values())
        })
        results_df.to_csv(self.save_dir / "test_metrics.csv", index=False)
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


class ConvLSTM(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, out_channels, conv_type: str = "standard"):
        super().__init__()
        """ ARCHITECTURE
        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model
        """
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.encoder_3_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.decoder_3_convlstm = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            bias=True,
            conv_type=conv_type,
        )

        self.decoder_CNN = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

    def autoencoder(self, x, seq_len, future_step, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3,
                    h_t4, c_t4, h_t5, c_t5, h_t6, c_t6):
        outputs = []

        # encoder
        for t in range(seq_len):
            h_t1, c_t1 = self.encoder_1_convlstm(
                input_tensor=x[:, t, :, :], cur_state=[h_t1, c_t1]
            )
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t1, cur_state=[h_t2, c_t2]
            )
            h_t3, c_t3 = self.encoder_3_convlstm(
                input_tensor=h_t2, cur_state=[h_t3, c_t3]
            )

        # encoder_vector
        encoder_vector = h_t3

        # decoder
        for t in range(future_step):
            h_t4, c_t4 = self.decoder_1_convlstm(
                input_tensor=encoder_vector, cur_state=[h_t4, c_t4]
            )
            h_t5, c_t5 = self.decoder_2_convlstm(
                input_tensor=h_t4, cur_state=[h_t5, c_t5]
            )
            h_t6, c_t6 = self.decoder_3_convlstm(
                input_tensor=h_t5, cur_state=[h_t6, c_t6]
            )

            encoder_vector = h_t6
            outputs += [h_t6]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, forecast_steps=0, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t1, c_t1 = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        h_t4, c_t4 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t5, c_t5 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t6, c_t6 = self.decoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(
            x, seq_len, forecast_steps,
            h_t1, c_t1, h_t2, c_t2, h_t3, c_t3,
            h_t4, c_t4, h_t5, c_t5, h_t6, c_t6
        )

        return outputs
