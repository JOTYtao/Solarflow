import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.checkpoint import checkpoint
from models.DGMR.layer_dgmr.ConditionStack import ContextConditioningStack, LatentConditioningStack
from models.DGMR.Discriminator import Discriminator
from models.DGMR.Generator import Generator, Sampler
from pytorch_lightning import Trainer
from models.DGMR.layer_dgmr.losses import (
    GridCellLoss,
    NowcastingLoss,
    grid_cell_regularizer,
    loss_hinge_disc,
    loss_hinge_gen,
)
from torchmetrics import MeanMetric
import numpy as np
import neptune.types
from models.layers.loss import get_loss
from torchmetrics import MeanMetric
from pathlib import Path
import pandas as pd
from typing import Union, Type
REGISTERED_MODELS = {}


def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls

def weight_fn(y, max_weight=2.0, epsilon=0.1):
    weights = torch.log(max_weight - y + epsilon)
    return torch.clamp(weights, min=0.1)

class DGMR(pl.LightningModule):
    """Deep Generative Model"""

    def __init__(
        self,
        forecast_steps: int = 6,
        input_channels: int = 1,
        output_shape: int = 128,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        visualize: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 384,
        context_channels: int = 192,
        generation_steps: int = 6,
        loss: Union[str, torch.nn.Module] = "mse",
        input_steps: int = 8,
        save_dir: str = None,
        **kwargs,
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels

        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            generation_steps: Number of generation steps to use in forward pass, in paper is 6 and the best is chosen for the loss
                this results in huge amounts of GPU memory though, so less might work better for training.
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super().__init__()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss(weight_fn=weight_fn)
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.generation_steps = generation_steps
        self.criterion = get_loss(loss)
        self.test_loss = MeanMetric()
        self.test_step_outputs = []
        self.save_dir = Path(save_dir)
        self.input_steps=input_steps
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
            num_context_steps=self.input_steps
        )
        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        self.discriminator = Discriminator(input_channels)
        self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.test_loader = None
        torch.autograd.set_detect_anomaly(True)
        
    @classmethod
    def from_config(cls, config):
        return DGMR(
            input_channels=config.get("input_channels"),
            output_shape=config.get("output_shape"),
            gen_lr=config.get("gen_lr"),
            disc_lr = config.get("disc_lr"),
            conv_type=config.get("conv_type"),
            num_samples=config.get("num_samples"),
            forecast_steps=config.get("forecast_steps"),
            grid_lambda=config.get("grid_lambda"),
            beta1=config.get("beta1"),
            beta2=config.get("beta2"),
            latent_channels=config.get("latent_channels"),
            context_channels=config.get("context_channels"),
            visualize=config.get("visualize"),
            save_dir=config.get("save_dir"),
            generation_steps=config.get("generation_steps"),
            input_steps=config.get("input_steps")
        )

    def forward(self, x):
        x = self.generator(x)
        return x
    def setup(self, stage='test'):
        if stage == 'test':
            self.test_loader = self.trainer.datamodule.test_dataloader()

    def training_step(self, batch, batch_idx):
        his_sis = batch["his_cal"].float()  # [B, 1, input_len, H, W]
        future_images = batch["target"].float()
        future_images = future_images.permute(0, 2, 1, 3, 4) 
        images = his_sis.permute(0, 2, 1, 3, 4)

        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()

        ##########################
        # Optimize Discriminator #
        ##########################
        for _ in range(2):
            d_opt.zero_grad()
            predictions = checkpoint(self.forward, images, use_reentrant=False)

            generated_sequence = torch.cat([images, predictions], dim=1) 
            real_sequence = torch.cat([images, future_images], dim=1)
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], generated_sequence.shape[0]], dim=0
            )

            score_real_spatial, score_real_temporal = torch.split(score_real, 1, dim=1)
            score_generated_spatial, score_generated_temporal = torch.split(
                score_generated, 1, dim=1
            )

            discriminator_loss = loss_hinge_disc(
                score_generated_spatial, score_real_spatial
            ) + loss_hinge_disc(score_generated_temporal, score_real_temporal)

            self.manual_backward(discriminator_loss)
            d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        predictions = [
            checkpoint(self.forward, images, use_reentrant=False)
            for _ in range(self.generation_steps)
        ]

        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)

        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)

        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
            )
            generated_scores.append(score_generated)

        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()
        self.log_dict(
            {
                "train/d_loss": discriminator_loss,
                "train/g_loss": generator_loss,
                "train/grid_loss": grid_cell_reg,
            },
            prog_bar=True,
        )
        generated_images = self(images)
        if self.visualize:
            self.visualize_step(
                images, future_images, generated_images,
                self.global_iteration, step="train"
            )

    def validation_step(self, batch, batch_idx):
        his_sis = batch["his_cal"].float()
        future_images = batch["target"].float()
        future_images = future_images.permute(0, 2, 1, 3, 4)
        images = his_sis.permute(0, 2, 1, 3, 4)
        ##########################
        # Evaluate Discriminator #
        ##########################
        for _ in range(2):
            predictions = self(images)
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], generated_sequence.shape[0]], dim=0
            )
            score_real_spatial, score_real_temporal = torch.split(score_real, 1, dim=1)
            score_generated_spatial, score_generated_temporal = torch.split(
                score_generated, 1, dim=1
            )
            discriminator_loss = loss_hinge_disc(
                score_generated_spatial, score_real_spatial
            ) + loss_hinge_disc(score_generated_temporal, score_real_temporal)

        ######################
        # Evaluate Generator #
        ######################
        predictions = [self(images) for _ in range(self.generation_steps)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)

        generated_sequence = [torch.cat([images[:, :1], x], dim=1) for x in predictions]
        real_sequence = torch.cat([images[:, :1], future_images], dim=1)

        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
            )
            generated_scores.append(score_generated)

        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        self.log_dict(
            {
                "val/d_loss": discriminator_loss,
                "val/g_loss": generator_loss,
                "val/grid_loss": grid_cell_reg,
            },
            prog_bar=True,
        )
    def test_step(self, batch, batch_idx):
        his_sis = batch["his_cal"].float()
        future_images = batch["target"].float()
        future_images = future_images.permute(0, 2, 1, 3, 4) 
        images = his_sis.permute(0, 2, 1, 3, 4)

        predictions = [self(images) for _ in range(self.generation_steps)]
        y_hat = torch.stack(predictions, dim=0).mean(dim=0) 
        loss = self.criterion(y_hat, future_images)


        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        output = {
            "loss": loss,
            "predictions": y_hat.detach().cpu(),
            "targets": future_images.detach().cpu(),
            "batch_idx": batch_idx
        }
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        all_preds = torch.cat([x["predictions"] for x in outputs])
        all_targets = torch.cat([x["targets"] for x in outputs])

        all_preds = all_preds.squeeze(2)
        all_targets = all_targets.squeeze(2)
        metrics_dict = {}
        mae = torch.mean(torch.abs(all_preds - all_targets))
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
        metrics_dict.update({
            "test_mae": mae.item(),
            "test_rmse": rmse.item(),
        })
        for name, value in metrics_dict.items():
            self.log(name, value)
        save_dir = Path(r"E:\research\my_code\solar_flow\results\DGRM")
        save_dir.mkdir(exist_ok=True, parents=True)

        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        np.save(save_dir / "predictions.npy", preds_np)
        np.save(save_dir / "targets.npy", targets_np)
        results_df = pd.DataFrame({
            "metric": list(metrics_dict.keys()),
            "value": list(metrics_dict.values())
        })
        results_df.to_csv(save_dir / "test_metrics.csv", index=False)
        self.test_step_outputs.clear()
        return metrics_dict

    def visualize_step(self, x, y, y_hat, batch_idx, step="train"):
        images = x[0].cpu().detach()
        future_images = y[0].cpu().detach()
        generated_images = y_hat[0].cpu().detach()
         
        for i, t in enumerate(images):
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)  #
            
            image_grid = torchvision.transforms.ToPILImage()(image_grid)
            
            self.logger.experiment[f"{step}/Input_Frame_{i}"].log(
                    neptune.types.File.as_image(image_grid)
                )
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            image_grid = torchvision.transforms.ToPILImage()(image_grid)
            self.logger.experiment[f"{step}/Target_Frame_{i}"].log(
                    neptune.types.File.as_image(image_grid)
                )
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            image_grid = torchvision.transforms.ToPILImage()(image_grid)
            self.logger.experiment[f"{step}/Generated_Frame_{i}"].log(
                    neptune.types.File.as_image(image_grid))
            


    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))
        scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=1000, gamma=0.95)
        scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=1000, gamma=0.95)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]
