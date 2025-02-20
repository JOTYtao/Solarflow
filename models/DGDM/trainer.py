
import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.DGDM.utils import (
    make_dir, metric, save_single_video, save_frames, AverageMeter, get_feats, compute_fvd
)
from models.DGDM.BrowBridge import BrownianBridgeModel
from models.DGDM.utils import weights_init
from models.DGDM.EMA import EMA

class BBDMRunnerPL(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.net = self.initialize_model(config)

        self.use_ema = config.model.EMA.use_ema
        if self.use_ema:

            self.ema = EMA(config.model.EMA.ema_decay)
            self.ema.register(self.net)
    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            net = BrownianBridgeModel(config)
        else:
            raise NotImplementedError("Unsupported model type")
        try:
            net.apply(weights_init)
        except Exception as e:
            print(f"Warning: Unable to apply weights initialization: {e}")
        return net

    def forward(self, x, x_cond):

        return self.net(x, x_cond)

    def training_step(self, batch, batch_idx):
        x, x_cond = batch
        loss, additional_info, cond = self.net(x, x_cond)
        total_loss = loss + cond

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cond_loss", cond, on_step=True, on_epoch=True, prog_bar=False)

        if self.use_ema and self.global_step % self.config.model.EMA.update_ema_interval == 0:
            self.ema.update(self.net)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤，计算验证损失。
        """
        x, x_cond = batch
        loss, additional_info, cond = self.net(x, x_cond)
        total_loss = loss + cond

        # 记录验证损失
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        测试步骤，用于评估模型性能。
        """
        x, x_cond = batch
        sample = self.net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised)
        sample, prediction = sample[0], sample[1]

        # 评估指标：MSE、MAE、SSIM、PSNR、LPIPS
        mse, mae, ssim, psnr = metric(
            prediction.unsqueeze(0).cpu().numpy(),
            x.unsqueeze(0).cpu().numpy(),
            mean=0,
            std=1,
            return_ssim_psnr=True
        )
        lpips_value = self.lpips_loss_fn(torch.clamp(prediction, 0, 1), torch.clamp(x, 0, 1)).mean().item()

        # 记录指标
        self.log("test_mse", mse, on_step=False, on_epoch=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True)
        self.log("test_ssim", ssim, on_step=False, on_epoch=True)
        self.log("test_psnr", psnr, on_step=False, on_epoch=True)
        self.log("test_lpips", lpips_value, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        配置优化器和调度器。
        """
        # 初始化优化器
        learning_params = [
            {"params": self.net.denoise_fn.parameters(), "lr": self.config.model.BB.optimizer.lr}
        ]
        if self.config.model.CondParams.train or self.config.model.CondParams.pretrained is None:
            learning_params.append({"params": self.net.cond_stage_model.parameters(), "lr": self.config.model.CondParams.lr})

        optimizer = Adam(
            learning_params,
            weight_decay=self.config.model.BB.optimizer.weight_decay,
            betas=(self.config.model.BB.optimizer.beta1, self.config.model.BB.optimizer.beta2)
        )

        # 初始化学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            verbose=True,
            threshold_mode="rel",
            **vars(self.config.model.BB.lr_scheduler)
        )

        # 返回优化器和调度器
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def sample(self, batch, sample_path, stage="train"):
        """
        执行采样操作，并保存结果。
        """
        x, x_cond = batch
        sample = self.net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised)
        sample, prediction = sample[0], sample[1]

        grid_size = max(x.size(1), x_cond.size(1))
        channels = self.config.data.channels

        for z, channel in enumerate(channels):
            x_cond_split = x_cond[0, :, z:z+1]
            x_split = x[0, :, z:z+1]
            sample_split = sample[:, z:z+1]
            prediction_split = prediction[:, z:z+1]

            save_single_video(x_cond_split, sample_path, f"{channel}_input.png", grid_size, to_normal=self.config.data.dataset_config.to_normal)
            save_single_video(x_split, sample_path, f"{channel}_target.png", grid_size, to_normal=self.config.data.dataset_config.to_normal)
            save_single_video(prediction_split, sample_path, f"{channel}_deter.png", grid_size, to_normal=self.config.data.dataset_config.to_normal)
            save_single_video(sample_split, sample_path, f"{channel}_proba.png", grid_size, to_normal=self.config.data.dataset_config.to_normal)