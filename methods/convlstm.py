
from models import ConvLSTM_Model
import torch
from models import PredRNN_Model
from utils import reshape_patch, reshape_patch_back, reserve_schedule_sampling_exp, schedule_sampling
from core import metric
import numpy as np
import os.path as osp
import os
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Type
from utils import print_log, check_dir
from core import timm_schedulers
from core import metric
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import neptune.types
REGISTERED_MODELS = {}

def register_model(cls: Type[pl.LightningModule]):
    global REGISTERED_MODELS
    name = cls.__name__
    assert name not in REGISTERED_MODELS, f"exists class: {REGISTERED_MODELS}"
    REGISTERED_MODELS[name] = cls
    return cls
@register_model
class ConvLSTM(pl.LightningModule):
    r"""ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """
    def __init__(
        self,
        metrics: list,
        spatial_norm: bool = False,
        channel_names: Optional[list] = None,
        criterion: nn.Module = nn.MSELoss(),
        num_hidden: str = '128,128,128,128',
        in_shape: list = [16, 1, 128, 128],
        patch_size: int = 4,
        filter_size: int = 5,
        stride: int = 1,
        layer_norm: bool = True,
        input_len: int = 4,
        pred_len: int = 12,
        lr: float = 0.001,
        reverse_scheduled_sampling: int = 0,
        r_sampling_step_1: int = 25000,
        r_sampling_step_2: int = 50000,
        r_exp_alpha: int = 50000,
        scheduled_sampling: int = 1,
        sampling_stop_iter: int = 50000,
        sampling_changing_rate: float = 0.00002,
        save_dir: str = "experiments/convlstm",
        **kwargs
    ):

        super().__init__()
        self.metric_list = metrics
        self.spatial_norm = spatial_norm
        self.channel_names = channel_names
        self.save_hyperparameters()
        self.num_hidden = num_hidden
        self.in_shape = in_shape
        self.patch_size = patch_size
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_length = self.input_len + self.pred_len
        self.reverse_scheduled_sampling = reverse_scheduled_sampling
        self.r_sampling_step_1 = r_sampling_step_1
        self.r_sampling_step_2 = r_sampling_step_2
        self.r_exp_alpha = r_exp_alpha
        self.scheduled_sampling = scheduled_sampling
        self.sampling_stop_iter = sampling_stop_iter
        self.sampling_changing_rate = sampling_changing_rate
        self.model = self._build_model()
        self.criterion = criterion
        self.lr = lr
        self.test_outputs = []
        self.save_dir = self._init_save_dir(save_dir)
        self.eta = 1.0
        
    @classmethod
    def from_config(cls, config):
        return ConvLSTM(
            metrics=config.get('metrics', ['mae', 'mse']),
            num_hidden=config.get('num_hidden', '128,128,128,128'),
            in_shape=config.get('in_shape', [16, 1, 128, 128]),
            patch_size=config.get('patch_size', 4),
            filter_size=config.get('filter_size', 5),
            stride=config.get('stride', 1),
            layer_norm=config.get('layer_norm', True),
            input_len=config.get('input_len', 4),
            pred_len=config.get('pred_len', 12),
            lr=config.get('lr', 0.001),
            reverse_scheduled_sampling=config.get('reverse_scheduled_sampling', 0),
            r_sampling_step_1=config.get('r_sampling_step_1', 25000),
            r_sampling_step_2=config.get('r_sampling_step_2', 50000),
            r_exp_alpha=config.get('r_exp_alpha', 50000),
            scheduled_sampling=config.get('scheduled_sampling', 1),
            sampling_stop_iter=config.get('sampling_stop_iter', 50000),
            sampling_changing_rate=config.get('sampling_changing_rate', 0.00002),
            save_dir=config.get('save_dir', "experiments/convlstm")
    )
    def _build_model(self):
        num_hidden = [int(x) for x in self.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return ConvLSTM_Model(
            num_layers=num_layers,
            num_hidden=num_hidden,
            in_shape=self.in_shape,
            patch_size=self.patch_size,
            filter_size=self.filter_size,
            stride=self.stride,
            layer_norm=self.layer_norm,
            input_len=self.input_len,
            pred_len=self.pred_len,
            reverse_scheduled_sampling=self.reverse_scheduled_sampling
        )
    def _init_save_dir(self, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        saved_dir = osp.join(save_dir, 'pred_sample')
        os.makedirs(saved_dir, exist_ok=True)
        return saved_dir
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=200,
            lr_min=1e-6,
            warmup_lr_init=1e-5,
            warmup_t=0,
            t_in_epochs=True,
            k_decay=1.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch_x, batch_y):
        # reverse schedule sampling
        if self.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.input_len
        _, img_channel, img_height, img_width = self.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = reshape_patch(test_ims, self.patch_size)
        test_ims = test_ims[:, :, :, :, :img_channel]

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.total_length - mask_input - 1,
            img_height // self.patch_size,
            img_width // self.patch_size,
            self.patch_size ** 2 * img_channel)).to(self.device)
            
        if self.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.input_len - 1, :, :] = 1.0

        img_gen, _ = self.model(test_dat, real_input_flag, return_loss=False)
        img_gen = reshape_patch_back(img_gen, self.patch_size)
        pred_y = img_gen[:, -self.pred_len:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y
    def visualize_predictions(self, inputs, trues, preds, times, batch_idx, step="train"):
        import matplotlib.pyplot as plt
        import torchvision.utils as vutils
        max_samples = min(3, inputs.shape[0])
        for sample_idx in range(max_samples):
        
            fig, axes = plt.subplots(2, self.pred_len, figsize=(20, 8))
            fig.suptitle(f'Sample {sample_idx} at {times[sample_idx]}')
            true_seq = trues[sample_idx].squeeze()
            pred_seq = preds[sample_idx].squeeze()
            vmin = min(true_seq.min(), pred_seq.min())
            vmax = max(true_seq.max(), pred_seq.max())
            for t in range(self.pred_len):
                im1 = axes[0, t].imshow(true_seq[t], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, t].set_title(f'True t={t}')
                plt.colorbar(im1, ax=axes[0, t])
                im2 = axes[1, t].imshow(pred_seq[t], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, t].set_title(f'Pred t={t}')
                plt.colorbar(im2, ax=axes[1, t])
            plt.tight_layout()
            self.logger.experiment[f"{step}/predictions/sample_{sample_idx}_batch_{batch_idx}"].log(
                neptune.types.File.as_image(fig)
            )
            plt.close(fig)
    
    def training_step(self, batch, batch_idx):
        input = batch["his"].permute(0, 2, 1, 3, 4).contiguous()
        target = batch["target"].permute(0, 2, 1, 3, 4).contiguous()
        ims = torch.cat([input, target], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        ims = reshape_patch(ims, self.patch_size)
        if self.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(
                self.global_step, ims.shape[0], self.in_shape, self.r_sampling_step_1, self.r_sampling_step_2, self.r_exp_alpha, self.input_len, self.pred_len, self.patch_size, self.total_length, self.device)
        else:
            self.eta, real_input_flag = schedule_sampling(
                self.eta, self.global_step, ims.shape[0], self.in_shape, self.scheduled_sampling, self.sampling_stop_iter, self.sampling_changing_rate, self.pred_len, self.patch_size, self.device)
            
        img_gen, loss = self.model(ims, real_input_flag)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % 100 == 0:
            pred_y = self(input, target)
            self.visualize_predictions(
                input.cpu().numpy(),
                target.cpu().numpy(),
                pred_y.detach().cpu().numpy(),
                batch['times_target'],
                batch_idx,
                step="train"
            )
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch["his"].permute(0, 2, 1, 3, 4).contiguous()
        target = batch["target"].permute(0, 2, 1, 3, 4).contiguous()
        pred_y = self(input, target)
        loss = self.criterion(pred_y, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        if batch_idx % 100 == 0:
            pred_y = self(input, target)
            self.visualize_predictions(
                input.cpu().numpy(),
                target.cpu().numpy(),
                pred_y.detach().cpu().numpy(),
                batch['times_target'],
                batch_idx,
                step="val"
            )
        return loss

    def test_step(self, batch, batch_idx):
        input = batch["his"].permute(0, 2, 1, 3, 4).contiguous()
        target = batch["target"].permute(0, 2, 1, 3, 4).contiguous()
        pred_y = self(input, target)
        
        times_input = batch['times_input'].cpu().numpy().astype('datetime64[ns]')
        times_target = batch['times_target'].cpu().numpy().astype('datetime64[ns]')
        outputs = {
            'inputs': input.cpu().numpy(), 
            'preds': pred_y.cpu().numpy(), 
            'trues': target.cpu().numpy(),
            'times_input': times_input,
            'times_target': times_target,
            'lons': batch['lons'].cpu().numpy(),
            'lats': batch['lats'].cpu().numpy()
        }
        self.test_outputs.append(outputs)
        return outputs
    def inverse_rescale_data(self, data: np.ndarray, min_val: float = 0.0, max_val: float = 1.2) -> np.ndarray:
        return (data + 1) * (max_val - min_val) / 2 + min_val

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            if k in ['inputs', 'preds', 'trues']:
                results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
            elif k in ['times_input', 'times_target']:
                results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
            elif k in ['lons', 'lats']:
                results_all[k] = self.test_outputs[0][k]
        results_inverse = {
            'inputs': self.inverse_rescale_data(results_all['inputs']),
            'preds': self.inverse_rescale_data(results_all['preds']),
            'trues': self.inverse_rescale_data(results_all['trues']),
            'times_input': results_all['times_input'],
            'times_target': results_all['times_target'],
            'lons': results_all['lons'],
            'lats': results_all['lats']
        }
        eval_res_inverse, eval_log_inverse = metric(
            results_inverse['preds'], 
            results_inverse['trues'], 
            metrics=self.metric_list, 
            channel_names=self.channel_names, 
            spatial_norm=self.spatial_norm
        )
        results_inverse['metrics'] = np.array([eval_res_inverse['mae'], eval_res_inverse['mse']])
        if self.trainer.is_global_zero:
            for key in results_inverse.keys():
                np.save(os.path.join(self.save_dir, f'{key}.npy'), results_inverse[key])
        return results_all