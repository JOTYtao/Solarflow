import torch
import pytorch_lightning as pl
import numpy as np
from typing import Union, Type
from models.layers.loss import get_loss
from pathlib import Path
from data.datamodules import SIS_DataModule
import os
import pandas as pd
from torchmetrics import MeanMetric

def compute_metrics(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }
def evaluate_persistence_model(model, dataloader):
    all_preds = []
    all_targets = []
    pred_path = Path(model.save_dir) / "predictions"
    pred_path.mkdir(parents=True, exist_ok=True)
    for batch in dataloader:
        in_seq = batch["his_cal"]
        out_seq = batch["target"]
        with torch.no_grad():
            preds = model(in_seq)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(out_seq.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    np.save(pred_path / "all_predictions.npy", all_preds)
    np.save(pred_path / "all_targets.npy", all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    print(f"Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics
class Persistence(pl.LightningModule):

    def __init__(self,
                forecast_steps: int = 6,
                save_dir: str = None,
        ):
        super(Persistence, self).__init__()
        self.t_axis = 2  # (B, C, T, H, W)
        self.forecast_steps = forecast_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    def forward(self, in_seq):
        output = in_seq[:, :, -1:, :, :]
        output = torch.repeat_interleave(output, repeats=self.forecast_steps, dim=self.t_axis)
        return output


def verify_dataloader_shapes(config):
    # 创建 DataModule
    datamodule = SIS_DataModule(
        dataset=config,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    datamodule.setup()

    print(f"Train samples: {datamodule.num_train_samples}")
    print(f"Validation samples: {datamodule.num_val_samples}")
    print(f"Test samples: {datamodule.num_test_samples}")

    test_loader = datamodule.test_dataloader()
    model = Persistence(
        forecast_steps=config["pred_len"],
        save_dir=r"E:\research\my_code\solar_flow\results\Persistence"
    )
    test_results = evaluate_persistence_model(model, test_loader)
    print("Test Results:", test_results)

if __name__ == '__main__':
    dataset_config = {
        "data_path": "E:/research/my_code/solar_flow/data",
        "years": {
            "train": [2017, 2018, 2019, 2020],
            "val": [2021],
            "test": [2022]
        },
        "input_len": 8,
        "pred_len": 8,
        "stride": 1,
        "forecast": True,
        "use_possible_starts": True,
        "batch_size": 16,
        "num_workers": 10,
        "pin_memory": True
    }
    verify_dataloader_shapes(dataset_config)