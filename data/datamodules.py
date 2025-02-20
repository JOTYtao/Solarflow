import logging
from data.datasets import SISDataset
_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)
from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import pandas as pd
class SIS_DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_config = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = SISDataset(
                **self.dataset_config,
                mode="train",
            )
            self.data_val = SISDataset(
                **self.dataset_config,
                mode="val",
            )
            self.data_test = SISDataset(
                **self.dataset_config,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    @property
    def num_train_samples(self):
        """Returns the number of samples in the training dataset."""
        return len(self.data_train) if self.data_train else 0

    @property
    def num_val_samples(self):
        """Returns the number of samples in the validation dataset."""
        return len(self.data_val) if self.data_val else 0

    @property
    def num_test_samples(self):
        """Returns the number of samples in the test dataset."""
        return len(self.data_test) if self.data_test else 0




