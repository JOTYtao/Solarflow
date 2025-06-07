
from datasets.utils import create_loader
import xarray as xr
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
class SolarDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 years: Dict[str, Union[List[int], List[str]]],
                 mode: str = "train",
                 input_len: int = 1 * 4,
                 pred_len: int = 3 * 4,
                 stride=1,
                 validation: bool = True,
                 use_possible_starts: bool = True,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.years = years[mode]
        self.data_path = data_path
        self.data_paths = {
            str(year): os.path.join(data_path, mode, str(year), f"CSI_{year}.nc")
            for year in self.years
        }
        self.valid_indices = {}
        self.nitems_per_year = {}
        self.timestamps = {}
        self.stride = stride
        self.input_len = input_len
        self.pred_len = pred_len
        self.validation = validation
        self.possible_starts = {}
        self.nitems = 0
        self.use_possible_starts = use_possible_starts
        # Load timestamps and possible_starts for each year
        self._load_auxiliary_files()
        self.total_len = self.input_len + self.pred_len

        self._initialize_indices()

        if self.validation:
            np.random.seed(0)
            self.seeds = np.random.randint(0, 1000000, self.nitems)
    def _load_auxiliary_files(self):
        """Load timestamps and possible_starts for each year."""
        for year in self.years:
            year_str = str(year)
            year_folder = os.path.join(self.data_path, self.mode, year_str)
            timestamps_path = os.path.join(year_folder, f"CSI_{year_str}_timestamps.csv")

            if os.path.exists(timestamps_path):
                self.timestamps[year_str] = pd.read_csv(timestamps_path)
            else:
                raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

            if self.use_possible_starts:
                possible_starts_path = os.path.join(year_folder, f"CSI_{year_str}_possible_starts.npy")
                if os.path.exists(possible_starts_path):
                    self.possible_starts[year_str] = np.load(possible_starts_path)
                else:
                    raise FileNotFoundError(f"Possible starts file not found: {possible_starts_path}")

    def _initialize_indices(self):
        """Initialize indices for valid samples using possible starts."""
        self.nitems = 0
        self.year_mapping = {}
        for year in self.years:
            year_str = str(year)
            with xr.open_dataset(self.data_paths[year_str]) as ds:
                if self.use_possible_starts and year_str in self.possible_starts:
                    valid_indices = self.possible_starts[year_str]
                    valid_indices = valid_indices[valid_indices <= len(ds.time) - self.total_len]
                else:
                    valid_indices = np.arange(0, len(ds.time) - self.total_len + 1, self.stride)
                self.valid_indices[year_str] = valid_indices
                self.nitems_per_year[year_str] = len(valid_indices)
                for i, idx in enumerate(valid_indices):
                    self.year_mapping[self.nitems + i] = (year_str, idx)
                self.nitems += len(valid_indices)

    def normalize_coords(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #    lon_norm = 2 * ((lon + 180) / 360) - 1
        #    lat_norm = 2 * ((lat + 90) / 180) - 1
        lon_norm = 2 * np.pi * (lon - lon.min()) / (lon.max() - lon.min())
        lat_norm = 2 * np.pi * (lat - lat.min()) / (lat.max() - lat.min())
        return lon_norm, lat_norm
    def _prepare_coords(self, ds: xr.Dataset) -> torch.Tensor:
        lon = ds.longitude.values if 'longitude' in ds else ds.lon.values
        lat = ds.latitude.values if 'latitude' in ds else ds.lat.values
        lon_norm, lat_norm = self.normalize_coords(lon, lat)
        lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm, indexing='xy')
        sin_lon = np.sin(lon_grid)[np.newaxis, np.newaxis, :, :]
        cos_lon = np.cos(lon_grid)[np.newaxis, np.newaxis, :, :]
        sin_lat = np.sin(lat_grid)[np.newaxis, np.newaxis, :, :]
        cos_lat = np.cos(lat_grid)[np.newaxis, np.newaxis, :, :]
        coords = np.concatenate([sin_lon, cos_lon, sin_lat, cos_lat], axis=0)
        coords = np.repeat(coords, self.input_len, axis=1)
        return torch.tensor(coords, dtype=torch.float16)
    def _prepare_time_coords(self, year: str, start_idx: int, idx_end: int) -> torch.Tensor:
        time_slice = self.timestamps[year].iloc[start_idx:start_idx+self.input_len]['StartTimeUTC'].values
        time_pd = pd.to_datetime(time_slice)
        months = time_pd.month.values / 12.0
        days = time_pd.day.values / 31.0
        hours = time_pd.hour.values / 24.0
        minutes = time_pd.minute.values / 60.0
        time_features = torch.tensor(np.stack([months, days, hours, minutes]), dtype=torch.float16)
        return time_features

    def __len__(self) -> int:
        return self.nitems
    def rescale_data(self, data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.2) -> torch.Tensor:
        return 2 * ((data - min_val) / (max_val - min_val)) - 1
    def inverse_rescale_data(self, scaled_data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.2) -> torch.Tensor:
        return ((scaled_data + 1) / 2) * (max_val - min_val) + min_val
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.validation:
            np.random.seed(self.seeds[idx])
        year_str, start_idx = self.year_mapping[idx]
        idx_end = (start_idx + self.input_len + self.pred_len)
        with xr.open_dataset(self.data_paths[year_str]) as ds:
            times = ds.time.values[start_idx:idx_end]  # 切片时间维度
            lons = ds.lon.values
            lats = ds.lat.values
            K = ds.CAL.values
            K_clip = K[start_idx:idx_end]
            data = torch.tensor(K_clip, dtype=torch.float16).unsqueeze(0)
            data = self.rescale_data(data, min_val=0.0, max_val=1.2)
            cached_coords = self._prepare_coords(ds)
            H, W = ds.CAL.shape[-2:]
            time_features = self._prepare_time_coords(year_str, start_idx, idx_end)
            time_coords = time_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            times_stamp = times.astype(np.int64)
        x = data[:, :self.input_len, :, :]
        y = data[:, self.input_len:, :, :]
        times_x = torch.tensor(times_stamp[:self.input_len], dtype=torch.long)
        times_y = torch.tensor(times_stamp[self.input_len:], dtype=torch.long)
        lons = torch.tensor(lons, dtype=torch.float32)
        lats = torch.tensor(lats, dtype=torch.float32)
        
        return {
            "his": x,
            "spatial_coordinates": cached_coords,
            "time_coordinates": time_coords,
            "target": y,
            "times_input": times_x,
            "times_target": times_y,
            "lons": lons,
            "lats": lats
            }


def load_data(batch_size, val_batch_size, data_root,
              num_workers=4, input_len=4, pred_len=12,
              distributed=False, use_possible_starts=True,
              use_prefetcher=False, drop_last=False):
    years = {
        'train': [2017, 2018, 2019, 2020],
        'val': [2021],
        'test': [2022]
    }

    train_set = SolarDataset(
        data_path=data_root,
        years=years,
        input_len=input_len,
        pred_len=pred_len,
        mode='train',
        use_possible_starts=use_possible_starts
    )
    val_set = SolarDataset(
        data_path=data_root,
        years=years,
        input_len=input_len,
        pred_len=pred_len,
        mode='val',
        use_possible_starts=use_possible_starts
    )
    test_set = SolarDataset(
        data_path=data_root,
        years=years,
        input_len=input_len,
        pred_len=pred_len,
        mode='test',
        use_possible_starts=use_possible_starts
    )

    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )

    dataloader_val = create_loader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )

    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )

    return dataloader_train, dataloader_val, dataloader_test

class SolarDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root: str,
            input_len: int = 4,
            pred_len: int = 12,
            batch_size: int = 16,
            val_batch_size: int = 16,
            num_workers: int = 4,
            distributed: bool = False,
            use_possible_starts: bool = True,
            use_prefetcher: bool = False,
            drop_last: bool = False,
            years: dict = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.input_len = input_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.use_possible_starts = use_possible_starts
        self.use_prefetcher = use_prefetcher
        self.drop_last = drop_last
        self.years = years if years is not None else {
            'train': [2017, 2018, 2019, 2020],
            'val': [2021],
            'test': [2022]
        }

    def setup(self, stage: Optional[str] = None):
        self.train_loader, self.val_loader, self.test_loader = load_data(
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            data_root=self.data_root,
            num_workers=self.num_workers,
            input_len=self.input_len,
            pred_len=self.pred_len,
            distributed=self.distributed,
            use_possible_starts=self.use_possible_starts,
            use_prefetcher=self.use_prefetcher,
            drop_last=self.drop_last
        )
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

if __name__ == '__main__':
    dataloader_train, dataloader_val, dataloader_test = \
        load_data(
            batch_size=16,
            val_batch_size=4,
            data_root='E:/research/my_code/Solar_D2P/data',
            num_workers=4,
            input_len=4,
            pred_len=12,
            use_possible_starts=True
        )
    print("\nDataset Statistics:")
    print(f"Training samples: {len(dataloader_train.dataset)}")
    print(f"Validation samples: {len(dataloader_val.dataset)}")
    print(f"Testing samples: {len(dataloader_test.dataset)}")
    print("\nDataloader Statistics:")
    print(f"Training batches: {len(dataloader_train)}")
    print(f"Validation batches: {len(dataloader_val)}")
    print(f"Testing batches: {len(dataloader_test)}")
    print("\nFirst Batch Shapes:")
    for batch in dataloader_train:
        print("Input shape:", batch["his_cal"].shape)
        print("Target shape:", batch["target"].shape)
        print("Spatial coords shape:", batch["spatial_coordinates"].shape)
        print("Time coords shape:", batch["time_coordinates"].shape)
        break
    print("\nSamples per Year:")
    print("Training years:")
    for year in dataloader_train.dataset.years:
        samples = dataloader_train.dataset.nitems_per_year[str(year)]
        print(f"Year {year}: {samples} samples")
    print("\nValidation years:")
    for year in dataloader_val.dataset.years:
        samples = dataloader_val.dataset.nitems_per_year[str(year)]
        print(f"Year {year}: {samples} samples")

    print("\nTest years:")
    for year in dataloader_test.dataset.years:
        samples = dataloader_test.dataset.nitems_per_year[str(year)]
        print(f"Year {year}: {samples} samples")