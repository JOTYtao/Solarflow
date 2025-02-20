import xarray as xr

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

class SISDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 years: Dict[str, Union[List[int], List[str]]],
                 mode: str = "train",
                 input_len: int = 2 * 4,
                 pred_len: int = 2 * 4,
                 stride=1,
                 validation: bool = True,
                 use_possible_starts: bool = True,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.years = years[mode]
        self.data_path = data_path
        self.data_paths = {
            str(year): os.path.join(data_path, mode, str(year), f"CAL_{year}.nc")
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
            timestamps_path = os.path.join(year_folder, f"CAL_{year_str}_timestamps.csv")

            if os.path.exists(timestamps_path):
                self.timestamps[year_str] = pd.read_csv(timestamps_path)
            else:
                raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

            if self.use_possible_starts:
                possible_starts_path = os.path.join(year_folder, f"CAL_{year_str}_possible_starts.npy")
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

    def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        if self.std == 0:
            return torch.zeros_like(data)
        return (data - self.mean) / (self.std + eps)

    def denormalize_data(self, normalized_data: torch.Tensor) -> torch.Tensor:
        return normalized_data * self.std + self.mean



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
        #    spatial_shape = (ds.SIS.shape[-2], ds.SIS.shape[-1])
        lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm, indexing='xy')

        #    lon_grid = lon_grid[np.newaxis, np.newaxis, :, :]
        #    lat_grid = lat_grid[np.newaxis, np.newaxis, :, :]

        sin_lon = np.sin(lon_grid)[np.newaxis, np.newaxis, :, :]
        cos_lon = np.cos(lon_grid)[np.newaxis, np.newaxis, :, :]
        sin_lat = np.sin(lat_grid)[np.newaxis, np.newaxis, :, :]
        cos_lat = np.cos(lat_grid)[np.newaxis, np.newaxis, :, :]
        coords = np.concatenate([sin_lon, cos_lon, sin_lat, cos_lat], axis=0)
        coords = np.repeat(coords, self.input_len, axis=1)
        return torch.tensor(coords, dtype=torch.float32)

    def _prepare_time_coords(self, year: str, start_idx: int, idx_end: int) -> torch.Tensor:
        #H, W = ds.CAL.shape[-2:]
        time_slice = self.timestamps[year].iloc[start_idx:start_idx+self.input_len]['StartTimeUTC'].values
        time_pd = pd.to_datetime(time_slice)
        months = time_pd.month.values / 12.0
        days = time_pd.day.values / 31.0
        hours = time_pd.hour.values / 24.0
        minutes = time_pd.minute.values / 60.0
        time_features = torch.tensor(np.stack([months, days, hours, minutes]), dtype=torch.float32)
        #time_coords = time_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return time_features

    def __len__(self) -> int:
        return self.nitems

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        '''
        return:
        SIS_data.shape [B, 1, input_len, H, W]
        cached_coords.shape [B,2, input_len, H, W]
        target.shape [B,1, predict_len, H, W]
        time_coords.shape [B,4, input_len, H, W]
        '''

        if self.validation:
            np.random.seed(self.seeds[idx])
        year_str, start_idx = self.year_mapping[idx]
        idx_end = (start_idx + self.input_len + self.pred_len)
        with xr.open_dataset(self.data_paths[year_str]) as ds:
            K = ds.CAL.values
            time = ds.time
            K_clip = K[start_idx:idx_end]
            times = time[start_idx:idx_end]
            data = torch.tensor(K_clip, dtype=torch.float32).unsqueeze(0)
            cached_coords = self._prepare_coords(ds)
            H, W = ds.CAL.shape[-2:]
            time_features = self._prepare_time_coords(year_str, start_idx, idx_end)
            time_coords = time_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        x = data[:, :self.input_len, :, :]
        y = data[:, self.input_len:, :, :]
        time_y = times[self.input_len:].values.astype('datetime64[s]').astype('int64')
        return {
            "his_cal": x,
            "spatial_coordinates": cached_coords,
            "time_coordinates": time_coords,
            "target": y,
            "time":time_y
            }


