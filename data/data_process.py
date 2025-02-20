import os
import numpy as np
import pandas as pd
import xarray as xr
import torch

def get_data(config, variable):
    data = xr.open_mfdataset(config['root_dir'])[variable]
    return data



def interpolate_to_latslons(data, config):

    if config['interpolation_factor'] > 1:
        lats = data.lat[::config['interpolation_factor']]
        lons = data.lon[::config['interpolation_factor']]
        data = data.interp(lat=lats, lon=lons, method='linear')
    return data


def interpolate_neighboring_nans(data):
    """
    Interpolate neighboring missing values.
    """
    data = data.interpolate_na(dim='time', limit=1)
    data = data.interpolate_na(dim='lat', limit=1)
    data = data.interpolate_na(dim='lon', limit=1)
    return data


def remove_nans(data, config):
    """
    Remove NaNs according to percentage NaNs allowed for a particular timestamp.
    This mainly removes observations during the night when there is no solar irradiance.
    """
    summed_hours = data.groupby('time').count(dim=xr.ALL_DIMS)
    image_size = data.shape[1] * data.shape[2]
    n_values_needed = image_size - int(config['nans_allowed_percentage'] * image_size)
    return data.where(summed_hours > n_values_needed, drop=True)


def get_possible_starts(data, config):
    """
    Ensure that our past and future observations can happen in sequence.
    """
    difference_range = np.diff(data.time)
    frames_total = config['n_past_steps'] + config['n_future_steps']

    counted = np.zeros(difference_range.shape)
    for idx, time in enumerate(difference_range):
        if idx != counted.shape[0] - 1:
            if time == np.timedelta64(1800000000000, 'ns'):  # 30 minutes in nanoseconds
                counted[idx + 1] = 1

    cum_sum = counted.copy()
    for idx, time in enumerate(counted):
        if idx > 0:
            if counted[idx] > 0:
                cum_sum[idx] = cum_sum[idx - 1] + cum_sum[idx]

    possible_indices = np.array(np.where(cum_sum >= (frames_total - 1))).ravel()
    possible_starts = possible_indices - (frames_total - 1)
    possible_starts = possible_starts.astype('int')
    possible_starts.sort()
    return possible_starts

def sanitize_data(data):
    """
    Sanitize xarray.DataArray by handling NaN, Inf, and out-of-range values.
    """
    print(f"Sanitizing data. Shape: {data.shape}, Type: {type(data)}")

    # 替换 NaN 和 Inf 值为 0
    data = data.fillna(0)  # 替换 NaN 为 0
    data = data.where(np.isfinite(data), 0)  # 将 Inf 和 -Inf 替换为 0

    data = data.clip(min=0, max=1.2)

    max_value = float(data.max())
    min_value = float(data.min())
    print(f"After sanitization:")
    print(f"- Max value: {max_value}")
    print(f"- Min value: {min_value}")

    return data
def transform(video, nan_to_num=True):
    """
    Transform xarray to torch.tensor.
    """
    video_clip = np.array(video)
    if nan_to_num:
        video_clip = np.nan_to_num(video_clip)
    video_clip = torch.stack([torch.Tensor(i) for i in video_clip])
    return video_clip


def convert_from_CAL_to_k(data):
    """
    Transforms from CAL to k using k = (1 - CAL).
    """
    return 1 - data


def save(data, possible_starts, config):
    """
    Save cloud albedo as video in torch.tensor, possible starts in numpy array, and timestamps in pandas dataframe.
    """


    data = sanitize_data(data)
    if data.isnull().any():
        print("Warning: NaN values still exist in the processed data!")
    else:
        print("No NaN values found in the processed data.")

    timestamps = data.time.values

    print('Saving torch data...')
    output_nc_path = config['out_dir'] + config['nc_out_filename']
    print('Saving processed data as NetCDF...')
    data.to_netcdf(output_nc_path)

    print('Saving possible starts...')
    with open(config['out_dir'] + config['nc_out_filename'].split('.')[0] + '_possible_starts.npy', 'wb') as f:
        np.save(f, possible_starts)

    print('Saving timestamps...')
    timestamps = pd.DataFrame(timestamps, columns=['StartTimeUTC'])
    timestamps.to_csv(config['out_dir'] + config['nc_out_filename'].split('.')[0] + '_timestamps.csv', index=False)


def process_data(config):
    """
    Process raw SARAH 2.1 data.
    """
    # Pipeline
    cloud_albedo = get_data(config, 'CAL')
    cloud_albedo = cloud_albedo.load()
    cloud_albedo = interpolate_to_latslons(cloud_albedo, config)
    cloud_albedo = interpolate_neighboring_nans(cloud_albedo)
    cloud_albedo = remove_nans(cloud_albedo, config)
    cloud_albedo = convert_from_CAL_to_k(cloud_albedo)

    possible_starts = get_possible_starts(cloud_albedo, config)

    print('Saving CAL data...')
    save(cloud_albedo, possible_starts, config)


if __name__ == '__main__':
    config = {
        'root_dir': r"J:\EUMETSAT\process_data\CAL\128\2021\CAL_2021.nc",
        'out_dir': r'E:\research\my_code\solar_flow\data\val\2021\\',
        'nc_out_filename': 'CAL_2021.nc',
        'nans_allowed_percentage': 0.05,
        'n_past_steps': 8,
        'n_future_steps': 8,
        'interpolation_factor': 1,
        'process_SIS': False
    }

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])

    print('Starting processing of SARAH 2.1 dataset...')
    try:
        process_data(config)
        print('Finished processing of SARAH 2.1 dataset...')
    except Exception as e:
        print('Could not process SARAH dataset due to {}, exiting script...'.format(e))