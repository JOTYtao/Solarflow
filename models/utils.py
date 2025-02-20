import einops
import math
import torch
import numpy as np
from models.layers import CoordConv

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda
def get_conv_layer(conv_type: str = "standard") -> torch.nn.Module:
    if conv_type == "standard":
        conv_layer = torch.nn.Conv2d
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "antialiased":
        # TODO Add anti-aliased coordconv here
        conv_layer = torch.nn.Conv2d
    elif conv_type == "3d":
        conv_layer = torch.nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


def reverse_space_to_depth(
    frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
    """Reverse space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b h w (dh dw c) -> b (h dh) (w dw) c",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    if len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    raise ValueError(
        "Frames should be of rank 4 (batch, height, width, channels)"
        " or rank 5 (batch, time, height, width, channels)"
    )


def space_to_depth(
    frames: np.ndarray, temporal_block_size: int = 1, spatial_block_size: int = 1
) -> np.ndarray:
    """Space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames,
            "b (h dh) (w dw) c -> b h w (dh dw c)",
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    if len(frames.shape) == 5:
        return einops.rearrange(
            frames,
            "b (t dt) (h dh) (w dw) c -> b t h w (dt dh dw c)",
            dt=temporal_block_size,
            dh=spatial_block_size,
            dw=spatial_block_size,
        )
    raise ValueError(
        "Frames should be of rank 4 (batch, height, width, channels)"
        " or rank 5 (batch, time, height, width, channels)"
    )



def reserve_schedule_sampling_exp(
    itr, in_shape, batch_size, pre_seq_length, aft_seq_length, total_length, r_sampling_step_1, r_sampling_step_2,
    r_exp_alpha,  patch_size, device
):
    T, img_channel, img_height, img_width = in_shape
    if itr < r_sampling_step_1:
        r_eta = 0.5
    elif itr < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - r_sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < r_sampling_step_1:
        eta = 0.5
    elif itr < r_sampling_step_2:
        eta = 0.5 - (0.5 / (r_sampling_step_2 - r_sampling_step_1)) * (itr - r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, pre_seq_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, aft_seq_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((img_height // patch_size,
                    img_width // patch_size,
                    patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // patch_size,
                      img_width // patch_size,
                      patch_size ** 2 * img_channel))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(total_length - 2):
            if j < pre_seq_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (pre_seq_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  total_length - 2,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size ** 2 * img_channel))
    return torch.FloatTensor(real_input_flag).to(device)


def schedule_sampling(
        in_shape,eta, itr, batch_size, aft_seq_length, sampling_stop_iter, sampling_changing_rate,patch_size, device, scheduled_sampling=True
):
    T, img_channel, img_height, img_width = in_shape
    zeros = np.zeros((batch_size,
                      aft_seq_length - 1,
                      img_height // patch_size,
                      img_width // patch_size,
                      patch_size ** 2 * img_channel))
    if not scheduled_sampling:
        return 0.0, zeros

    if itr < sampling_stop_iter:
        eta -= sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, aft_seq_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_height // patch_size,
                    img_width // patch_size,
                    patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // patch_size,
                      img_width // patch_size,
                      patch_size ** 2 * img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(aft_seq_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  aft_seq_length - 1,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size ** 2 * img_channel))
    return eta, torch.FloatTensor(real_input_flag).to(device)


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    a = img_tensor.reshape(batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels)
    b = a.transpose(3, 4)
    patch_tensor = b.reshape(batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels)
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape(batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels)
    b = a.transpose(3, 4)
    img_tensor = b.reshape(batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels)
    return img_tensor