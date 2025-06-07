import math
import torch
import numpy as np


def reserve_schedule_sampling_exp(itr, batch_size, in_shape, r_sampling_step_1, r_sampling_step_2, r_exp_alpha, input_len, pre_len, patch_size, total_length, device):
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
        (batch_size, input_len - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, pre_len - 1))
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
            if j < input_len - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (input_len - 1)]:
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


def schedule_sampling(eta, itr,  batch_size, in_shape, scheduled_sampling, sampling_stop_iter, sampling_changing_rate, pre_len, patch_size, device):
    T, img_channel, img_height, img_width = in_shape
    zeros = np.zeros((batch_size,
                      pre_len - 1,
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
        (batch_size, pre_len - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_height // patch_size,
                    img_width // patch_size,
                    patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // patch_size,
                      img_width // patch_size,
                      patch_size ** 2 * img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(pre_len - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  pre_len - 1,
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
