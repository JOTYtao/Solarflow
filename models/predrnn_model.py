import torch
import torch.nn as nn

from modules import SpatioTemporalLSTMCell

class PredRNN_Model(nn.Module):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.
    """
    def __init__(
            self,
            num_layers: int,
            num_hidden: list,
            in_shape: list,
            patch_size: int,
            filter_size: int,
            stride: int,
            layer_norm: bool,
            input_len: int,
            pred_len: int,
            reverse_scheduled_sampling: int,
            **kwargs):
        super(PredRNN_Model, self).__init__()
        T, C, H, W = in_shape
        self.frame_channel = patch_size * patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = patch_size
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.input_len = input_len
        self.pred_len = pred_len
        self.reverse_scheduled_sampling = reverse_scheduled_sampling


        cell_list = []

        height = H // patch_size
        width = W // patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width,
                                       filter_size, stride, layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, channel, length, height, width] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width]).to(device)

        for t in range(self.input_len + self.pred_len - 1):
            # reverse schedule sampling
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_len:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.input_len] * frames[:, t] + \
                          (1 - mask_true[:, t - self.input_len]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, channel, length, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss
