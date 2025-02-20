

import torch
from models.DGMR.layer_dgmr.utils import get_conv_layer






class LBlock(torch.nn.Module):
    """Residual block for the Latent Stack."""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        kernel_size: int = 3,
        conv_type: str = "standard",
    ):
        """
        L-Block for increasing the number of channels in the input
         from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Which type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        # Output size should be channel_out - channel_in
        self.input_channels = input_channels
        self.output_channels = output_channels
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels - input_channels,
            kernel_size=1,
        )

        self.first_conv_3x3 = conv2d(
            input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        self.relu = torch.nn.ReLU()
        self.last_conv_3x3 = conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )

    def forward(self, x) -> torch.Tensor:
        if self.input_channels < self.output_channels:
            sc = self.conv_1x1(x)
            sc = torch.cat([x, sc], dim=1)
        else:
            sc = x

        x2 = self.relu(x)
        x2 = self.first_conv_3x3(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        return x2 + sc
