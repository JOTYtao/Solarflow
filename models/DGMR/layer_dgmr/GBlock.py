

import torch
from torch.nn.utils.parametrizations import spectral_norm
from models.DGMR.layer_dgmr.utils import get_conv_layer



class GBlock(torch.nn.Module):
    """Residual generator block without upsampling"""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        spectral_normalized_eps=0.0001,
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            eps=spectral_normalized_eps,
        )
        # Upsample 2D conv
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally spectrally normalized 1x1 convolution
        if x.shape[1] != self.output_channels:
            sc = self.conv_1x1(x)
        else:
            sc = x

        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        x = x2 + sc
        return x
class UpsampleGBlock(torch.nn.Module):
    """Residual generator block with upsampling"""

    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        spectral_normalized_eps=0.0001,
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        # Upsample 2D conv
        self.first_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            ),
            eps=spectral_normalized_eps,
        )
        self.last_conv_3x3 = spectral_norm(
            conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            ),
            eps=spectral_normalized_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectrally normalized 1x1 convolution
        sc = self.upsample(x)
        sc = self.conv_1x1(sc)

        x2 = self.bn1(x)
        x2 = self.relu(x2)
        # Upsample
        x2 = self.upsample(x2)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        x = x2 + sc
        return x
