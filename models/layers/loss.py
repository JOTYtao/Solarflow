"""Sereval loss functions and high level loss function get'er."""
import logging
import math
logger = logging.getLogger(__name__)
from typing import List, Optional, Union
from torch.autograd import Variable
from pytorch_msssim import MS_SSIM, SSIM
import torch
from torch import nn as nn
from torch.nn import functional as F


def get_outnorm(x: torch.Tensor, out_norm: str = "") -> torch.Tensor:
    """
    Common function to get a loss normalization value.

    Can normalize by either the
    - batch size ('b'),
    - the number of channels ('c'),
    - the image size ('i')
    - or combinations ('bi', 'bci', etc)

    Args:
        x: the tensor to be normalized
        out_norm: the string dimension to be normalized

    Returns: the normalized tensor

    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if "b" in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if "c" in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if "i" in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


def get_4dim_image_gradients(image: torch.Tensor):
    """
    Returns image gradients (dy, dx) for each color channel

    This uses the finite-difference approximation.
    Similar to get_image_gradients(), but additionally calculates the
    gradients in the two diagonal directions: 'dp' (the positive
    diagonal: bottom left to top right) and 'dn' (the negative
    diagonal: top left to bottom right).
    Only 1-step finite difference has been tested and is available.

    Args:
        image: Tensor with shape [b, c, h, w].

    Returns: tensors (dy, dx, dp, dn) holding the vertical, horizontal and
        diagonal image gradients (1-step finite difference). dx will
        always have zeros in the last column, dy will always have zeros
        in the last row, dp will always have zeros in the last row.

    """
    right = F.pad(image, (0, 1, 0, 0))[..., :, 1:]
    bottom = F.pad(image, (0, 0, 0, 1))[..., 1:, :]
    botright = F.pad(image, (0, 1, 0, 1))[..., 1:, 1:]

    dx, dy = right - image, bottom - image
    dn, dp = botright - image, right - bottom

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    dp[:, :, -1, :] = 0

    return dx, dy, dp, dn


def get_image_gradients(image: torch.Tensor, step: int = 1):
    """
    Returns image gradients (dy, dx) for each color channel,

    This use the finite-difference approximation.
    Places the gradient [ie. I(x+1,y) - I(x,y)] on the base pixel (x, y).
    Both output tensors have the same shape as the input: [b, c, h, w].

    Args:
        image: Tensor with shape [b, c, h, w].
        step: the size of the step for the finite difference

    Returns: Pair of tensors (dy, dx) holding the vertical and horizontal
        image gradients (ie. 1-step finite difference). To match the
        original size image, for example with step=1, dy will always
        have zeros in the last row, and dx will always have zeros in
        the last column.

    """
    right = F.pad(image, (0, step, 0, 0))[..., :, step:]
    bottom = F.pad(image, (0, 0, 0, step))[..., step:, :]

    dx, dy = right - image, bottom - image

    dx[:, :, :, -step:] = 0
    dy[:, :, -step:, :] = 0

    return dx, dy


class TVLoss(nn.Module):
    """Calculate the L1 or L2 total variation regularization.

    Also can calculate experimental 4D directional total variation.
    Ref:
        Mahendran et al. https://arxiv.org/pdf/1412.0035.pdf
    """

    def __init__(
        self,
        tv_type: str = "tv",
        p=2,
        reduction: str = "mean",
        out_norm: str = "b",
        beta: int = 2,
    ) -> None:
        """
        Init

        Args:
            tv_type: regular 'tv' or 4D 'dtv'
            p: use the absolute values '1' or Euclidean distance '2' to
                calculate the tv. (alt names: 'l1' and 'l2')
            reduction: aggregate results per image either by their 'mean' or
                by the total 'sum'. Note: typically, 'sum' should be
                normalized with out_norm: 'bci', while 'mean' needs only 'b'.
            out_norm: normalizes the TV loss by either the batch size ('b'), the
                number of channels ('c'), the image size ('i') or combinations
                ('bi', 'bci', etc).
            beta: β factor to control the balance between sharp edges (1<β<2)
                and washed out results (penalizing edges) with β >= 2.
        """
        super(TVLoss, self).__init__()
        if isinstance(p, str):
            p = 1 if "1" in p else 2
        if p not in [1, 2]:
            raise ValueError(f"Expected p value to be 1 or 2, but got {p}")

        self.p = p
        self.tv_type = tv_type.lower()
        self.reduction = torch.sum if reduction == "sum" else torch.mean
        self.out_norm = out_norm
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method

        Args:
            x: data

        Returns: model outputs

        """
        norm = get_outnorm(x, self.out_norm)
        img_shape = x.shape
        if len(img_shape) == 3:
            # reduce all axes. (None is an alias for all axes.)
            reduce_axes = None
            _ = 1
        elif len(img_shape) == 4:
            # reduce for the last 3 axes.
            # results in a 1-D tensor with the tv for each image.
            reduce_axes = (-3, -2, -1)
            _ = x.size()[0]
        else:
            raise ValueError(
                "Expected input tensor to be of ndim " f"3 or 4, but got {len(img_shape)}"
            )

        if self.tv_type in ("dtv", "4d"):
            # 'dtv': dx, dy, dp, dn
            gradients = get_4dim_image_gradients(x)
        else:
            # 'tv': dx, dy
            gradients = get_image_gradients(x)

        # calculate the TV loss for each image in the batch
        loss = 0
        for grad_dir in gradients:
            if self.p == 1:
                loss += self.reduction(grad_dir.abs(), dim=reduce_axes)
            elif self.p == 2:
                loss += self.reduction(torch.pow(grad_dir, 2), dim=reduce_axes)

        # calculate the scalar loss-value for tv loss
        # Note: currently producing same result if 'b' norm or not,
        # but for some cases the individual image loss could be used
        loss = loss.sum() if "b" in self.out_norm else loss.mean()
        if self.beta != 2:
            loss = torch.pow(loss, self.beta / 2)

        return loss * norm
class SSIMLoss(nn.Module):
    """SSIM Loss, optionally converting input range from [-1,1] to [0,1]"""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Init

        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to SSIM
        """
        super(SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method

        Args:
            x: one tensor
            y: second tensor

        Returns: SSIM loss

        """
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss, optionally converting input range from [-1,1] to [0,1]"""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Initialize

        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to MS_SSIM
        """
        super(MS_SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method

        Args:
            x: tensor one
            y: tensor two

        Returns:M S SSIM Loss

        """
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class SSIMLossDynamic(nn.Module):
    """
    SSIM Loss on only dynamic part of the images

    Optionally converting input range from [-1,1] to [0,1]

    In Mathieu et al. to stop SSIM regressing towards the mean and predicting
    only the background, they only run SSIM on the dynamic parts of the image.
    We can accomplish that by subtracting the current image from the future ones
    """

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Initialize

        Args:
            convert_range: Whether to convert from -1,1 to 0,1 as required for SSIM
            **kwargs: Kwargs for the ssim_module
        """
        super(SSIMLossDynamic, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, current_image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method

        Args:
            current_image: The last 'real' image given to the mode
            x: The target future sequence
            y: The predicted future sequence

        Returns:
            The SSIM loss computed only for the parts of the image that has changed
        """
        if self.convert_range:
            current_image = torch.div(torch.add(current_image, 1), 2)
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        # Subtract 'now' image to get what changes for both x and y
        x = x - current_image
        y = y - current_image
        # TODO: Mask out loss from pixels that don't change
        return 1.0 - self.ssim_module(x, y)

class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(
        self,
        gamma: Union[int, float, List] = 0,
        alpha: Optional[Union[int, float, List]] = None,
        size_average: bool = True,
    ):
        """
        Focal loss is described in https://arxiv.org/abs/1708.02002

        Copied from: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

        Courtesy of carwin, MIT License

        Args:
            alpha: (tensor, float, or list of floats) The scalar factor for this criterion
            gamma: (float,double) gamma > 0 reduces the relative loss for well-classified
                examples (p>0.5) putting more focus on hard misclassified example
            size_average: (bool, optional) By default, the losses are averaged over
                each loss element in the batch.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
        Forward model

        Args:
            x: prediction
            target: truth

        Returns: loss value

        """
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1, x.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(x)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.data.type_as(x)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class WeightedLosses:
    """Class: Weighted loss depending on the forecast horizon."""

    def __init__(self, decay_rate: Optional[int] = None, forecast_length: int = 6):
        """
        Want to set up the MSE loss function so the weights only have to be calculated once.

        Args:
            decay_rate: The weights exponentially decay depending on the 'decay_rate'.
            forecast_length: The forecast length is needed to make sure the weights sum to 1
        """
        self.decay_rate = decay_rate
        self.forecast_length = forecast_length

        logger.debug(
            f"Setting up weights with decay rate {decay_rate} and of length {forecast_length}"
        )

        # set default rate of ln(2) if not set
        if self.decay_rate is None:
            self.decay_rate = math.log(2)

        # make weights from decay rate
        weights = torch.FloatTensor(
            [math.exp(-self.decay_rate * i) for i in range(0, self.forecast_length)]
        )

        # normalized the weights, so there mean is 1.
        # To calculate the loss, we times the weights by the differences between truth
        # and predictions and then take the mean across all forecast horizons and the batch
        self.weights = weights / weights.sum() * len(weights)

        # move weights to gpu is needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = self.weights.to(device)

    def get_mse_exp(self, output, target):
        """Loss function weighted MSE"""

        # get the differences weighted by the forecast horizon weights
        diff_with_weights = self.weights * ((output - target) ** 2)

        # average across batches
        return torch.mean(diff_with_weights)

    def get_mae_exp(self, output, target):
        """Loss function weighted MAE"""

        # get the differences weighted by the forecast horizon weights
        diff_with_weights = self.weights * torch.abs(output - target)

        # average across batches
        return torch.mean(diff_with_weights)


class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss that penalizes blurry images more than MSE.
    """

    def __init__(self, alpha: int = 2):
        """
        Initalize the Loss Class.

        Args:
            alpha: #TODO
        """
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculate the Gradient Difference Loss.

        Args:
            x: vector one
            y: vector two

        Returns: the Gradient Difference Loss value

        """
        t1 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
                - torch.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])
            ),
            self.alpha,
        )
        t2 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])
                - torch.abs(y[:, :, :, :, :-1] - y[:, :, :, :, 1:])
            ),
            self.alpha,
        )
        # Pad out the last dim in each direction so shapes match
        t1 = F.pad(input=t1, pad=(0, 0, 1, 0), mode="constant", value=0)
        t2 = F.pad(input=t2, pad=(0, 1, 0, 0), mode="constant", value=0)
        loss = t1 + t2
        return loss.mean()


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer loss from Skillful Nowcasting,

    see https://arxiv.org/pdf/2104.00954.pdf.
    """

    def __init__(self, weight_fn=None):
        """
        Initialize the model.

        Args:
            weight_fn: the weight function the be called when #TODO?
        """
        super().__init__()
        self.weight_fn = weight_fn  # In Paper, weight_fn is max(y+1,24)

    def forward(self, generated_images, targets):
        """
        Calculates the grid cell regularizer value.

        This assumes generated images are the mean predictions from
        6 calls to the generater
        (Monte Carlo estimation of the expectations for the latent variable)

        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            difference *= self.weight_fn(targets)
        difference /= targets.size(1) * targets.size(3) * targets.size(4)  # 1/HWN
        return difference.mean()


class NowcastingLoss(nn.Module):
    """
    Loss described in Skillful-Nowcasting GAN,  see https://arxiv.org/pdf/2104.00954.pdf.
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

    def forward(self, x, real_flag):
        """
        Forward step.

        Args:
            x: the data to work with
            real_flag: boolean if its real or not

        Returns: #TODO

        """
        if real_flag is True:
            x = -x
        return F.relu(1.0 + x).mean()


def get_loss(loss: str = "mse", **kwargs) -> torch.nn.Module:
    """
    Function to get different losses easily.

    Args:
        loss: name of the loss, or torch.nn.Module, if a Module, returns that Module
        **kwargs: kwargs to pass to the loss function

    Returns:
        torch.nn.Module
    """
    if isinstance(loss, torch.nn.Module):
        return loss
    assert loss in [
        "mse",
        "bce",
        "binary_crossentropy",
        "crossentropy",
        "focal",
        "ssim",
        "ms_ssim",
        "l1",
        "tv",
        "total_variation",
        "ssim_dynamic",
        "gdl",
        "gradient_difference_loss",
        "weighted_mse",
        "weighted_mae",
    ]
    if loss == "mse":
        criterion = F.mse_loss
    elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
        criterion = F.nll_loss
    elif loss in ["focal"]:
        criterion = FocalLoss()
    elif loss in ["ssim"]:
        criterion = SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ms_ssim"]:
        criterion = MS_SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ssim_dynamic"]:
        criterion = SSIMLossDynamic(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["l1"]:
        criterion = torch.nn.L1Loss()
    elif loss in ["tv", "total_variation"]:
        criterion = TVLoss(
            tv_type=kwargs.get("tv_type", "tv"),
            p=kwargs.get("p", 1),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss in ["gdl", "gradient_difference_loss"]:
        criterion = GradientDifferenceLoss(alpha=kwargs.get("alpha", 2))
    elif loss in ["weighted_mse", "weighted_mae"]:
        criterion = WeightedLosses(**kwargs)
    else:
        raise ValueError(f"loss {loss} not recognized")
    return criterion