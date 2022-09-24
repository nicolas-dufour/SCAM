import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from typing import Union, Sequence

import torch.nn.utils.spectral_norm as spectral_norm
from collections import OrderedDict
from math import ceil

from models.utils_blocks.base import BaseNetwork

from models.utils_blocks.equallr import EqualConv2d
from functools import partial


class MultiScalePatchGanDiscriminator(BaseNetwork):
    """
    This is the Multi Scale Discriminator. We use multiple discriminators at different scales

    Parameters:
    -----------
    num_discriminator: int,
        How many discriminators do we use
    image_num_channels: int,
        Number of input images channels
    segmap_num_channels: int,
        Number of segmentation map channels
    num_features_fst_conv: int,
        How many kernels at the first convolution
    num_layers: int,
        How many layers per discriminator
    apply_spectral_norm: bool = True,
        Wheter or not to apply spectral normalization
    keep_intermediate_results: bool = True
        Whether or not to keep intermediate discriminators feature maps

    """

    def __init__(
        self,
        num_discriminator: int,
        image_num_channels: int,
        segmap_num_channels: int,
        num_features_fst_conv: int,
        num_layers: int,
        apply_spectral_norm: bool = True,
        apply_grad_norm: bool = False,
        keep_intermediate_results: bool = True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        self.keep_intermediate_results = keep_intermediate_results

        self.image_num_channels = image_num_channels

        self.discriminators = nn.ModuleDict(OrderedDict())

        for i in range(num_discriminator):
            self.discriminators.update(
                {
                    f"discriminator_{i}": PatchGANDiscriminator(
                        image_num_channels=image_num_channels,
                        segmap_num_channels=segmap_num_channels,
                        num_features_fst_conv=num_features_fst_conv,
                        num_layers=num_layers,
                        apply_spectral_norm=apply_spectral_norm,
                        apply_grad_norm=apply_grad_norm,
                        keep_intermediate_results=keep_intermediate_results,
                        use_equalized_lr=use_equalized_lr,
                        lr_mul=lr_mul,
                    ),
                }
            )
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(
        self,
        input: TensorType["batch_size", "input_channels", "height", "width"],
    ) -> Union[
        Sequence[
            Sequence[TensorType["batch_size", 1, "output_height", "output_width"]]
        ],
        Sequence[TensorType["batch_size", 1, "output_height", "output_width"]],
    ]:
        results = []
        for disc_name in self.discriminators:
            result = self.discriminators[disc_name](input)
            if not self.keep_intermediate_results:
                result = [result]
            results.append(result)
            image = self.downsample(input[:, : self.image_num_channels])
            segmentation_map = F.interpolate(
                input[:, self.image_num_channels :],
                size=image.size()[2:],
                mode="nearest",
            )
            input = torch.cat([image, segmentation_map], dim=1)
        return results


class PatchGANDiscriminator(BaseNetwork):
    """
    PatchGAN Discriminator. Perform patch-wise GAN discrimination to avoid ignoring local high dimensional features.

    Parameters:
    -----------
        image_num_channels: int,
            Number of channels in the real/generated image
        segmap_num_channels: int,
            Number of channels in the segmentation map
        num_features_fst_conv: int,
            Number of features in the first convolution layer
        num_layers: int,
            Number of convolution layers
        apply_spectral_norm: bool, default True
            Whether or not to apply spectral normalization
        keep_intermediate_results: bool, default True
            Whether or not to keep each feature map output
    """

    def __init__(
        self,
        image_num_channels: int,
        segmap_num_channels: int,
        num_features_fst_conv: int,
        num_layers: int,
        apply_spectral_norm: bool = True,
        apply_grad_norm: bool = False,
        keep_intermediate_results: bool = True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        kernel_size = 4
        padding = int(ceil((kernel_size - 1.0) / 2))
        nffc = num_features_fst_conv
        self.keep_intermediate_results = keep_intermediate_results
        self.apply_grad_norm = apply_grad_norm
        self.model = nn.ModuleDict(OrderedDict())
        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )
        self.model.update(
            {
                "conv_0": nn.Sequential(
                    ConvLayer(
                        in_channels=image_num_channels + segmap_num_channels,
                        out_channels=nffc,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    nn.LeakyReLU(0.2, False),
                )
            }
        )

        for n in range(1, num_layers):
            nffc_prev = nffc
            nffc = min(2 * nffc_prev, 512)
            stride = 1 if n == num_layers - 1 else 2
            conv = ConvLayer(
                in_channels=nffc_prev,
                out_channels=nffc,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            if apply_spectral_norm:
                conv = spectral_norm(conv)
            self.model.update(
                {
                    f"conv_{n}": nn.Sequential(
                        conv,
                        nn.InstanceNorm2d(nffc),
                        nn.LeakyReLU(0.2, False),
                    )
                }
            )
        self.model.update(
            {
                f"last_conv": nn.Sequential(
                    ConvLayer(
                        in_channels=nffc,
                        out_channels=1,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    )
                )
            }
        )

    def forward(
        self,
        input: TensorType["batch_size", "input_channels", "height", "width"],
    ) -> Union[
        Sequence[TensorType["batch_size", 1, "output_height", "output_width"]],
        TensorType["batch_size", 1, "output_height", "output_width"],
    ]:
        results = [input]
        for conv_name in self.model:
            results.append(self.model[conv_name](results[-1]))
        if self.keep_intermediate_results:
            return results[1:]
        else:
            return results[-1]
