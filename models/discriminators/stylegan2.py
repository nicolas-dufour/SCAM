from genericpath import exists
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchtyping import TensorType
from typing import Union, Sequence
from collections import OrderedDict
import torch.nn.utils.spectral_norm as spectral_norm

from models.utils_blocks.equallr import EqualConv2d, EqualLinear

from math import sqrt

from einops.layers.torch import Rearrange

from kornia.filters import filter2d

from models.utils_blocks.base import BaseNetwork


class MultiStyleGan2Discriminator(BaseNetwork):
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
        fmap_max: int,
        apply_spectral_norm: bool = True,
        apply_grad_norm: bool = False,
        keep_intermediate_results: bool = True,
        use_equalized_lr=False,
        lr_mul=1,
    ):
        super().__init__()

        self.keep_intermediate_results = keep_intermediate_results

        self.image_num_channels = image_num_channels

        self.discriminators = nn.ModuleDict(OrderedDict())

        for i in range(num_discriminator):
            self.discriminators.update(
                {
                    f"discriminator_{i}": StyleGan2Discriminator(
                        image_num_channels=image_num_channels,
                        segmap_num_channels=segmap_num_channels,
                        num_features_fst_conv=num_features_fst_conv,
                        num_layers=num_layers - i,
                        fmap_max=fmap_max,
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


class StyleGan2Discriminator(BaseNetwork):
    """
    Taken from lucidrain repo https://github.com/lucidrains/stylegan2-pytorch

    """

    def __init__(
        self,
        num_layers,
        image_num_channels,
        segmap_num_channels,
        num_features_fst_conv,
        fmap_max,
        keep_intermediate_results=True,
        apply_grad_norm=False,
        apply_spectral_norm=False,
        use_equalized_lr=False,
        lr_mul=1,
    ):
        super().__init__()
        self.apply_grad_norm = apply_grad_norm
        self.keep_intermediate_results = keep_intermediate_results
        init_channels = image_num_channels + segmap_num_channels
        blocks = []
        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )
        LinearLayer = (
            partial(EqualLinear, lr_mul=lr_mul) if use_equalized_lr else nn.Linear
        )
        filters = [init_channels] + [
            num_features_fst_conv * (2**i) for i in range(num_layers + 1)
        ]
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))
        for i, (input_dim, output_dim) in enumerate(chan_in_out):
            is_not_last = i != len(chan_in_out) - 1
            block = StyleGan2DiscBlock(
                input_dim,
                output_dim,
                downsample=is_not_last,
                apply_spectral_norm=apply_spectral_norm,
                use_equalized_lr=use_equalized_lr,
                lr_mul=lr_mul,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_block = nn.Sequential(
            ConvLayer(chan_last, chan_last, 3, padding=1),
            Rearrange("b c h w -> b (c h w)"),
            LinearLayer(latent_dim, 1),
        )

    def forward(self, input):
        if self.apply_grad_norm:
            input.requires_grad_(True)
        results = [input]
        for block in self.blocks:
            results.append(block(results[-1]))
        results.append(self.final_block(results[-1]))
        if self.keep_intermediate_results:
            return results[1:]
        else:
            return results[-1]


class StyleGan2DiscBlock(nn.Module):
    """
    Taken from lucidrain repo https://github.com/lucidrains/stylegan2-pytorch

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=True,
        apply_spectral_norm=False,
        use_equalized_lr=False,
        lr_mul=1,
    ):
        super().__init__()
        spectral_norm_op = spectral_norm if apply_spectral_norm else nn.Identity()
        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )
        self.conv_res = spectral_norm_op(
            ConvLayer(in_channels, out_channels, 1, stride=(2 if downsample else 1))
        )
        self.net = nn.Sequential(
            spectral_norm_op(ConvLayer(in_channels, out_channels, 3, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm_op(ConvLayer(out_channels, out_channels, 3, padding=1)),
            nn.LeakyReLU(0.2),
        )

        self.downsample = (
            nn.Sequential(
                # Blur(),
                spectral_norm_op(
                    ConvLayer(out_channels, out_channels, 3, padding=1, stride=2)
                ),
            )
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) / sqrt(2)
        return x


class Blur(nn.Module):
    """
    Taken from lucidrain repo https://github.com/lucidrains/stylegan2-pytorch

    """

    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)
