import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from torchtyping import TensorType
from typing import Tuple
from collections import OrderedDict


class CLADEResUpscale(nn.Module):
    """
    This combines the upscaling and the SEANResBlock.

    Args:
    -----
        upscale_size: float or tuple (w,h)
            Parameter for upscaling. Can be either a scaling factor or an output size
        input_dim: int,
            Dimension of the input feature map
        output_dim: int,
            Dimension of the output feature map
        num_labels: int,
            Number of segmentation labels
        kernel_size: int,
            Size of the kernel for all convolutions in the kernel (padding = kernel_size//2)
        apply_spectral_norm: bool, default True
            Whether to use or not the spectral norm in the block
        use_styles: bool, default True
            Whether to modulate using a mix of styles and segmentation masks
            or only using segmentation masks.
    """

    def __init__(
        self,
        upscale_size: int,
        input_dim: int,
        output_dim: int,
        num_labels: int,
        kernel_size: int,
        apply_spectral_norm: bool = True,
    ):
        super().__init__()
        if upscale_size == 0:
            self.up = nn.Identity()
        elif type(upscale_size) == int:
            self.up = nn.Upsample(scale_factor=upscale_size)
        elif type(upscale_size) == tuple and len(upscale_size) == 2:
            self.up = nn.Upsample(size=upscale_size)
        else:
            raise "Upscaling size not supported"

        self.clade_block = CLADEResBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            num_labels=num_labels,
            kernel_size=kernel_size,
            apply_spectral_norm=apply_spectral_norm,
        )

    def forward(
        self,
        x: TensorType[
            "batch_size",
            "input_dim",
            "input_heigth",
            "input_width",
        ],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
    ) -> TensorType["batch_size", "output_dim", "output_heigth", "output_width",]:
        x = self.clade_block(x, segmentation_map)
        x = self.up(x)
        return x


class CLADEResBlock(nn.Module):
    """
    This is the base block for the SEAN system. We have to modulation-convolution
    and a modulation-convolution skip connection.

    Args:
    -----
        input_dim: int,
            Dimension of the input feature map
        output_dim: int,
            Dimension of the output feature map
        num_labels: int,
            Number of segmentation labels
        style_dim: int,
            Dimension of the style vectors
        kernel_size: int,
            Size of the kernel for all convolutions in the kernel (padding = kernel_size//2)
        apply_spectral_norm: bool, default True
            Whether to use or not the spectral norm in the block
        use_styles: bool, default True
            Whether to modulate using a mix of styles and segmentation masks
            or only using segmentation masks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_labels: int,
        kernel_size: int,
        apply_spectral_norm: bool = True,
    ):

        super().__init__()

        self.resnet_connection = input_dim != output_dim
        middle_dim = min(input_dim, output_dim)

        padding = kernel_size // 2

        self.activation = nn.LeakyReLU(2e-1)
        # Defining convolutions
        conv_0 = nn.Conv2d(
            input_dim, middle_dim, kernel_size=kernel_size, padding=padding
        )

        conv_1 = nn.Conv2d(
            middle_dim, output_dim, kernel_size=kernel_size, padding=padding
        )

        if self.resnet_connection:
            self.conv_res = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        if apply_spectral_norm:
            conv_0 = spectral_norm(conv_0)
            conv_1 = spectral_norm(conv_1)
            if self.resnet_connection:
                self.conv_res = spectral_norm(self.conv_res)
        self.conv_0 = nn.Sequential(nn.LeakyReLU(2e-1), conv_0)
        self.conv_1 = nn.Sequential(nn.LeakyReLU(2e-1), conv_1)
        # Defining modulation
        mod_params = {
            "num_labels": num_labels,
            "kernel_size": kernel_size,
        }
        self.mod_0 = CLADEModulation(feature_map_dim=input_dim, **mod_params)
        self.mod_1 = CLADEModulation(feature_map_dim=middle_dim, **mod_params)
        if self.resnet_connection:
            self.mod_res = CLADEModulation(feature_map_dim=input_dim, **mod_params)

    def forward(
        self,
        x: TensorType["batch_size", "input_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
    ) -> TensorType["batch_size", "output_dim", "fmap_height", "fmap_width"]:
        if self.resnet_connection:
            x_res = self.mod_res(x, segmentation_map)
            x_res = self.conv_res(x_res)
        else:
            x_res = x
        dx = F.leaky_relu(self.mod_0(x, segmentation_map), 0.2)
        dx = self.conv_0(dx)

        dx = F.leaky_relu(self.mod_1(dx, segmentation_map), 0.2)
        dx = self.conv_1(dx)

        return x_res + dx


class CLADEModulation(nn.Module):
    """
    This module implements the method proposed in the SEAN paper to modulate the input according
    to the different styles and the segmentation mask.

    Parameters:
    -----------

        num_labels: int,
            Number of segmentation labels
        feature_map_dim: int,
            Dimension of the layer we are going to modulate
        kernel_size: int,
            Size of the kernel for all convolutions in the kernel (padding = kernel_size//2)
        style_dim: int,
            Dimension of the style vectors
        use_styles: bool, default True
            Whether to modulate using a mix of styles and segmentation masks
            or only using segmentation masks.
    """

    def __init__(
        self,
        num_labels: int,
        feature_map_dim: int,
        kernel_size: int,
    ):
        super().__init__()

        self.segmap_encoder = ClassAffine(num_labels, feature_map_dim)

        self.normalization_layer = nn.BatchNorm2d(feature_map_dim, affine=False)

    def forward(
        self,
        x: TensorType["batch_size", "feature_map_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
    ) -> TensorType["batch_size", "feature_map_dim", "fmap_height", "fmap_width",]:
        # We first normalize the input

        x = self.normalization_layer(x)

        # We scale the segmentation map to match input dimension (HxW)
        segmentation_map = F.interpolate(
            segmentation_map, size=x.size()[2:], mode="nearest"
        )

        gamma_segmap, beta_segmap = self.segmap_encoder(segmentation_map)
        out = gamma_segmap * x + beta_segmap
        return out


class ClassAffine(nn.Module):
    """
    This block encode the segmentation map and output 2 parameters gamma and beta
    that contributes to the modulation of the generator last layer.

    Parameters:
    -----------
        num_labels: int,
            Number of labels in the segmentation map
        out_channels: int,
            Number of output channels. Must correspond to generator last layer dimension
        kernel_size: int,
            Kernel size of the convolutions that are used in this layer
    """

    def __init__(self, num_labels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.Tensor(self.num_labels, self.out_channels))
        self.bias = nn.Parameter(torch.Tensor(self.num_labels, self.out_channels))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        segmentation_map: TensorType[
            "batch_size", "num_labels", "fmap_height", "fmap_width"
        ],
    ) -> Tuple[
        TensorType["batch_size", "out_channels", "fmap_height", "fmap_width"],
        TensorType["batch_size", "out_channels", "fmap_height", "fmap_width"],
    ]:
        arg_mask = torch.argmax(segmentation_map, 1).long()  # [n, h, w]
        class_weight = F.embedding(arg_mask, self.weight).permute(
            0, 3, 1, 2
        )  # [n, c, h, w]
        class_bias = F.embedding(arg_mask, self.bias).permute(
            0, 3, 1, 2
        )  # [n, c, h, w]
        return class_weight, class_bias
