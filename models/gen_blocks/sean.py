import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from torchtyping import TensorType
from typing import Tuple
from collections import OrderedDict


class SEANResUpscale(nn.Module):
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
        upscale_size: int,
        input_dim: int,
        output_dim: int,
        num_labels: int,
        style_dim: int,
        kernel_size: int,
        apply_spectral_norm: bool = True,
        use_styles: bool = True,
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

        self.sean_block = SEANResBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            num_labels=num_labels,
            style_dim=style_dim,
            kernel_size=kernel_size,
            apply_spectral_norm=apply_spectral_norm,
            use_styles=use_styles,
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
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
    ) -> TensorType["batch_size", "output_dim", "output_heigth", "output_width",]:
        x = self.sean_block(x, segmentation_map, style_codes=style_codes)
        x = self.up(x)
        return x


class SEANResBlock(nn.Module):
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
        style_dim: int,
        kernel_size: int,
        apply_spectral_norm: bool = True,
        use_styles: bool = True,
    ):

        super().__init__()

        self.use_styles = True

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
            "style_dim": style_dim,
            "use_styles": use_styles,
        }
        self.mod_0 = SEANModulation(feature_map_dim=input_dim, **mod_params)
        self.mod_1 = SEANModulation(feature_map_dim=middle_dim, **mod_params)
        if self.resnet_connection:
            self.mod_res = SEANModulation(feature_map_dim=input_dim, **mod_params)

    def forward(
        self,
        x: TensorType["batch_size", "input_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
    ) -> TensorType["batch_size", "output_dim", "fmap_height", "fmap_width"]:
        if self.resnet_connection:
            x_res = self.mod_res(x, segmentation_map, style_codes=style_codes)
            x_res = self.conv_res(x_res)
        else:
            x_res = x
        dx = self.mod_0(x, segmentation_map, style_codes=style_codes)
        dx = self.conv_0(dx)

        dx = self.mod_1(dx, segmentation_map, style_codes=style_codes)
        dx = self.conv_1(dx)

        return x_res + dx


class SEANModulation(nn.Module):
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
        style_dim: int,
        use_styles: bool = True,
    ):
        super().__init__()
        self.use_styles = use_styles
        self.num_labels = num_labels
        self.style_dim = style_dim

        self.segmap_encoder = SegMapEncoder(num_labels, feature_map_dim, kernel_size)

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(feature_map_dim), requires_grad=True)

        padding = kernel_size // 2

        self.normalization_layer = nn.BatchNorm2d(feature_map_dim, affine=False)

        if self.use_styles:
            self.st_proj = nn.ModuleDict(
                OrderedDict(
                    {
                        f"style_code_{i}_proj": nn.Sequential(
                            nn.Linear(style_dim, style_dim), nn.ReLU()
                        )
                        for i in range(num_labels)
                    }
                )
            )
            self.mlp_st_gamma = nn.Conv2d(
                in_channels=style_dim,
                out_channels=feature_map_dim,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.mlp_st_beta = nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=feature_map_dim,
                kernel_size=kernel_size,
                padding=padding,
            )

    def forward(
        self,
        x: TensorType["batch_size", "feature_map_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
    ) -> TensorType["batch_size", "feature_map_dim", "fmap_height", "fmap_width",]:
        # We first add some noise and normalize the input
        noise = (
            self.noise_var
            * torch.randn((x.shape[0], x.shape[3], x.shape[2], 1), device=x.device)
        ).transpose(1, 3)
        normalized_and_noised = self.normalization_layer(x + noise)

        # We scale the segmentation map to match input dimension (HxW)
        segmentation_map = F.interpolate(
            segmentation_map, size=x.size()[2:], mode="nearest"
        )

        if self.use_styles:
            # If we use the styles we proceed to broadcast them and encode them
            (batch_size, _, height, width) = normalized_and_noised.shape
            broadcasted_style_codes = torch.zeros(
                (batch_size, self.style_dim, height, width),
                device=normalized_and_noised.device,
            )
            for i in range(self.num_labels):
                projected_style_code = (
                    (self.st_proj[f"style_code_{i}_proj"](style_codes[:, i, :]))
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )

                broadcasted_style_codes += projected_style_code * segmentation_map[
                    :, i, :, :
                ].unsqueeze(1)

            # We then map to the modulation and modulate the map mixing style and segmap info
            gamma_st = self.mlp_st_gamma(broadcasted_style_codes)
            beta_st = self.mlp_st_beta(broadcasted_style_codes)

            gamma_segmap, beta_segmap = self.segmap_encoder(segmentation_map)

            gamma_alpha = torch.sigmoid(self.blending_gamma)
            beta_alpha = torch.sigmoid(self.blending_beta)

            gamma_final = gamma_alpha * gamma_st + (1 - gamma_alpha) * gamma_segmap
            beta_final = beta_alpha * beta_st + (1 - beta_alpha) * beta_segmap

            out = (1 + gamma_final) * normalized_and_noised + beta_final

        else:
            gamma_segmap, beta_segmap = self.segmap_encoder(segmentation_map)
            out = (1 + gamma_segmap) * normalized_and_noised + beta_segmap
        return out


class SegMapEncoder(nn.Module):
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

    def __init__(self, num_labels: int, out_channels: int, kernel_size: int):
        super().__init__()

        padding = kernel_size // 2

        hidden_dim = 128

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=num_labels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
        )

        self.gamma_mlp = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.mu_mlp = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        segmentation_map: TensorType[
            "batch_size", "num_labels", "fmap_height", "fmap_width"
        ],
    ) -> Tuple[
        TensorType["batch_size", "out_channels", "fmap_height", "fmap_width"],
        TensorType["batch_size", "out_channels", "fmap_height", "fmap_width"],
    ]:
        actv = self.shared_mlp(segmentation_map)

        gamma = self.gamma_mlp(actv)
        beta = self.mu_mlp(actv)

        return gamma, beta
