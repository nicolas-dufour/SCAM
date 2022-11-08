import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from torchtyping import TensorType
from typing import Tuple
from collections import OrderedDict

from einops.einops import rearrange
from models.utils_blocks.base import BaseNetwork

class INADEGenerator(BaseNetwork):
    """ "
    This is the SEAN Generator that generates images provided a segmentation mask and style codes.

    Parameters:
    -----------
        num_filters_last_layer: int,
            Number of convolution filters at the last layer of the generator.
        num_up_layers: int,
            Number of SEAN upscaling layers
        height: int,
            Height of the generated image
        width: int,
            Width of the generated image
        num_labels: int,
            Number of segmentation labels
        style_dim: int,
            Dimension of the style vectors
        kernel_size: int,
            Size of the kernel for all convolutions in the kernel (padding = kernel_size//2)
        num_output_channels: int.
            Number of output channels (3 for RGB)
        apply_spectral_norm: bool, default True
            Whether to use or not the spectral norm in the block
    """

    def __init__(
        self,
        num_filters_last_layer: int,
        num_up_layers: int,
        height: int,
        width: int,
        num_labels: int,
        noise_dim: int,
        kernel_size: int,
        num_output_channels: int,
        apply_spectral_norm: bool = True,
        use_vae: bool = True,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        nf = num_filters_last_layer
        self.nf = nf
        if num_up_layers >= 4:
            self.num_up_layers = num_up_layers
        else:
            raise ValueError("Number of layers must be bigger than 4")
        aspect_ratio = width / height

        if aspect_ratio < 1:
            self.small_w = width // (2 ** (num_up_layers - 2))

            self.small_h = round(self.small_w / aspect_ratio)

            if self.small_w < 2:
                raise ValueError(
                    "You picked to many layers for the given image dimension."
                )
        else:
            self.small_h = height // (2 ** (num_up_layers - 2))

            self.small_w = round(self.small_h / aspect_ratio)

            if self.small_h < 2:
                raise "You picked to many layers for the given image dimension."

        inade_params = {
            "num_labels": num_labels,
            "kernel_size": kernel_size,
            "apply_spectral_norm": apply_spectral_norm,
            "noise_dim": noise_dim,
        }

        padding = kernel_size // 2

        self.fc = nn.Linear(256, 16 * nf * self.small_w * self.small_h)

        dict_of_inade_blocks = OrderedDict()

        # Down layers with constant number of filters
        dict_of_inade_blocks.update(
            {
                f"INADE_block_0": INADEResUpscale(
                    upscale_size=2,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    **inade_params,
                ),
            }
        )
        dict_of_inade_blocks.update(
            {
                f"INADE_block_1": INADEResUpscale(
                    upscale_size=0,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    **inade_params,
                ),
            }
        )
        for i in range(2, num_up_layers - 4):
            dict_of_inade_blocks.update(
                {
                    f"INADE_block_{i}": INADEResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        **inade_params,
                    ),
                }
            )
        # We progressively diminish the feature map
        for i in range(3):
            dict_of_inade_blocks.update(
                {
                    f"INADE_block_{num_up_layers - 4+i}": INADEResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        **inade_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_inade_blocks.update(
            {
                f"INADE_block_{num_up_layers - 1}": INADEResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    **inade_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_inade_blocks)

        # And the last convolution
        self.last_conv = nn.Sequential(
            nn.LeakyReLU(2e-1),
            nn.Conv2d(
                in_channels=nf,
                out_channels=num_output_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
        content_code: TensorType["batch_size", "content_dim"] = None,
    ) -> TensorType["batch_size", "num_output_channels", "height", "height"]:

        batch_size = segmentation_map.shape[0]

        noise_start = torch.randn(
            [batch_size, 256], dtype=torch.float32, device=segmentation_map.device
        )
        x = self.fc(noise_start)
        x = rearrange(x, "b (c h w) -> b c h w", h=self.small_h, w=self.small_w)

        noise = torch.randn(
            [batch_size, segmentation_map.shape[1], 2, self.noise_dim],
            device=x.device,
        )
        s_noise = torch.unsqueeze(
            noise[:, :, 0, :].mul(torch.exp(0.5 * style_codes[1])) + style_codes[0], 2
        )
        b_noise = torch.unsqueeze(
            noise[:, :, 1, :].mul(torch.exp(0.5 * style_codes[3])) + style_codes[2], 2
        )
        noise = torch.cat([s_noise, b_noise], 2)

        # Then proceed to apply the different sean_block
        for i in range(len(self.backbone)):
            x = self.backbone[f"INADE_block_{i}"](x, segmentation_map, noise)

        # Finally a last convolution
        x = self.last_conv(x)
        return x

    def get_last_layer(self):
        return self.last_conv[-2].weight
        
class INADEResUpscale(nn.Module):
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
        noise_dim: int,
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

        self.inade_block = INADEResBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            num_labels=num_labels,
            kernel_size=kernel_size,
            noise_dim=noise_dim,
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
        noise: TensorType["batch_size", "num_labels", 2, "noise_dim"],
    ) -> TensorType["batch_size", "output_dim", "output_heigth", "output_width",]:
        x = self.inade_block(x, segmentation_map, noise)
        x = self.up(x)
        return x


class INADEResBlock(nn.Module):
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
        noise_dim: int,
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
            "noise_dim": noise_dim,
        }
        self.mod_0 = INADEModulation(feature_map_dim=input_dim, **mod_params)
        self.mod_1 = INADEModulation(feature_map_dim=middle_dim, **mod_params)
        if self.resnet_connection:
            self.mod_res = INADEModulation(feature_map_dim=input_dim, **mod_params)

    def forward(
        self,
        x: TensorType["batch_size", "input_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        noise: TensorType["batch_size", "num_labels", 2, "noise_dim"],
    ) -> TensorType["batch_size", "output_dim", "fmap_height", "fmap_width"]:
        if self.resnet_connection:
            x_res = self.mod_res(x, segmentation_map, noise)
            x_res = self.conv_res(x_res)
        else:
            x_res = x
        dx = F.leaky_relu(self.mod_0(x, segmentation_map, noise), 0.2)
        dx = self.conv_0(dx)

        dx = F.leaky_relu(self.mod_1(dx, segmentation_map, noise), 0.2)
        dx = self.conv_1(dx)

        return x_res + dx


class INADEModulation(nn.Module):
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

    def __init__(self, num_labels: int, feature_map_dim: int, noise_dim: int):
        super().__init__()

        self.segmap_encoder = SegMapEncoder(num_labels, feature_map_dim, noise_dim)

        self.normalization_layer = nn.InstanceNorm2d(feature_map_dim, affine=False)

    def forward(
        self,
        x: TensorType["batch_size", "feature_map_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        noise: TensorType["batch_size", "num_labels", 2, "noise_dim"],
    ) -> TensorType["batch_size", "feature_map_dim", "fmap_height", "fmap_width",]:
        # We first normalize the input

        x = self.normalization_layer(x)

        # We scale the segmentation map to match input dimension (HxW)
        segmentation_map = F.interpolate(
            segmentation_map, size=x.size()[2:], mode="nearest"
        )

        gamma_segmap, beta_segmap = self.segmap_encoder(segmentation_map, noise)
        out = gamma_segmap * x + beta_segmap
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

    def __init__(self, num_labels: int, out_channels: int, noise_dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(num_labels, out_channels, 2))
        self.bias = nn.Parameter(torch.Tensor(num_labels, out_channels, 2))
        self.reset_parameters()
        self.fc_noise = nn.Linear(noise_dim, out_channels)

    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        segmentation_map: TensorType[
            "batch_size", "num_labels", "fmap_height", "fmap_width"
        ],
        noise: TensorType["batch_size", "num_labels", 2, "noise_dim"],
    ) -> Tuple[
        TensorType["batch_size", "out_channels", "fmap_height", "fmap_width"],
        TensorType["batch_size", "out_channels", "fmap_height", "fmap_width"],
    ]:
        input_instances = segmentation_map
        # Part 3. class affine with noise
        noise_fc = self.fc_noise(noise)
        # create weigthed instance noise for scale
        class_weight = torch.einsum(
            "ic,nihw->nchw", self.weight[..., 0], segmentation_map
        )
        class_bias = torch.einsum("ic,nihw->nchw", self.bias[..., 0], segmentation_map)
        # init_noise = torch.randn([x.size()[0], input_instances.size()[1], self.norm_nc], device=x.get_device())
        instance_noise = torch.einsum(
            "nic,nihw->nchw", noise_fc[:, :, 0, :], input_instances
        )
        scale_instance_noise = class_weight * instance_noise + class_bias
        # create weighted instance noise for bias
        class_weight = torch.einsum(
            "ic,nihw->nchw", self.weight[..., 1], segmentation_map
        )
        class_bias = torch.einsum("ic,nihw->nchw", self.bias[..., 1], segmentation_map)
        # init_noise = torch.randn([x.size()[0], input_instances.size()[1], self.norm_nc], device=x.get_device())
        instance_noise = torch.einsum(
            "nic,nihw->nchw", noise_fc[:, :, 1, :], input_instances
        )
        bias_instance_noise = class_weight * instance_noise + class_bias

        return scale_instance_noise, bias_instance_noise
