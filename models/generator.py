from typing import List, Sequence
from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gen_blocks.spade import SPADEResUpscale
from models.gen_blocks.sean import SEANResUpscale
from models.gen_blocks.scam import SCAMResUpscale
from models.gen_blocks.clade import CLADEResUpscale
from models.gen_blocks.sean_clade import SEANCLADEResUpscale
from models.gen_blocks.groupdnet import GroupDNetResUpscale
from models.gen_blocks.inade import INADEResUpscale
from models.utils_blocks.attention import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from models.utils_blocks.base import BaseNetwork

from models.utils_blocks.equallr import EqualConv2d
from functools import partial

from torchtyping import TensorType
from collections import OrderedDict


class SCAMGenerator(BaseNetwork):
    """ "
    This is the SCAM Generator that generates images provided a segmentation mask and style codes using regional duplex attention

    Parameters:
    -----------
        num_filters_last_layer: int,
            Number of convolution filters at the last layer of the generator.
        num_up_layers: int,
            Number of SCAM upscaling layers
        num_up_layers_with_mask_adain: int,
            Number of SCAM blocks where we include explicitly mask info in the convolution
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
        num_heads: int,
            Number of heads to use in the attention blocks
        attention_type: str, default="duplex"
            Whether to use simplex or duplex attention
        num_output_channels: int.
            Number of output channels (3 for RGB)
        apply_spectral_norm: bool, default True
            Whether to use or not the spectral norm in the block
        split_latents: bool, default True
            Whether or not to split the latents into sublatents
        num_labels_split: bool, default 1
            Number of splits per latent
    """

    def __init__(
        self,
        num_filters_last_layer: int,
        num_up_layers: int,
        num_up_layers_with_mask_adain: int,
        height: int,
        width: int,
        num_labels: int,
        style_dim: int,
        kernel_size: int,
        attention_latent_dim: int,
        num_heads: int,
        attention_type: str,
        num_output_channels: int,
        latent_pos_emb: str = "learned",
        architecture: str = "skip",
        apply_spectral_norm: bool = True,
        split_latents: bool = True,
        num_labels_split: int = 1,
        num_labels_bg: int = None,
        norm_type: bool = "InstanceNorm",
        add_noise: bool = True,
        modulate: bool = True,
        use_equalized_lr: bool = False,
        use_vae: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        nf = num_filters_last_layer
        self.num_labels = num_labels
        self.num_labels_split = num_labels_split
        self.split_latents = split_latents
        self.attention_type = attention_type

        if num_labels_bg is None:
            num_labels_bg = num_labels_split

        self.num_labels_bg = num_labels_bg

        self.latent_pos_emb = latent_pos_emb
        if num_up_layers >= 4:
            self.num_up_layers = num_up_layers
        else:
            raise ValueError("Number of layers must be bigger than 4")
        assert num_up_layers_with_mask_adain <= num_up_layers
        aspect_ratio = width / height

        if aspect_ratio < 1:
            self.small_w = width // (2 ** (num_up_layers))

            self.small_h = round(self.small_w / aspect_ratio)

            if self.small_w < 2:
                raise ValueError(
                    "You picked to many layers for the given image dimension."
                )
        else:
            self.small_h = height // (2**num_up_layers)

            self.small_w = round(self.small_h / aspect_ratio)

            if self.small_h < 2:
                raise "You picked to many layers for the given image dimension."
        self.num_style_tokens = (num_labels - 1) * num_labels_split + num_labels_bg
        if split_latents:
            if not style_dim % num_labels_split == 0:
                raise ValueError(
                    "Style vector dimension must be divisible by the number of labels splits"
                )

            self.latents_dim = style_dim // num_labels_split
        else:
            self.latents_dim = style_dim

        if self.latent_pos_emb == "learned":
            self.style_code_pos_encoding = LearnedPositionalEmbedding(
                self.num_style_tokens, self.latents_dim
            )
        elif self.latent_pos_emb == "fourrier":
            self.style_code_pos_encoding = SinusoidalPositionalEmbedding(
                self.latents_dim
            )
        else:
            self.style_code_pos_encoding = nn.Identity()

        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )

        scam_params = {
            "num_labels": self.num_style_tokens,
            "style_dim": self.latents_dim,
            "kernel_size": kernel_size,
            "attention_latent_dim": attention_latent_dim,
            "num_heads": num_heads,
            "attention_type": attention_type,
            "apply_spectral_norm": apply_spectral_norm,
            "norm_type": norm_type,
            "architecture": architecture,
            "image_dim": num_output_channels,
            "add_noise": add_noise,
            "modulate": modulate,
            "use_equalized_lr": use_equalized_lr,
            "lr_mul": lr_mul,
        }

        padding = kernel_size // 2

        # First convolution
        self.first_conv = ConvLayer(
            in_channels=num_labels,
            out_channels=16 * nf,
            kernel_size=kernel_size,
            padding=padding,
        )

        dict_of_scam_blocks = OrderedDict()

        # Down layers with constant number of filters
        for i in range(num_up_layers - 4):
            dict_of_scam_blocks.update(
                {
                    f"SCAM_block_{i}": SCAMResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        use_mask_adain=i < num_up_layers_with_mask_adain,
                        **scam_params,
                    ),
                }
            )

        # We progressively diminish the feature map
        for i in range(3):
            dict_of_scam_blocks.update(
                {
                    f"SCAM_block_{num_up_layers - 4+i}": SCAMResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        use_mask_adain=num_up_layers - 4 + i
                        < num_up_layers_with_mask_adain,
                        **scam_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_scam_blocks.update(
            {
                f"SCAM_block_{num_up_layers - 1}": SCAMResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    last_block=True,
                    use_mask_adain=num_up_layers - 1 == num_up_layers_with_mask_adain,
                    **scam_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_scam_blocks)

        # And the last convolution
        self.last_act = nn.Tanh()

        self.copy = None
        self.use_vae = use_vae

    def forward(
        self,
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
        content_code: TensorType["batch_size", "content_dim"] = None,
    ) -> TensorType["batch_size", "num_output_channels", "height", "width"]:

        batch_size = segmentation_map.shape[0]

        x = F.interpolate(segmentation_map, size=(self.small_h, self.small_w))
        x = self.first_conv(x)

        # if self.copy is None:

        #     self.copy = torch.clone(self.first_conv.weight)
        # else:
        #     print(torch.sum(torch.abs(self.copy - self.first_conv.weight)))

        segmentation_map = torch.cat(
            [
                torch.repeat_interleave(
                    segmentation_map[:, 0].unsqueeze(1),
                    self.num_labels_bg,
                    dim=1,
                ),
                torch.repeat_interleave(
                    segmentation_map[:, 1:],
                    self.num_labels_split,
                    dim=1,
                ),
            ],
            dim=1,
        )
        if self.use_vae:
            assert isinstance(style_codes, Sequence)
            assert len(style_codes) == 2
            mu, logvar = style_codes
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                style_codes = eps.mul(std) + mu
            else:
                style_codes = mu
        if self.split_latents:
            style_codes = style_codes.view(
                batch_size, self.num_labels * self.num_labels_split, -1
            )
        if self.latent_pos_emb == "fourier":
            style_codes = self.style_code_pos_encoding(
                style_codes.unsqueeze(1)
            ).squeeze(1)
        else:
            style_codes = self.style_code_pos_encoding(style_codes)

        # Then proceed to apply the different sean_block
        img = None
        for i in range(len(self.backbone)):
            if self.attention_type == "duplex":
                x, style_codes, img = self.backbone[f"SCAM_block_{i}"](
                    x,
                    segmentation_map,
                    style_codes=style_codes,
                    content_code=content_code,
                    previous_rgb=img,
                )
            else:
                x, _, img = self.backbone[f"SCAM_block_{i}"](
                    x,
                    segmentation_map,
                    style_codes=style_codes,
                    content_code=content_code,
                    previous_rgb=img,
                )
        # Finally a last convolution
        img = self.last_act(img)
        return img
    def get_last_layer(self):
        return self.backbone[f"SCAM_block_{len(self.backbone)-1}"].torgb.conv_rgb.weight

class SEANGenerator(BaseNetwork):
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
        style_dim: int,
        kernel_size: int,
        num_output_channels: int,
        apply_spectral_norm: bool = True,
    ):
        super().__init__()

        nf = num_filters_last_layer
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

        sean_params = {
            "num_labels": num_labels,
            "style_dim": style_dim,
            "kernel_size": kernel_size,
            "apply_spectral_norm": apply_spectral_norm,
        }

        padding = kernel_size // 2

        # First convolution
        self.first_conv = nn.Conv2d(
            in_channels=num_labels,
            out_channels=16 * nf,
            kernel_size=kernel_size,
            padding=padding,
        )

        dict_of_sean_blocks = OrderedDict()

        dict_of_sean_blocks.update(
            {
                f"SEAN_block_0": SEANResUpscale(
                    upscale_size=2,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    use_styles=True,
                    **sean_params,
                ),
            }
        )
        dict_of_sean_blocks.update(
            {
                f"SEAN_block_1": SEANResUpscale(
                    upscale_size=0,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    use_styles=True,
                    **sean_params,
                ),
            }
        )
        # Down layers with constant number of filters
        for i in range(2, num_up_layers - 4):
            dict_of_sean_blocks.update(
                {
                    f"SEAN_block_{i}": SEANResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        use_styles=True,
                        **sean_params,
                    ),
                }
            )
        # We progressively diminish the feature map
        for i in range(3):
            dict_of_sean_blocks.update(
                {
                    f"SEAN_block_{num_up_layers - 4+i}": SEANResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        use_styles=True,
                        **sean_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_sean_blocks.update(
            {
                f"SEAN_block_{num_up_layers - 1}": SEANResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    use_styles=False,
                    **sean_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_sean_blocks)

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

        # We first downsample the segmentation map
        x = F.interpolate(segmentation_map, size=(self.small_h, self.small_w))

        # First convolution
        x = self.first_conv(x)

        # Then proceed to apply the different sean_block
        for i in range(len(self.backbone)):
            x = self.backbone[f"SEAN_block_{i}"](x, segmentation_map, style_codes)

        # Finally a last convolution
        x = self.last_conv(x)
        return x


class SPADEGenerator(BaseNetwork):
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
        kernel_size: int,
        num_output_channels: int,
        apply_spectral_norm: bool = True,
        use_vae: bool = True,
    ):
        super().__init__()

        self.use_vae = use_vae
        nf = num_filters_last_layer
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

        spade_params = {
            "num_labels": num_labels,
            "kernel_size": kernel_size,
            "apply_spectral_norm": apply_spectral_norm,
        }

        padding = kernel_size // 2

        # First convolution
        if self.use_vae:
            self.fc = nn.Linear(256, 16 * nf * self.small_w * self.small_h)

        else:
            self.first_conv = nn.Conv2d(
                in_channels=num_labels,
                out_channels=16 * nf,
                kernel_size=kernel_size,
                padding=padding,
            )

        dict_of_spade_blocks = OrderedDict()

        # Down layers with constant number of filters
        dict_of_spade_blocks.update(
            {
                f"SPADE_block_0": SPADEResUpscale(
                    upscale_size=2,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    **spade_params,
                ),
            }
        )
        dict_of_spade_blocks.update(
            {
                f"SPADE_block_1": SPADEResUpscale(
                    upscale_size=0,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    **spade_params,
                ),
            }
        )
        for i in range(2, num_up_layers - 4):
            dict_of_spade_blocks.update(
                {
                    f"SPADE_block_{i}": SPADEResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        **spade_params,
                    ),
                }
            )
        # We progressively diminish the feature map
        for i in range(3):
            dict_of_spade_blocks.update(
                {
                    f"SPADE_block_{num_up_layers - 4+i}": SPADEResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        **spade_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_spade_blocks.update(
            {
                f"SPADE_block_{num_up_layers - 1}": SPADEResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    **spade_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_spade_blocks)

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

        if self.use_vae:
            mu, logvar = style_codes
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = eps.mul(std) + mu
            x = self.fc(x)
            x = rearrange(x, "b (c h w) -> b c h w", h=self.small_h, w=self.small_w)

        else:
            # We first downsample the segmentation map
            x = F.interpolate(segmentation_map, size=(self.small_h, self.small_w))

            # First convolution
            x = self.first_conv(x)

        # Then proceed to apply the different sean_block
        for i in range(len(self.backbone)):
            x = self.backbone[f"SPADE_block_{i}"](x, segmentation_map)

        # Finally a last convolution
        x = self.last_conv(x)
        return x


class CLADEGenerator(BaseNetwork):
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
        kernel_size: int,
        num_output_channels: int,
        apply_spectral_norm: bool = True,
        use_vae: bool = True,
        use_dists: bool = False,
    ):
        super().__init__()

        self.use_vae = use_vae
        nf = num_filters_last_layer
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

        clade_params = {
            "num_labels": num_labels,
            "kernel_size": kernel_size,
            "apply_spectral_norm": apply_spectral_norm,
        }

        padding = kernel_size // 2

        # First convolution
        if self.use_vae:
            self.fc = nn.Linear(256, 16 * nf * self.small_w * self.small_h)

        else:
            self.first_conv = nn.Conv2d(
                in_channels=num_labels,
                out_channels=16 * nf,
                kernel_size=kernel_size,
                padding=padding,
            )

        dict_of_clade_blocks = OrderedDict()

        # Down layers with constant number of filters
        dict_of_clade_blocks.update(
            {
                f"CLADE_block_0": CLADEResUpscale(
                    upscale_size=2,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    **clade_params,
                ),
            }
        )
        dict_of_clade_blocks.update(
            {
                f"CLADE_block_1": CLADEResUpscale(
                    upscale_size=0,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    **clade_params,
                ),
            }
        )
        for i in range(2, num_up_layers - 4):
            dict_of_clade_blocks.update(
                {
                    f"CLADE_block_{i}": CLADEResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        **clade_params,
                    ),
                }
            )
        # We progressively diminish the feature map
        for i in range(3):
            dict_of_clade_blocks.update(
                {
                    f"CLADE_block_{num_up_layers - 4+i}": CLADEResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        **clade_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_clade_blocks.update(
            {
                f"CLADE_block_{num_up_layers - 1}": CLADEResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    **clade_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_clade_blocks)

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

        if self.use_vae:
            mu, logvar = style_codes
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = eps.mul(std) + mu
            x = self.fc(x)
            x = rearrange(x, "b (c h w) -> b c h w", h=self.small_h, w=self.small_w)

        else:
            # We first downsample the segmentation map
            x = F.interpolate(segmentation_map, size=(self.small_h, self.small_w))

            # First convolution
            x = self.first_conv(x)

        # Then proceed to apply the different sean_block
        for i in range(len(self.backbone)):
            x = self.backbone[f"CLADE_block_{i}"](x, segmentation_map)

        # Finally a last convolution
        x = self.last_conv(x)
        return x


class SEANCLADEGenerator(BaseNetwork):
    """ "
    This is the SEANCLADE Generator that generates images provided a segmentation mask and style codes.

    Parameters:
    -----------
        num_filters_last_layer: int,
            Number of convolution filters at the last layer of the generator.
        num_up_layers: int,
            Number of SEANCLADE upscaling layers
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
        style_dim: int,
        kernel_size: int,
        num_output_channels: int,
        apply_spectral_norm: bool = True,
        use_dists: bool = False,
    ):
        super().__init__()

        nf = num_filters_last_layer
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

        sean_clade_params = {
            "num_labels": num_labels,
            "style_dim": style_dim,
            "kernel_size": kernel_size,
            "apply_spectral_norm": apply_spectral_norm,
        }

        padding = kernel_size // 2

        # First convolution
        self.first_conv = nn.Conv2d(
            in_channels=num_labels,
            out_channels=16 * nf,
            kernel_size=kernel_size,
            padding=padding,
        )

        dict_of_sean_clade_blocks = OrderedDict()

        dict_of_sean_clade_blocks.update(
            {
                f"SEANCLADE_block_0": SEANCLADEResUpscale(
                    upscale_size=2,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    use_styles=True,
                    **sean_clade_params,
                ),
            }
        )
        dict_of_sean_clade_blocks.update(
            {
                f"SEANCLADE_block_1": SEANCLADEResUpscale(
                    upscale_size=0,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    use_styles=True,
                    **sean_clade_params,
                ),
            }
        )
        # Down layers with constant number of filters
        for i in range(2, num_up_layers - 4):
            dict_of_sean_clade_blocks.update(
                {
                    f"SEANCLADE_block_{i}": SEANCLADEResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        use_styles=True,
                        **sean_clade_params,
                    ),
                }
            )
        # We progressively diminish the feature map
        for i in range(3):
            dict_of_sean_clade_blocks.update(
                {
                    f"SEANCLADE_block_{num_up_layers - 4+i}": SEANCLADEResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        use_styles=True,
                        **sean_clade_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_sean_clade_blocks.update(
            {
                f"SEANCLADE_block_{num_up_layers - 1}": SEANCLADEResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    use_styles=False,
                    **sean_clade_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_sean_clade_blocks)

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

        # We first downsample the segmentation map
        x = F.interpolate(segmentation_map, size=(self.small_h, self.small_w))

        # First convolution
        x = self.first_conv(x)

        # Then proceed to apply the different sean_clade_block
        for i in range(len(self.backbone)):
            x = self.backbone[f"SEANCLADE_block_{i}"](x, segmentation_map, style_codes)

        # Finally a last convolution
        x = self.last_conv(x)
        return x


class GroupDNetGenerator(BaseNetwork):
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
        kernel_size: int,
        num_output_channels: int,
        apply_spectral_norm: bool = True,
        use_vae: bool = True,
    ):
        super().__init__()

        self.use_vae = use_vae
        nf = (num_filters_last_layer // num_labels) * num_labels
        if num_up_layers >= 4:
            self.num_up_layers = num_up_layers
        else:
            raise ValueError("Number of layers must be bigger than 4")
        aspect_ratio = width / height

        if aspect_ratio < 1:
            self.small_w = width // (2 ** (num_up_layers - 1))

            self.small_h = round(self.small_w / aspect_ratio)

            if self.small_w < 2:
                raise ValueError(
                    "You picked to many layers for the given image dimension."
                )
        else:
            self.small_h = height // (2 ** (num_up_layers - 1))

            self.small_w = round(self.small_h / aspect_ratio)

            if self.small_h < 2:
                raise "You picked to many layers for the given image dimension."

        groupdnet_params = {
            "num_labels": num_labels,
            "kernel_size": kernel_size,
            "apply_spectral_norm": apply_spectral_norm,
        }

        padding = kernel_size // 2

        # First convolution
        self.fc = nn.Conv2d(
            8 * num_labels,
            16 * num_labels,
            kernel_size=3,
            padding=1,
            groups=num_labels,
        )

        dict_of_groupdnet_blocks = OrderedDict()

        # Down layers with constant number of filters
        dict_of_groupdnet_blocks.update(
            {
                f"GroupDNet_block_0": GroupDNetResUpscale(
                    upscale_size=2,
                    input_dim=16 * num_labels,
                    output_dim=16 * nf,
                    groups=16,
                    **groupdnet_params,
                ),
            }
        )
        dict_of_groupdnet_blocks.update(
            {
                f"GroupDNet_block_1": GroupDNetResUpscale(
                    upscale_size=0,
                    input_dim=16 * nf,
                    output_dim=16 * nf,
                    groups=16,
                    **groupdnet_params,
                ),
            }
        )
        for i in range(2, num_up_layers - 4):
            dict_of_groupdnet_blocks.update(
                {
                    f"GroupDNet_block_{i}": GroupDNetResUpscale(
                        upscale_size=2,
                        input_dim=16 * nf,
                        output_dim=16 * nf,
                        groups=16,
                        **groupdnet_params,
                    ),
                }
            )
        # We progressively diminish the feature map
        for i in range(3):
            dict_of_groupdnet_blocks.update(
                {
                    f"GroupDNet_block_{num_up_layers - 4+i}": GroupDNetResUpscale(
                        upscale_size=2,
                        input_dim=2 ** (4 - i) * nf,
                        output_dim=2 ** (3 - i) * nf,
                        groups=2 ** (3 - i),
                        **groupdnet_params,
                    ),
                }
            )
        # Finally we upscale to the given size
        dict_of_groupdnet_blocks.update(
            {
                f"GroupDNet_block_{num_up_layers - 1}": GroupDNetResUpscale(
                    upscale_size=(height, width),
                    input_dim=2 * nf,
                    output_dim=nf,
                    groups=1,
                    **groupdnet_params,
                ),
            }
        )

        self.backbone = nn.ModuleDict(dict_of_groupdnet_blocks)

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

        if self.use_vae:
            mu, logvar = style_codes
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = eps.mul(std) + mu
            x = self.fc(x)

        else:
            # We first downsample the segmentation map
            x = F.interpolate(segmentation_map, size=(self.small_h, self.small_w))

            # First convolution
            x = self.fc(x)

        # Then proceed to apply the different sean_block
        for i in range(len(self.backbone)):
            x = self.backbone[f"GroupDNet_block_{i}"](x, segmentation_map)

        # Finally a last convolution
        x = self.last_conv(x)
        return x


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
