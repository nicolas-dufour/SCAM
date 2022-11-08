import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from torchtyping import TensorType
from typing import Tuple, Sequence
from einops import rearrange
from functools import partial
from models.utils_blocks.equallr import EqualLinear, EqualConv2d
import math
from collections import OrderedDict

from models.utils_blocks.attention import (
    MaskedAttention,
    MaskedTransformer,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding
)

from einops.einops import rearrange
from models.utils_blocks.base import BaseNetwork

class SCAMGenerator(BaseNetwork):
    """
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

class ToRGB(nn.Module):
    def __init__(
        self,
        input_dim: int,
        image_dim: int,
        num_labels: int,
        style_dim: int,
        kernel_size: int,
        attention_latent_dim: int,
        num_heads: int,
        attention_type: str = "duplex",
        apply_spectral_norm: bool = True,
        use_mask_adain: bool = False,
        norm_type: int = "InstanceNorm",
        modulate_fmap: bool = True,
        modulate: bool = True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        self.modulate_fmap = modulate_fmap
        if modulate_fmap:
            mod_params = {
                "num_labels": num_labels,
                "kernel_size": kernel_size,
                "style_dim": style_dim,
                "attention_latent_dim": attention_latent_dim,
                "num_heads": num_heads,
                "use_mask_adain": use_mask_adain,
                "norm_type": norm_type,
                "attention_type": attention_type,
                "modulate": modulate,
                "add_noise": False,
                "use_equalized_lr": use_equalized_lr,
                "lr_mul": lr_mul,
            }
            self.mod_rgb = SCAMModulation(feature_map_dim=input_dim, **mod_params)
        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )
        self.conv_rgb = ConvLayer(input_dim, image_dim, kernel_size=1)
        if apply_spectral_norm:
            self.conv_rgb = spectral_norm(self.conv_rgb)

    def forward(
        self,
        x: TensorType["batch_size", "input_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
        previous_rgb: TensorType[
            "batch_size", "image_dim", "fmap_height", "fmap_width"
        ] = None,
    ) -> TensorType["batch_size", "image_dim", "fmap_height", "fmap_width"]:
        if self.modulate_fmap:
            x, style_codes = self.mod_rgb(x, segmentation_map, style_codes=style_codes)
        x_rgb = self.conv_rgb(x)
        if previous_rgb is None:
            return x_rgb
        else:
            return x_rgb + previous_rgb


class SCAMResUpscale(nn.Module):
    """
    This is the base block for the SCAM system. We have two modulation-convolution
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
        num_heads: int,
            Number of heads to use in the attention blocks
        attention_type: str, default="duplex"
            Whether to use simplex or duplex attention
        apply_spectral_norm: bool, default True
            Whether to use or not the spectral norm in the block
        use_mask_adain: bool, default False,
            Wheter to include mask encoding information in the feature map modulation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_labels: int,
        style_dim: int,
        kernel_size: int,
        attention_latent_dim: int,
        num_heads: int,
        image_dim: int = 3,
        attention_type: str = "duplex",
        apply_spectral_norm: bool = True,
        use_mask_adain: bool = False,
        upscale_size: int = 0,
        norm_type: str = "InstanceNorm",
        architecture: str = "skip",
        last_block: bool = False,
        add_noise: bool = True,
        modulate: bool = True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
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
        self.architecture = architecture
        self.last_block = last_block
        middle_dim = min(input_dim, output_dim)

        padding = kernel_size // 2

        self.activation = nn.LeakyReLU(2e-1)

        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )

        # Defining modulation
        mod_params = {
            "num_labels": num_labels,
            "style_dim": style_dim,
            "kernel_size": kernel_size,
            "attention_latent_dim": attention_latent_dim,
            "num_heads": num_heads,
            "use_mask_adain": use_mask_adain,
            "norm_type": norm_type,
            "modulate": modulate,
            "use_equalized_lr": use_equalized_lr,
            "lr_mul": lr_mul,
        }
        self.mod_0 = SCAMModulation(
            feature_map_dim=input_dim,
            attention_type=attention_type,
            **mod_params,
            add_noise=add_noise,
        )
        self.mod_1 = SCAMModulation(
            feature_map_dim=middle_dim,
            attention_type=attention_type,
            **mod_params,
            add_noise=add_noise,
        )
        # Defining convolutions
        conv_0 = ConvLayer(
            input_dim, middle_dim, kernel_size=kernel_size, padding=padding
        )

        conv_1 = ConvLayer(
            middle_dim, output_dim, kernel_size=kernel_size, padding=padding
        )

        if apply_spectral_norm:
            conv_0 = spectral_norm(conv_0)
            conv_1 = spectral_norm(conv_1)

        self.conv_0 = nn.Sequential(nn.LeakyReLU(2e-1), conv_0)
        self.conv_1 = nn.Sequential(nn.LeakyReLU(2e-1), conv_1)

        ## Architecture details

        if self.architecture == "resnet":
            self.mod_res = SCAMModulation(
                feature_map_dim=input_dim,
                attention_type=attention_type,
                **mod_params,
                add_noise=add_noise,
            )
            self.conv_res = ConvLayer(input_dim, output_dim, kernel_size=1, bias=False)
            if apply_spectral_norm:
                self.conv_res = spectral_norm(self.conv_res)
        if self.architecture == "skip" or last_block:
            self.torgb = ToRGB(
                input_dim=output_dim,
                image_dim=image_dim,
                apply_spectral_norm=apply_spectral_norm,
                attention_type="simplex",
                **mod_params,
            )
        if last_block:
            self.torgb = ToRGB(
                input_dim=output_dim,
                image_dim=image_dim,
                apply_spectral_norm=apply_spectral_norm,
                attention_type="simplex",
                modulate_fmap=False,
                **mod_params,
            )
        if self.architecture not in ["skip", "resnet", "none"]:
            ValueError("Architecture type not supported")

    def forward(
        self,
        x: TensorType[
            "batch_size", "input_dim", "input_fmap_height", "input_fmap_width"
        ],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
        content_code: TensorType["batch_size", "content_dim"] = None,
        previous_rgb: TensorType[
            "batch_size", "image_dim", "input_fmap_height", "input_fmap_width"
        ] = None,
    ) -> Tuple[
        TensorType[
            "batch_size", "output_dim", "output_fmap_height", "output_fmap_width"
        ],
        TensorType["batch_size", "num_labels", "style_dim"],
        TensorType[
            "batch_size", "image_dim", "output_fmap_height", "output_fmap_width"
        ],
    ]:
        if self.architecture == "resnet":
            x_res = self.up(x)
            x_res, _ = self.mod_res(
                x_res,
                segmentation_map,
                style_codes=style_codes,
                content_code=content_code,
            )
            x_res = self.conv_res(x_res)
        dx, style_codes = self.mod_0(
            x, segmentation_map, style_codes=style_codes, content_code=content_code
        )
        dx = self.conv_0(dx)
        dx = self.up(dx)

        dx, style_codes = self.mod_1(
            dx, segmentation_map, style_codes=style_codes, content_code=content_code
        )
        dx = self.conv_1(dx)
        if self.architecture == "resnet":
            out_fmap = (x_res + dx) * (1 / math.sqrt(2))
        else:
            out_fmap = dx
        if self.architecture == "skip" or self.last_block:
            if previous_rgb is not None:
                previous_rgb = self.up(previous_rgb)
            rgb_image = self.torgb(
                out_fmap, segmentation_map, style_codes, previous_rgb
            )
        else:
            rgb_image = None

        return out_fmap, style_codes, rgb_image


class SCAMModulation(nn.Module):
    """
    Performs regional duplex attention modulation. We modulate the feature map attending to a set of latent variables.

    Args:
    -----
        num_labels: int,
            Number of segmentation labels
        feature_map_dim: int,
            Dimension of the feature map
        style_dim: int,
            Dimension of the style vectors
        kernel_size: int,
            Size of the kernel for all convolutions in the kernel (padding = kernel_size//2)
        num_heads: int,
            Number of heads to use in the attention blocks
        attention_type: str, default="duplex"
            Whether to use simplex or duplex attention
        use_mask_adain: bool, default False,
            Wheter to include mask encoding information in the feature map modulation
    """

    def __init__(
        self,
        num_labels: int,
        feature_map_dim: int,
        style_dim: int,
        attention_latent_dim: int,
        kernel_size: int,
        num_heads: int,
        attention_type: str = "duplex",
        use_mask_adain: bool = False,
        norm_type: bool = "InstanceNorm",
        add_noise: bool = True,
        modulate: bool = True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.style_dim = style_dim
        self.use_mask_adain = use_mask_adain
        self.add_noise = add_noise
        self.modulate = modulate
        self.attention_type = attention_type
        self.norm_type = norm_type
        if norm_type == "LayerNorm":
            self.normalization_layer = nn.Identity()
        elif norm_type == "InstanceNorm":
            self.normalization_layer = nn.InstanceNorm2d(feature_map_dim, affine=False)
        elif norm_type == "BatchNorm":
            self.normalization_layer = nn.BatchNorm2d(feature_map_dim, affine=False)
        else:
            raise ValueError("Normalization type not supported")

        LinearLayer = (
            partial(EqualLinear, lr_mul=lr_mul) if use_equalized_lr else nn.Linear
        )
        self.lr_mul = lr_mul

        if self.use_mask_adain:
            self.segmap_encoder = SegMapEncoder(
                num_labels,
                feature_map_dim,
                kernel_size,
                use_equalized_lr,
                lr_mul,
            )
            self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        if self.add_noise:
            self.noise_var = nn.Parameter(
                torch.zeros(feature_map_dim), requires_grad=True
            )

        ### SCAM Parameters

        self.feature_map_pos_encoding = SinusoidalPositionalEmbedding(feature_map_dim)

        self.latent_to_image = MaskedAttention(
            feature_map_dim,
            self.style_dim,
            attention_latent_dim,
            num_heads,
            use_equalized_lr,
            lr_mul,
        )

        self.return_attention = False
        self.mlp_lti_gamma = LinearLayer(feature_map_dim, feature_map_dim)
        self.mlp_lti_beta = LinearLayer(feature_map_dim, feature_map_dim)

        if attention_type == "duplex":
            self.image_to_latent = MaskedTransformer(
                self.style_dim,
                self.num_labels,
                feature_map_dim,
                attention_latent_dim,
                num_heads,
                use_equalized_lr,
                lr_mul,
            )
        else:
            self.fc_itl = nn.Sequential(
                LinearLayer(self.style_dim, self.style_dim),
                nn.LeakyReLU(2e-1),
                LinearLayer(self.style_dim, self.style_dim),
            )
        self.activation = nn.LeakyReLU(2e-1)

    def forward(
        self,
        x: TensorType["batch_size", "fmap_dim", "fmap_height", "fmap_width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
        style_codes: TensorType["batch_size", "num_labels", "style_dim"] = None,
        content_code: TensorType["batch_size", "content_dim"] = None,
    ) -> Tuple[
        TensorType["batch_size", "fmap_dim", "fmap_height", "fmap_width"],
        TensorType["batch_size", "num_labels", "style_dim"],
    ]:
        # We first add some noise and normalize the input
        batch_size, _, height, width = x.shape

        self.height = height
        self.width = width

        segmentation_map = F.interpolate(
            segmentation_map, size=x.size()[2:], mode="nearest"
        )
        attention_mask = segmentation_map.view(
            batch_size, self.num_labels, height * width
        )

        x_encoded = rearrange(self.feature_map_pos_encoding(x), " b c h w -> b (h w) c")

        if self.attention_type == "duplex":
            if self.return_attention:
                style_codes, _ = self.image_to_latent(
                    style_codes, x_encoded, attention_mask, return_attention=True
                )
            else:
                style_codes = self.image_to_latent(
                    style_codes,
                    x_encoded,
                    attention_mask,
                )
        else:
            style_codes = self.fc_itl(style_codes)

        if self.return_attention:

            x_out, _ = self.latent_to_image(
                x_encoded,
                style_codes,
                attention_mask.transpose(-1, -2),
                return_attention=True,
            )
        else:
            x_out = self.latent_to_image(
                x_encoded, style_codes, attention_mask.transpose(-1, -2)
            )
        gamma_lti = rearrange(
            self.mlp_lti_gamma(x_out), "b (h w) k -> b k h w", h=height, w=width
        )
        beta_lti = rearrange(
            self.mlp_lti_beta(x_out), "b (h w) k -> b k h w", h=height, w=width
        )
        if self.norm_type == "BatchNorm" or self.norm_type == "InstanceNorm":
            normalized = self.normalization_layer(x)
        elif self.norm_type == "LayerNorm":
            normalized = F.layer_norm(x, x.shape[1:])

        if self.modulate:
            if self.use_mask_adain:
                gamma_segmap, beta_segmap = self.segmap_encoder(segmentation_map)

                gamma_alpha = torch.sigmoid(self.blending_gamma * self.lr_mul)
                beta_alpha = torch.sigmoid(self.blending_beta * self.lr_mul)

                gamma_final = gamma_alpha * gamma_lti + (1 - gamma_alpha) * gamma_segmap
                beta_final = beta_alpha * beta_lti + (1 - beta_alpha) * beta_segmap

                out = (1 + gamma_final) * normalized + beta_final
            else:
                out = (1 + gamma_lti) * normalized + beta_lti
        else:
            out = normalized

        if self.add_noise:
            noise = (
                self.noise_var
                * self.lr_mul
                * torch.randn((batch_size, width, height, 1), device=x.device)
            ).transpose(1, 3)
            out = self.activation(out + noise)
        else:
            out = self.activation(out)

        return out, style_codes


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

    def __init__(
        self,
        num_labels: int,
        out_channels: int,
        kernel_size: int,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        padding = kernel_size // 2

        hidden_dim = 128
        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )

        self.shared_mlp = nn.Sequential(
            ConvLayer(
                in_channels=num_labels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
        )

        self.gamma_mlp = ConvLayer(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.mu_mlp = ConvLayer(
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
