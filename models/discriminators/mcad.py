import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from typing import Union, Sequence

import torch.nn.utils.spectral_norm as spectral_norm
from collections import OrderedDict

from einops import repeat, rearrange

from models.utils_blocks.base import BaseNetwork

from models.utils_blocks.attention import (
    MaskedTransformer,
    SinusoidalPositionalEmbedding,
)


class MultiScaleMCADiscriminator(BaseNetwork):
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
        positional_embedding_dim: int,
        num_labels: int,
        num_latent_per_labels: int,
        latent_dim: int,
        num_blocks: int,
        attention_latent_dim: int,
        num_cross_heads: int,
        num_self_heads: int,
        apply_spectral_norm: bool = True,
        concat_segmaps: bool = False,
        output_type: str = "patchgan",
        keep_intermediate_results: bool = True,
    ):
        super().__init__()

        self.keep_intermediate_results = keep_intermediate_results

        self.discriminators = nn.ModuleDict(OrderedDict())

        for i in range(num_discriminator):
            self.discriminators.update(
                {
                    f"discriminator_{i}": MaskedCrossAttentionDiscriminator(
                        image_num_channels=image_num_channels,
                        segmap_num_channels=segmap_num_channels,
                        positional_embedding_dim=positional_embedding_dim,
                        num_labels=num_labels,
                        num_latent_per_labels=num_latent_per_labels,
                        latent_dim=latent_dim,
                        num_blocks=num_blocks,
                        attention_latent_dim=attention_latent_dim,
                        num_cross_heads=num_cross_heads,
                        num_self_heads=num_self_heads,
                        apply_spectral_norm=apply_spectral_norm,
                        concat_segmaps=concat_segmaps,
                        output_type=output_type,
                        keep_intermediate_results=keep_intermediate_results,
                    ),
                }
            )
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(
        self,
        input: TensorType["batch_size", "input_channels", "height", "width"],
        segmentation_map: TensorType[
            "batch_size", "num_labels", "height", "width"
        ] = None,
    ) -> Union[
        Sequence[
            Sequence[TensorType["batch_size", 1, "output_height", "output_width"]]
        ],
        Sequence[TensorType["batch_size", 1, "output_height", "output_width"]],
    ]:
        results = []
        for disc_name in self.discriminators:
            result = self.discriminators[disc_name](input, segmentation_map)
            if not self.keep_intermediate_results:
                result = [result]
            results.append(result)
            input = self.downsample(input)
            segmentation_map = F.interpolate(
                segmentation_map, size=input.size()[2:], mode="nearest"
            )

        return results


class MaskedCrossAttentionDiscriminator(BaseNetwork):
    """
    Encoder that encode style vectors for each segmentation labels doing regional average pooling

    Parameters:
    -----------
        num_input_channels: int,
            Number of input channels
        latent_dim: int,
            Number of output channels (style_dim)
        num_features_fst_conv: iny,
            Number of kernels at first conv

    """

    def __init__(
        self,
        image_num_channels: int,
        segmap_num_channels: int,
        positional_embedding_dim: int,
        num_labels: int,
        num_latent_per_labels: int,
        latent_dim: int,
        num_blocks: int,
        attention_latent_dim: int,
        num_cross_heads: int,
        num_self_heads: int,
        apply_spectral_norm: bool = True,
        concat_segmaps: bool = False,
        output_type: str = "patchgan",
        keep_intermediate_results: bool = True,
    ):
        super().__init__()

        self.concat_segmaps = concat_segmaps
        self.output_type = output_type
        self.keep_intermediate_results = keep_intermediate_results
        self.image_pos_embs = nn.ModuleList(
            [SinusoidalPositionalEmbedding(positional_embedding_dim, emb_type="concat")]
        )
        self.convs = nn.ModuleList([nn.Identity()])

        num_input_channels = image_num_channels + segmap_num_channels

        image_emb_dim = num_input_channels + positional_embedding_dim

        num_latents = num_labels * num_latent_per_labels
        self.num_latent_per_labels = num_latent_per_labels

        latents_mask = torch.block_diag(
            *[
                torch.FloatTensor(
                    [
                        [1.0 for _ in range(num_latent_per_labels)]
                        for _ in range(num_latent_per_labels)
                    ]
                )
                for _ in range(num_labels)
            ]
        ).unsqueeze(0)
        self.register_buffer("latents_mask", latents_mask)

        self.latents = nn.Parameter(torch.Tensor(num_latents, latent_dim))

        self.backbone = nn.ModuleDict(OrderedDict())

        for i in range(num_blocks):
            module = nn.ModuleDict(OrderedDict())
            module.update(
                {
                    "cross_attention": MaskedTransformer(
                        latent_dim,
                        num_latents,
                        image_emb_dim if i == 0 else 32 * 2 ** i,
                        attention_latent_dim,
                        num_cross_heads,
                    ),
                }
            )
            module.update(
                {
                    "self_attention": MaskedTransformer(
                        latent_dim,
                        num_latents,
                        latent_dim,
                        attention_latent_dim,
                        num_self_heads,
                    ),
                }
            )
            self.backbone.update({f"block_{i}": module})
            if i > 0:
                conv = nn.Conv2d(
                    num_input_channels if i == 1 else 32 * 2 ** (i - 1),
                    32 * 2 ** i,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
                if apply_spectral_norm:
                    conv = spectral_norm(conv)
                self.convs.append(
                    nn.Sequential(
                        conv,
                        nn.LeakyReLU(0.2),
                    )
                )
                self.image_pos_embs.append(
                    SinusoidalPositionalEmbedding(32 * 2 ** i, emb_type="add")
                )
            self.conv_to_latent = nn.Sequential(
                nn.Linear(32 * 2 ** (num_blocks - 1), latent_dim),
                nn.LeakyReLU(0.2),
            )
            if self.output_type == "attention_pool":
                self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
                self.attention_pool = MaskedTransformer(
                    latent_dim,
                    1,
                    latent_dim,
                    attention_latent_dim,
                    num_self_heads,
                )
            self.classification_head = nn.Linear(latent_dim, 1)

    def forward(
        self,
        input: TensorType["batch_size", "num_input_channels", "height", "width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
    ) -> TensorType["batch_size", "num_input_channels", "style_dim"]:
        batch_size = input.shape[0]

        latents = repeat(self.latents, " l d -> b l d", b=batch_size)

        if self.concat_segmaps:
            input = torch.cat([input, segmentation_map], dim=1)

        cross_attention_masks = []
        flattened_inputs = []
        for i, conv in enumerate(self.convs):
            input = conv(input)
            cross_attention_masks.append(
                rearrange(
                    torch.repeat_interleave(
                        F.interpolate(
                            segmentation_map, size=input.size()[2:], mode="nearest"
                        ),
                        self.num_latent_per_labels,
                        dim=1,
                    ),
                    "b n h w -> b n (h w)",
                )
            )
            flattened_inputs.append(
                rearrange(self.image_pos_embs[i](input), " b c h w -> b (h w) c")
            )

        for i, block_name in enumerate(self.backbone):
            latents = self.backbone[block_name]["cross_attention"](
                latents, flattened_inputs[i], cross_attention_masks[i]
            )
        convs_features = self.conv_to_latent(flattened_inputs[-1])
        latents_and_convs = torch.cat([latents, convs_features], dim=1)

        if self.output_type == "patchgan":
            flattened_inputs.append(self.classification_head(latents_and_convs))
        elif self.output_type == "mean_pool":
            flattened_inputs.append(
                self.classification_head(latents_and_convs.mean(dim=1))
            )
        elif self.output_type == "attention_pool":
            output_token = repeat(self.cls_token, "() n d -> b n d", b=batch_size)
            output_token = self.attention_pool(output_token, latents_and_convs)
            flattened_inputs.append(self.classification_head(output_token))
        else:
            raise ValueError("Not a supported disc output type")

        if self.keep_intermediate_results:
            return flattened_inputs[1:]
        else:
            return flattened_inputs[-1]
