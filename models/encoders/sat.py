import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from torchtyping import TensorType
from typing import Tuple
from collections import OrderedDict
from models.utils_blocks.base import BaseNetwork


from models.utils_blocks.attention import (
    MaskedTransformer,
    SinusoidalPositionalEmbedding,
)

from models.utils_blocks.equallr import EqualConv2d
from functools import partial


class SemanticAttentionTransformerEncoder(BaseNetwork):
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
        num_input_channels: int,
        positional_embedding_dim: int,
        num_labels: int,
        num_latent_per_labels: int,
        latent_dim: int,
        num_blocks: int,
        attention_latent_dim: int,
        num_cross_heads: int,
        num_self_heads: int,
        type_of_initial_latents: str = "learned",
        image_conv: bool = False,
        reverse_conv: bool = False,
        num_latents_bg: int = None,
        conv_features_dim_first: int = 16,
        content_dim: int = 512,
        use_vae: bool = False,
        use_self_attention: bool = True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        nf = conv_features_dim_first
        if num_latents_bg is None:
            num_latents_bg = num_latent_per_labels
        if image_conv:
            self.image_pos_embs = nn.ModuleList(
                [
                    SinusoidalPositionalEmbedding(
                        positional_embedding_dim, emb_type="concat"
                    )
                ]
            )
            self.convs = nn.ModuleList([nn.Identity()])

        else:
            self.image_pos_emb = SinusoidalPositionalEmbedding(
                positional_embedding_dim, emb_type="concat"
            )
        ConvLayer = (
            partial(EqualConv2d, lr_mul=lr_mul) if use_equalized_lr else nn.Conv2d
        )
        self.lr_mul = lr_mul
        self.reverse_conv = reverse_conv
        self.image_conv = image_conv
        self.use_self_attention = use_self_attention

        image_emb_dim = num_input_channels + positional_embedding_dim

        num_latents = (num_labels - 1) * num_latent_per_labels + num_latents_bg
        self.num_latent_per_labels = num_latent_per_labels
        self.num_latents_bg = num_latents_bg

        self.return_attention = False

        latents_mask = [
            torch.FloatTensor(
                [[1.0 for _ in range(num_latents_bg)] for _ in range(num_latents_bg)]
            )
        ] + [
            torch.FloatTensor(
                [
                    [1.0 for _ in range(num_latent_per_labels)]
                    for _ in range(num_latent_per_labels)
                ]
            )
            for _ in range(num_labels - 1)
        ]

        latents_mask = torch.block_diag(*latents_mask).unsqueeze(0)
        self.register_buffer("latents_mask", latents_mask)

        self.type_of_initial_latents = type_of_initial_latents
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        if self.type_of_initial_latents == "learned":
            self.latents = nn.Parameter(
                torch.randn(num_latents, latent_dim).div_(lr_mul)
            )
        elif self.type_of_initial_latents == "fixed_random":
            self.latents = torch.randn(num_latents, latent_dim)

        self.backbone = nn.ModuleDict(OrderedDict())

        for i in range(num_blocks):
            module = nn.ModuleDict(OrderedDict())
            if reverse_conv:
                module.update(
                    {
                        "cross_attention": MaskedTransformer(
                            latent_dim,
                            num_latents,
                            image_emb_dim
                            if i == num_blocks - 1 or not image_conv
                            else nf * 2 ** (num_blocks - 1 - i),
                            attention_latent_dim,
                            num_cross_heads,
                            use_equalized_lr=use_equalized_lr,
                            lr_mul=lr_mul,
                        ),
                    }
                )
            else:
                module.update(
                    {
                        "cross_attention": MaskedTransformer(
                            latent_dim,
                            num_latents,
                            image_emb_dim if i == 0 or not image_conv else nf * 2**i,
                            attention_latent_dim,
                            num_cross_heads,
                            use_equalized_lr=use_equalized_lr,
                            lr_mul=lr_mul,
                        ),
                    }
                )
            if use_self_attention:
                module.update(
                    {
                        "self_attention": MaskedTransformer(
                            latent_dim,
                            num_latents,
                            latent_dim,
                            attention_latent_dim,
                            num_self_heads,
                            use_equalized_lr=use_equalized_lr,
                            lr_mul=lr_mul,
                        ),
                    }
                )
            self.backbone.update({f"block_{i}": module})
            if i > 0 and image_conv:
                self.convs.append(
                    nn.Sequential(
                        ConvLayer(
                            num_input_channels if i == 1 else nf * 2 ** (i - 1),
                            nf * 2**i,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                        ),
                        nn.LeakyReLU(0.2),
                    )
                )
                self.image_pos_embs.append(
                    SinusoidalPositionalEmbedding(nf * 2**i, emb_type="add")
                )
            self.use_vae = use_vae

            if self.use_vae:
                self.vae_mapper = nn.ModuleList([nn.Linear(latent_dim, 2*latent_dim) for _ in range(num_latents)])

    def forward(
        self,
        input: TensorType["batch_size", "num_input_channels", "height", "width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
    ) -> Tuple[
        TensorType["batch_size", "num_segmap_labels", "style_dim"],
        TensorType[
            "batch_size",
            "output_dim",
            "output_fmap_heigth",
            "output_fmap_width",
        ],
    ]:
        batch_size = input.shape[0]
        if self.type_of_initial_latents == "learned":
            latents = repeat(self.latents, " l d -> b l d", b=batch_size) * self.lr_mul
        elif self.type_of_initial_latents == "fixed_random":
            latents = (
                repeat(self.latents.to(input.device), " l d -> b l d", b=batch_size)
                * self.lr_mul
            )
        elif self.type_of_initial_latents == "random":
            latents = torch.randn(
                batch_size, self.num_latents, self.latent_dim, device=input.device
            )

        if self.return_attention:
            self.image_sizes = []

        if self.image_conv:
            cross_attention_masks = []
            flattened_inputs = []
            for i, conv in enumerate(self.convs):
                input = conv(input)
                if i == len(self.convs) - 1:
                    output_fmap = input

                interpolated_segmap = F.interpolate(
                    segmentation_map, size=input.size()[2:], mode="nearest"
                )
                if self.return_attention:
                    self.image_sizes.append(input.size()[2:])
                cross_attention_masks.append(
                    rearrange(
                        torch.cat(
                            [
                                torch.repeat_interleave(
                                    interpolated_segmap[:, 0].unsqueeze(1),
                                    self.num_latents_bg,
                                    dim=1,
                                ),
                                torch.repeat_interleave(
                                    interpolated_segmap[:, 1:],
                                    self.num_latent_per_labels,
                                    dim=1,
                                ),
                            ],
                            dim=1,
                        ),
                        "b n h w -> b n (h w)",
                    )
                )

                flattened_inputs.append(
                    rearrange(self.image_pos_embs[i](input), " b c h w -> b (h w) c")
                )
            if self.reverse_conv:
                cross_attention_masks = cross_attention_masks[::-1]
                flattened_inputs = flattened_inputs[::-1]
        else:
            cross_attention_mask = rearrange(
                torch.cat(
                    [
                        torch.repeat_interleave(
                            segmentation_map[:, 0].unsqueeze(1),
                            self.num_latents_bg,
                            dim=1,
                        ),
                        torch.repeat_interleave(
                            segmentation_map[:, 1:],
                            self.num_latent_per_labels,
                            dim=1,
                        ),
                    ],
                    dim=1,
                ),
                "b n h w -> b n (h w)",
            )
            flattened_input = rearrange(
                self.image_pos_emb(input), " b c h w -> b (h w) c"
            )

        for i, block_name in enumerate(self.backbone):
            if self.image_conv:
                if self.return_attention:
                    latents, _ = self.backbone[block_name]["cross_attention"](
                        latents,
                        flattened_inputs[i],
                        cross_attention_masks[i],
                        return_attention=self.return_attention,
                    )
                else:
                    latents = self.backbone[block_name]["cross_attention"](
                        latents, flattened_inputs[i], cross_attention_masks[i]
                    )
            else:
                if self.return_attention:
                    latents, _ = self.backbone[block_name]["cross_attention"](
                        latents,
                        flattened_input,
                        cross_attention_mask,
                        return_attention=self.return_attention,
                    )
                else:
                    latents = self.backbone[block_name]["cross_attention"](
                        latents, flattened_input, cross_attention_mask
                    )
            if self.use_self_attention:
                latents = self.backbone[block_name]["self_attention"](
                    latents, latents, self.latents_mask
                )
        if self.use_vae:
            latents_vae = torch.zeros((*latents.shape[:-1], 2*latents.shape[-1]), device=latents.device)
            for i, mapper in enumerate(self.vae_mapper):
                latents_vae[:, i, :] = mapper(latents[:, i, :])
            latents = latents_vae.chunk(2, dim=-1)
        return latents, None
