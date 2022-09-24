from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from torchtyping import TensorType
from typing import Tuple
from collections import OrderedDict
import torch.nn.utils.spectral_norm as spectral_norm
from models.utils_blocks.base import BaseNetwork


from models.utils_blocks.attention import (
    MaskedTransformer,
    SinusoidalPositionalEmbedding,
)
from utils.partial_conv import InstanceAwareConv2d

from models.utils_blocks.equallr import EqualConv2d
from functools import partial


class MaskedAttentionNoiseMapper(BaseNetwork):
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
        num_labels: int,
        num_latent_per_labels: int,
        latent_dim: int,
        num_blocks: int,
        attention_latent_dim: int,
        num_self_heads: int,
        num_latents_bg: int = None,
        use_semantic_masking=True,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        if num_latents_bg is None:
            num_latents_bg = num_latent_per_labels

        self.lr_mul = lr_mul
        num_latents = (num_labels - 1) * num_latent_per_labels + num_latents_bg
        self.num_latent_per_labels = num_latent_per_labels
        self.num_latents_bg = num_latents_bg

        if use_semantic_masking:
            latents_mask = [
                torch.FloatTensor(
                    [
                        [1.0 for _ in range(num_latents_bg)]
                        for _ in range(num_latents_bg)
                    ]
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
        else:
            self.latents_mask = None

        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.backbone = nn.ModuleDict(OrderedDict())

        for i in range(num_blocks):
            module = nn.ModuleDict(OrderedDict())
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
        latents = torch.randn(
            batch_size, self.num_latents, self.latent_dim, device=input.device
        )
        for i, block_name in enumerate(self.backbone):
            latents = self.backbone[block_name]["self_attention"](
                latents, latents, self.latents_mask
            )
        return latents, None


class MaskedAttentionStyleEncoder(BaseNetwork):
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


class RegionalAveragePoolingStyleEncoder(BaseNetwork):
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
        latent_dim: int,
        num_features_fst_conv: int = 32,
    ):
        super().__init__()

        nffc = num_features_fst_conv
        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_input_channels, nffc, kernel_size=3, padding=0),
            nn.InstanceNorm2d(nffc),
            nn.LeakyReLU(0.2, False),
        )
        self.bottleneck = nn.ModuleDict(OrderedDict())

        for i in range(2):
            mult = 2**i
            module = nn.Sequential(
                nn.Conv2d(
                    mult * nffc,
                    2 * mult * nffc,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(2 * mult * nffc),
                nn.LeakyReLU(0.2, False),
            )
            self.bottleneck.update({f"down_{i}": module})

        self.bottleneck.update(
            {
                f"up_{1}": nn.Sequential(
                    nn.ConvTranspose2d(
                        4 * nffc,
                        nffc * 8,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(4 * nffc),
                    nn.LeakyReLU(0.2, False),
                )
            }
        )

        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nffc * 8, latent_dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(nffc),
            nn.Tanh(),
        )

    def forward(
        self,
        input: TensorType["batch_size", "num_input_channels", "height", "width"],
        segmentation_map: TensorType["batch_size", "num_labels", "height", "width"],
    ) -> TensorType["batch_size", "num_input_channels", "style_dim"]:
        x = self.first_layer(input)
        for _, block_name in enumerate(self.bottleneck):
            x = self.bottleneck[block_name](x)
        x = self.last_layer(x)
        segmentation_map = F.interpolate(
            segmentation_map, size=x.size()[2:], mode="nearest"
        )
        (batch_size, style_dim, *_) = x.shape
        num_labels = segmentation_map.shape[1]
        style_codes = torch.zeros(batch_size, num_labels, style_dim, device=x.device)
        for i in range(num_labels):
            num_components = segmentation_map[:, i].unsqueeze(1).sum((2, 3))
            num_components[num_components == 0] = 1
            style_codes[:, i] = (segmentation_map[:, i].unsqueeze(1) * x).sum(
                (2, 3)
            ) / num_components
        return style_codes, None


class SPADEStyleEncoder(BaseNetwork):
    def __init__(self):
        super().__init__()
        kw = 3
        pw = 1
        ndf = 64
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, ndf, kw, stride=2, padding=pw)),
            nn.InstanceNorm2d(ndf, affine=False),
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw)),
            nn.InstanceNorm2d(ndf * 2, affine=False),
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw)),
            nn.InstanceNorm2d(ndf * 4, affine=False),
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw)),
            nn.InstanceNorm2d(ndf * 8, affine=False),
        )
        self.layer5 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw)),
            nn.InstanceNorm2d(ndf * 8, affine=False),
        )
        self.layer6 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw)),
            nn.InstanceNorm2d(ndf * 8, affine=False),
        )

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, image, segmap=None):
        if image.shape[2] != 256 or image.shape[3] != 256:
            image = F.interpolate(image, size=(256, 256), mode="bilinear")
        image = self.layer1(image)
        image = self.layer2(self.actvn(image))
        image = self.layer3(self.actvn(image))
        image = self.layer4(self.actvn(image))
        image = self.layer5(self.actvn(image))
        image = self.layer6(self.actvn(image))

        image = self.actvn(image)

        image = image.view(image.size(0), -1)
        mu = self.fc_mu(image)
        logvar = self.fc_var(image)

        return [mu, logvar], None


class GroupDNetStyleEncoder(BaseNetwork):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        kw = 3
        pw = 1
        ndf = 32 * num_labels
        self.layer1 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    3 * num_labels, ndf, kw, stride=2, padding=pw, groups=num_labels
                )
            ),
            nn.InstanceNorm2d(ndf, affine=False),
        )
        self.layer2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw, groups=num_labels)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=False),
        )
        self.layer3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw, groups=num_labels)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=False),
        )
        self.layer4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw, groups=num_labels)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=False),
        )
        self.layer5 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw, groups=num_labels)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=False),
        )
        self.layer6 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw, groups=num_labels)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=False),
        )

        self.so = s0 = 4
        self.fc_mu = nn.Conv2d(
            ndf * 8, num_labels * 8, kw, stride=1, padding=pw, groups=num_labels
        )
        self.fc_var = nn.Conv2d(
            ndf * 8, num_labels * 8, kw, stride=1, padding=pw, groups=num_labels
        )

        self.actvn = nn.LeakyReLU(0.2, False)

    def trans_img(self, input_semantics, real_image):
        images = None
        seg_range = input_semantics.size()[1]
        for i in range(input_semantics.size(0)):
            resize_image = None
            for n in range(0, seg_range):
                seg_image = real_image[i] * input_semantics[i][n]
                # resize seg_image
                c_sum = seg_image.sum(dim=0)
                y_seg = c_sum.sum(dim=0)
                x_seg = c_sum.sum(dim=1)
                y_id = y_seg.nonzero()
                if y_id.size()[0] == 0:
                    seg_image = seg_image.unsqueeze(dim=0)
                    # resize_image = torch.cat((resize_image, seg_image), dim=0)
                    if resize_image is None:
                        resize_image = seg_image
                    else:
                        resize_image = torch.cat((resize_image, seg_image), dim=1)
                    continue
                # print(y_id)
                y_min = y_id[0][0]
                y_max = y_id[-1][0]
                x_id = x_seg.nonzero()
                x_min = x_id[0][0]
                x_max = x_id[-1][0]
                seg_image = seg_image.unsqueeze(dim=0)
                seg_image = F.interpolate(
                    seg_image[:, :, x_min : x_max + 1, y_min : y_max + 1],
                    size=[256, 256],
                )
                if resize_image is None:
                    resize_image = seg_image
                else:
                    resize_image = torch.cat((resize_image, seg_image), dim=1)
            if images is None:
                images = resize_image
            else:
                images = torch.cat((images, resize_image), dim=0)
        return images

    def forward(self, image, segmap=None):
        image = self.trans_img(segmap, image)
        image = self.layer1(image)
        image = self.layer2(self.actvn(image))
        image = self.layer3(self.actvn(image))
        image = self.layer4(self.actvn(image))
        image = self.layer5(self.actvn(image))
        image = self.layer6(self.actvn(image))

        image = self.actvn(image)

        mu = self.fc_mu(image)
        logvar = self.fc_var(image)

        return [mu, logvar], None


class InstanceAdaptiveEncoder(BaseNetwork):
    def __init__(self, num_labels, noise_dim):
        super().__init__()
        kw = 3
        pw = 1
        ndf = 64
        conv_layer = InstanceAwareConv2d

        self.layer1 = conv_layer(3, ndf, kw, stride=2, padding=pw)
        self.norm1 = nn.InstanceNorm2d(ndf)
        self.layer2 = conv_layer(ndf * 1, ndf * 2, kw, stride=2, padding=pw)
        self.norm2 = nn.InstanceNorm2d(ndf * 2)
        self.layer3 = conv_layer(ndf * 2, ndf * 4, kw, stride=2, padding=pw)
        self.norm3 = nn.InstanceNorm2d(ndf * 4)
        self.layer4 = conv_layer(ndf * 4, ndf * 8, kw, stride=2, padding=pw)
        self.norm4 = nn.InstanceNorm2d(ndf * 8)

        self.middle = conv_layer(ndf * 8, ndf * 4, kw, stride=1, padding=pw)
        self.norm_middle = nn.InstanceNorm2d(ndf * 4)
        self.up1 = conv_layer(ndf * 8, ndf * 2, kw, stride=1, padding=pw)
        self.norm_up1 = nn.InstanceNorm2d(ndf * 2)
        self.up2 = conv_layer(ndf * 4, ndf * 1, kw, stride=1, padding=pw)
        self.norm_up2 = nn.InstanceNorm2d(ndf)
        self.up3 = conv_layer(ndf * 2, ndf, kw, stride=1, padding=pw)
        self.norm_up3 = nn.InstanceNorm2d(ndf)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.num_labels = num_labels

        self.scale_conv_mu = conv_layer(ndf, noise_dim, kw, stride=1, padding=pw)
        self.scale_conv_var = conv_layer(ndf, noise_dim, kw, stride=1, padding=pw)
        self.bias_conv_mu = conv_layer(ndf, noise_dim, kw, stride=1, padding=pw)
        self.bias_conv_var = conv_layer(ndf, noise_dim, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)

    def instAvgPooling(self, x, instances):
        inst_num = instances.size()[1]
        for i in range(inst_num):
            inst_mask = torch.unsqueeze(instances[:, i, :, :], 1)  # [n,1,h,w]
            pixel_num = torch.sum(
                torch.sum(inst_mask, dim=2, keepdim=True), dim=3, keepdim=True
            )
            pixel_num[pixel_num == 0] = 1
            feat = x * inst_mask
            feat = (
                torch.sum(torch.sum(feat, dim=2, keepdim=True), dim=3, keepdim=True)
                / pixel_num
            )
            if i == 0:
                out = torch.unsqueeze(feat[:, :, 0, 0], 1)  # [n,1,c]
            else:
                out = torch.cat([out, torch.unsqueeze(feat[:, :, 0, 0], 1)], 1)
            # inst_pool_feats.append(feat[:,:,0,0]) # [n, 64]
        return out

    def forward(self, real_image, input_semantics):
        # instances [n,1,h,w], input_instances [n,inst_nc,h,w]
        instances = torch.argmax(input_semantics, 1, keepdim=True).float()
        x1 = self.actvn(self.norm1(self.layer1(real_image, instances)))
        x2 = self.actvn(self.norm2(self.layer2(x1, instances)))
        x3 = self.actvn(self.norm3(self.layer3(x2, instances)))
        x4 = self.actvn(self.norm4(self.layer4(x3, instances)))
        y = self.up(self.actvn(self.norm_middle(self.middle(x4, instances))))
        y1 = self.up(
            self.actvn(self.norm_up1(self.up1(torch.cat([y, x3], 1), instances)))
        )
        y2 = self.up(
            self.actvn(self.norm_up2(self.up2(torch.cat([y1, x2], 1), instances)))
        )
        y3 = self.up(
            self.actvn(self.norm_up3(self.up3(torch.cat([y2, x1], 1), instances)))
        )

        scale_mu = self.scale_conv_mu(y3, instances)
        scale_var = self.scale_conv_var(y3, instances)
        bias_mu = self.bias_conv_mu(y3, instances)
        bias_var = self.bias_conv_var(y3, instances)

        scale_mus = self.instAvgPooling(scale_mu, input_semantics)
        scale_vars = self.instAvgPooling(scale_var, input_semantics)
        bias_mus = self.instAvgPooling(bias_mu, input_semantics)
        bias_vars = self.instAvgPooling(bias_var, input_semantics)

        return (scale_mus, scale_vars, bias_mus, bias_vars), None
