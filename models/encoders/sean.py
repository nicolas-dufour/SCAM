import torch
import torch.nn as nn
import torch.nn.functional as F


from torchtyping import TensorType
from collections import OrderedDict
from models.utils_blocks.base import BaseNetwork


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
        use_vae: bool = False,
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

        self.use_vae = use_vae
        if self.use_vae:
            self.vae_mapper = nn.Linear(latent_dim, 2*latent_dim)
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
        if self.use_vae:
            style_codes = self.vae_mapper(style_codes)
            style_codes = style_codes.chunk(2, dim=-1)
        return style_codes, None
