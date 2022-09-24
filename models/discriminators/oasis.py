import torch
import torch.nn as nn

import torch.nn.utils.spectral_norm as spectral_norm

from models.utils_blocks.base import BaseNetwork


class OASISDiscriminator(BaseNetwork):
    """
    PatchGAN Discriminator. Perform patch-wise GAN discrimination to avoid ignoring local high dimensional features.

    Parameters:
    -----------
        image_num_channels: int,
            Number of channels in the real/generated image
        segmap_num_channels: int,
            Number of channels in the segmentation map
        num_features_fst_conv: int,
            Number of features in the first convolution layer
        num_layers: int,
            Number of convolution layers
        apply_spectral_norm: bool, default True
            Whether or not to apply spectral normalization
        keep_intermediate_results: bool, default True
            Whether or not to keep each feature map output
    """

    def __init__(
        self,
        image_num_channels: int,
        segmap_num_channels: int,
        apply_spectral_norm: bool = True,
        apply_grad_norm: bool = False,
    ):
        super().__init__()
        self.apply_grad_norm = apply_grad_norm
        output_channel = segmap_num_channels + 1  # for N+1 loss
        self.channels = [image_num_channels, 128, 128, 256, 256, 512, 512]
        num_res_blocks = 6
        self.body_up = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(num_res_blocks):
            self.body_down.append(
                OASISBlock(
                    self.channels[i],
                    self.channels[i + 1],
                    -1,
                    first=(i == 0),
                    apply_spectral_norm=apply_spectral_norm,
                )
            )
        # decoder part
        self.body_up.append(
            OASISBlock(
                self.channels[-1],
                self.channels[-2],
                1,
                apply_spectral_norm=apply_spectral_norm,
            )
        )
        for i in range(1, num_res_blocks - 1):
            self.body_up.append(
                OASISBlock(
                    2 * self.channels[-1 - i],
                    self.channels[-2 - i],
                    1,
                    apply_spectral_norm=apply_spectral_norm,
                )
            )
        self.body_up.append(
            OASISBlock(
                2 * self.channels[1], 64, 1, apply_spectral_norm=apply_spectral_norm
            )
        )
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input, segmap=None):
        x = input
        # encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        # decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i - 1], x), dim=1))
        ans = self.layer_up_last(x)
        if self.apply_grad_norm:
            grad = torch.autograd.grad(
                ans,
                [input],
                torch.ones_like(ans),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
            grad_norm = grad_norm.view(-1, *[1 for _ in range(len(ans.shape) - 1)])
            ans = ans / (grad_norm + torch.abs(ans))
        return ans


class OASISBlock(nn.Module):
    def __init__(
        self,
        fin: int,
        fout: int,
        up_or_down: int,
        first: bool = False,
        apply_spectral_norm: bool = True,
    ):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = fin != fout
        fmiddle = fout
        if apply_spectral_norm:
            norm_layer = spectral_norm
        else:
            norm_layer = nn.Identity()
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2, False),
                    nn.Upsample(scale_factor=2),
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)),
                )
            else:
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2, False),
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)),
                )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1))
        )
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
