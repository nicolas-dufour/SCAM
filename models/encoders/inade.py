import torch
import torch.nn as nn
from models.utils_blocks.base import BaseNetwork
from utils.partial_conv import InstanceAwareConv2d

class InstanceAdaptiveEncoder(BaseNetwork):
    def __init__(self, num_labels, noise_dim, use_vae=True):
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
