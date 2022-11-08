import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from models.utils_blocks.base import BaseNetwork


class SPADEStyleEncoder(BaseNetwork):
    """
    Encoder that encode one style vector for the whole image as done in SPADE.

    Parameters:
    -----------


    """
    def __init__(self, use_vae=True):
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
