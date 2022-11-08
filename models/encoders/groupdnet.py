import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from models.utils_blocks.base import BaseNetwork

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