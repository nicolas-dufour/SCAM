import kornia.augmentation as K
import torch.nn as nn


class SimpleAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            K.Denormalize(mean=0.5, std=0.5),
            K.ColorJitter(p=0.8, brightness=0.2, contrast=0.3, hue=0.2),
            K.RandomErasing(p=0.5),
            K.Normalize(mean=0.5, std=0.5),
        )

    def forward(self, x):
        return self.aug(x)
