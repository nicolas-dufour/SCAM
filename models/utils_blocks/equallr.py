## Modified from https://github.com/rosinality/stylegan2-pytorch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torch


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        lr_mul=1,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size).div_(
                lr_mul
            ),
            requires_grad=True,
        )
        # torch.nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        self.scale = (1 / sqrt(in_channels * kernel_size**2)) * lr_mul
        self.stride = stride
        self.padding = padding
        self.lr_mul = lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias * self.lr_mul,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim).div_(lr_mul), requires_grad=True
        )
        # torch.nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_dim).fill_(bias_init), requires_grad=True
            )

        else:
            self.bias = None
        self.scale = (1 / sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )
