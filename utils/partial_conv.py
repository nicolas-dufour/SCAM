import torch
import torch.nn.functional as F
from torch import nn
from torch import nn
import math
from torch.nn import init


class InstanceAwareConv2d(nn.Module):
    def __init__(self, fin, fout, kw, stride=1, padding=1):
        super().__init__()
        self.kw = kw
        self.stride = stride
        self.padding = padding
        self.fin = fin
        self.fout = fout
        self.unfold = nn.Unfold(kw, stride=stride, padding=padding)
        self.weight = nn.Parameter(torch.Tensor(fout, fin, kw, kw))
        self.bias = nn.Parameter(torch.Tensor(fout))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, instances, check=False):
        N, C, H, W = x.size()
        # cal the binary mask from instance map
        instances = F.interpolate(instances, x.size()[2:], mode="nearest")  # [n,1,h,w]
        inst_unf = self.unfold(instances)
        # substract the center pixel
        center = torch.unsqueeze(inst_unf[:, self.kw * self.kw // 2, :], 1)
        mask_unf = inst_unf - center
        # clip the absolute value to 0~1
        mask_unf = torch.abs(mask_unf)
        mask_unf = torch.clamp(mask_unf, 0, 1)
        mask_unf = 1.0 - mask_unf  # [n,k*k,L]
        # multiply mask_unf and x
        x_unf = self.unfold(x)  # [n,c*k*k,L]
        x_unf = x_unf.view(N, C, -1, x_unf.size()[-1])  # [n,c,,k*k,L]
        mask = torch.unsqueeze(mask_unf, 1)  # [n,1,k*k,L]
        mask_x = mask * x_unf  # [n,c,k*k,L]
        mask_x = mask_x.view(N, -1, mask_x.size()[-1])  # [n,c*k*k,L]
        # conv operation
        weight = self.weight.view(self.fout, -1)  # [fout, c*k*k]
        out = torch.einsum("cm,nml->ncl", weight, mask_x)
        # x_unf = torch.unsqueeze(x_unf, 1)  # [n,1,c*k*k,L]
        # out = torch.mul(masked_weight, x_unf).sum(dim=2, keepdim=False) # [n,fout,L]
        bias = torch.unsqueeze(torch.unsqueeze(self.bias, 0), -1)  # [1,fout,1]
        out = out + bias
        out = out.view(N, self.fout, H // self.stride, W // self.stride)
        # print('weight:',self.weight[0,0,...])
        # print('bias:',self.bias)

        if check:
            out2 = nn.functional.conv2d(
                x, self.weight, self.bias, stride=self.stride, padding=self.padding
            )
            print((out - out2).abs().max())
        return out
