import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weight(self, do_init: bool = True):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

        if do_init:
            self.apply(init_func)

        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(do_init)
