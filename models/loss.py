from email.policy import default
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchtyping import TensorType
from collections import OrderedDict
from typing import Literal, Union, Sequence, Tuple
from einops import rearrange


class LossWrapper(nn.Module):
    def __init__(self, cfg_losses, mode="generator"):
        super().__init__()

        self.mode = mode
        self.cfg_losses = cfg_losses
        self.losses = nn.ModuleDict()
        self.losses_conf = []
        if (
            self.cfg_losses.lambda_gan > 0 or self.cfg_losses.lambda_gan_end > 0
        ) and mode == "generator":
            self.losses_conf.append(
                {
                    "name": "gen_gan_loss",
                    "loss_weight": SwitchLambdaLoss(
                        self.cfg_losses.lambda_gan,
                        self.cfg_losses.lambda_gan_end,
                        self.cfg_losses.lambda_gan_decay_steps,
                    ),
                    "loss_type": "disc_features_and_segmaps",
                    "sub_group": "gen_gan",
                }
            )
            self.losses.update(
                {
                    "gen_gan_loss": GANLoss(
                        self.cfg_losses.gan_loss_type, mode="generator"
                    )
                }
            )
        if (
            self.cfg_losses.lambda_perceptual > 0
            or self.cfg_losses.lambda_perceptual_end > 0
        ) and mode == "generator":
            self.losses_conf.append(
                {
                    "name": "perceptual_loss",
                    "loss_weight": DecayLambdaLoss(
                        self.cfg_losses.lambda_perceptual,
                        self.cfg_losses.lambda_perceptual_end,
                        self.cfg_losses.lambda_perceptual_decay_steps,
                    ),
                    "loss_type": "images",
                    "sub_group": "reconstruction",
                }
            )
            self.losses.update({"perceptual_loss": PerceptualLoss()})
        if (
            self.cfg_losses.lambda_fm > 0 or self.cfg_losses.lambda_fm_end > 0
        ) and mode == "generator":
            self.losses_conf.append(
                {
                    "name": "fm_loss",
                    "loss_weight": DecayLambdaLoss(
                        self.cfg_losses.lambda_fm,
                        self.cfg_losses.lambda_fm_end,
                        self.cfg_losses.lambda_fm_decay_steps,
                    ),
                    "loss_type": "disc_features",
                    "sub_group": "reconstruction",
                }
            )
            self.losses.update({"fm_loss": FeatureMatchingLoss()})
        if (
            self.cfg_losses.lambda_l1 > 0 or self.cfg_losses.lambda_l1_end > 0
        ) and mode == "generator":
            self.losses_conf.append(
                {
                    "name": "l1_loss",
                    "loss_weight": DecayLambdaLoss(
                        self.cfg_losses.lambda_l1,
                        self.cfg_losses.lambda_l1_end,
                        self.cfg_losses.lambda_l1_decay_steps,
                    ),
                    "loss_type": "images",
                    "sub_group": "reconstruction",
                }
            )
            self.losses.update({"l1_loss": nn.L1Loss()})
        if (
            self.cfg_losses.lambda_kld > 0 or self.cfg_losses.lambda_kld_end > 0
        ) and mode == "generator":
            self.losses_conf.append(
                {
                    "name": "kld_loss",
                    "loss_weight": DecayLambdaLoss(
                        self.cfg_losses.lambda_kld,
                        self.cfg_losses.lambda_kld_end,
                        self.cfg_losses.lambda_kld_decay_steps,
                    ),
                    "loss_type": "latents",
                    "sub_group": "enc_regularization",
                }
            )
            self.losses.update({"kld_loss": KLDLoss()})

        if (
            self.cfg_losses.lambda_gan > 0 or self.cfg_losses.lambda_gan_end > 0
        ) and mode == "discriminator":
            self.losses_conf.append(
                {
                    "name": [
                        "true_disc_gan_loss",
                        "false_disc_gan_loss",
                        "disc_gan_loss",
                    ],
                    "loss_weight": SwitchLambdaLoss(
                        self.cfg_losses.lambda_gan,
                        self.cfg_losses.lambda_gan_end,
                        self.cfg_losses.lambda_gan_decay_steps,
                    ),
                    "loss_type": "disc_features_and_segmaps",
                    "sub_group": "disc_gan",
                }
            )
            self.losses.update(
                {
                    "disc_gan_loss": GANLoss(
                        self.cfg_losses.gan_loss_type, mode="discriminator"
                    )
                }
            )

        if (
            self.cfg_losses.lambda_r1 > 0 or self.cfg_losses.lambda_r1_end > 0
        ) and mode == "discriminator":
            self.losses_conf.append(
                {
                    "name": "r1_reg",
                    "loss_weight": DecayLambdaLoss(
                        self.cfg_losses.lambda_r1,
                        self.cfg_losses.lambda_r1_end,
                        self.cfg_losses.lambda_r1_decay_steps,
                    ),
                    "loss_type": "real_inputs",
                    "sub_group": "disc_regularization",
                }
            )
            self.losses.update({"r1_reg": R1Reg(self.cfg_losses.lazy_r1_step)})
        if (
            self.cfg_losses.lambda_label_mix > 0
            or self.cfg_losses.lambda_label_mix_end > 0
        ) and mode == "discriminator":
            self.losses_conf.append(
                {
                    "name": "label_mix_reg",
                    "loss_weight": DecayLambdaLoss(
                        self.cfg_losses.lambda_label_mix,
                        self.cfg_losses.lambda_label_mix_end,
                        self.cfg_losses.lambda_label_mix_decay_steps,
                    ),
                    "loss_type": "inputs_and_segmap",
                    "sub_group": "disc_regularization",
                }
            )
            self.losses.update({"label_mix_reg": LabelMixReg()})

    def forward(
        self,
        real_images=None,
        fake_img=None,
        segmap=None,
        real_disc=None,
        fake_disc=None,
        latents=None,
        last_layer=None,
    ):
        loss = (
            0  # torch.FloatTensor(1).fill_(0).to(fake_img.device).requires_grad_(True)
        )
        losses_log = []
        losses_subgroups = {}
        lambda_subgroups = {}

        for loss_item in self.losses_conf:
            name = loss_item["name"]
            if self.training:
                loss_weight = loss_item["loss_weight"].step()
            else:
                loss_weight = loss_item["loss_weight"].get()
            loss_type = loss_item["loss_type"]
            sub_group = loss_item["sub_group"]
            if type(name) == list:
                loss_func = self.losses[name[-1]]
            else:
                loss_func = self.losses[name]
            if loss_type == "images":
                loss_value = loss_func(real_images, fake_img)
            elif loss_type == "disc_features":
                loss_value = loss_func(real_disc, fake_disc)
            elif loss_type == "disc_features_and_segmaps":
                loss_value = loss_func(real_disc, fake_disc, segmap)
            elif loss_type == "latents":
                loss_value = loss_func(latents)
            elif loss_type == "real_inputs":
                loss_value = loss_func(real_images, real_disc)
            elif loss_type == "inputs_and_segmap":
                loss_value = loss_func(
                    real_images, fake_img, segmap, real_disc, fake_disc
                )
            if type(loss_value) == tuple:
                loss_tuple = 0
                for i in range(len(name) - 1):
                    loss_tuple += loss_value[i]
                    losses_log.append(
                        {"name": name[i], "value": loss_value[i].detach()}
                    )
                losses_subgroups[sub_group] = (
                    losses_subgroups.get(sub_group, 0) + loss_weight * loss_tuple
                )
                lambda_subgroups[sub_group] = (
                    lambda_subgroups.get(sub_group, 0) + loss_weight
                )
                losses_log.append({"name": name[-1], "value": loss_tuple.detach()})
                losses_log.append({"name": f"lambda_{name[-1]}", "value": loss_weight})
            else:
                if loss_value is not None:
                    losses_subgroups[sub_group] = (
                        losses_subgroups.get(sub_group, 0) + loss_weight * loss_value
                    )
                    lambda_subgroups[sub_group] = (
                        lambda_subgroups.get(sub_group, 0) + loss_weight
                    )
                    losses_log.append({"name": name, "value": loss_value.detach()})
                    losses_log.append({"name": f"lambda_{name}", "value": loss_weight})
        for key, value in losses_subgroups.items():
            losses_log.append({"name": key, "value": value.detach()})
            if self.cfg_losses.use_adaptive_lambda and key == "gen_gan":
                if lambda_subgroups["gen_gan"]>0 and lambda_subgroups["reconstruction"]>0:
                    try:
                        lambda_adaptive = self.compute_adaptive_lambda(
                            losses_subgroups["reconstruction"]
                            / lambda_subgroups["reconstruction"],
                            value / lambda_subgroups["gen_gan"],
                            last_layer,
                        )
                        losses_log.append(
                            {"name": "lambda_adaptive", "value": lambda_adaptive}
                        )
                    except RuntimeError:
                        assert not self.training
                        lambda_adaptive = 1
                else:
                    lambda_adaptive = 1
                value = lambda_adaptive * value
            loss += value
        losses_log.append(
            {
                "name": "gen_loss" if self.mode == "generator" else "disc_loss",
                "value": loss.detach(),
            }
        )
        return loss, losses_log

    def compute_adaptive_lambda(self, nll_loss, g_loss, last_layer=None):
        "Taken from Taming Transformers"
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight


class DecayLambdaLoss:
    def __init__(self, lambda_start, lambda_end=None, n_iterations=None):
        super(DecayLambdaLoss, self).__init__()
        self.lambda_start = lambda_start
        if lambda_end is None or n_iterations is None:
            self.lambda_end = lambda_start
            self.n_iterations = 1
        else:
            self.lambda_end = lambda_end
            self.n_iterations = n_iterations
        self.current_iteration = 0

    def step(self):
        self.current_iteration += 1
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (
            min(self.current_iteration, self.n_iterations) / self.n_iterations
        )

    def get(self):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (
            min(self.current_iteration, self.n_iterations) / self.n_iterations
        )


class SwitchLambdaLoss:
    def __init__(self, lambda_start, lambda_end=None, n_iterations=None):
        super(SwitchLambdaLoss, self).__init__()
        self.lambda_start = lambda_start
        if lambda_end is None or n_iterations is None:
            self.lambda_end = lambda_start
            self.n_iterations = 1
        else:
            self.lambda_end = lambda_end
            self.n_iterations = n_iterations
        self.current_iteration = 0

    def step(self):
        self.current_iteration += 1
        if self.current_iteration < self.n_iterations:
            return float(self.lambda_start)
        else:
            return float(self.lambda_end)

    def get(self):
        if self.current_iteration < self.n_iterations:
            return float(self.lambda_start)
        else:
            return float(self.lambda_end)


class GANLoss(nn.Module):
    """
    GAN Hinge Loss
    """

    def __init__(
        self,
        loss_name: str,
        mode: Literal["generator", "discriminator"] = "generator",
    ):
        super().__init__()

        self.loss_name = loss_name
        self.mode = mode

    def _hinge_loss(
        self,
        true_input: TensorType[...],
        false_input: TensorType[...],
    ) -> Union[Tuple[float, float], float]:
        if self.mode == "generator":
            loss_gen = -false_input.mean()
            return loss_gen
        elif self.mode == "discriminator":
            loss_true = torch.max(1 - true_input, torch.zeros_like(true_input)).mean()
            loss_false = torch.max(
                1 + false_input, torch.zeros_like(false_input)
            ).mean()
            return loss_true, loss_false
        else:
            raise ValueError("Mode not supported")

    def _non_saturating_loss(
        self,
        true_input: TensorType[...],
        false_input: TensorType[...],
    ) -> Union[Tuple[float, float], float]:
        if self.mode == "generator":
            loss_gen = F.softplus(-false_input).mean()
            return loss_gen
        elif self.mode == "discriminator":
            loss_true = F.softplus(-true_input).mean()
            loss_false = F.softplus(false_input).mean()
            return loss_true, loss_false
        else:
            raise ValueError("Mode not supported")

    def _oasis(
        self,
        true_input: TensorType[...],
        false_input: TensorType[...],
        segmap: TensorType[...] = None,
    ) -> Union[Tuple[float, float], float]:
        weight_map = get_class_balancing(segmap)

        fake_labels = torch.zeros(
            segmap.shape[0],
            segmap.shape[2],
            segmap.shape[3],
            device=true_input.device,
        ).long()

        true_labels = segmap.argmax(dim=1) + 1
        if self.mode == "generator":
            loss_gen = F.cross_entropy(false_input, true_labels, reduction="none")
            loss_gen = torch.mean(loss_gen * weight_map[:, 0, :, :])
            return loss_gen
        elif self.mode == "discriminator":
            loss_true = F.cross_entropy(true_input, true_labels, reduction="none")
            loss_true = torch.mean(loss_true * weight_map[:, 0, :, :])
            loss_false = F.cross_entropy(false_input, fake_labels, reduction="none")
            loss_false = torch.mean(loss_false)
            return loss_true, loss_false
        else:
            raise ValueError("Mode not supported")

    def loss(
        self,
        true_input: TensorType[...],
        false_input: TensorType[...],
        segmap: TensorType[...] = None,
    ) -> Union[Tuple[float, float], float]:
        if self.loss_name == "hinge":
            return self._hinge_loss(true_input, false_input)
        elif self.loss_name == "non_saturating":
            return self._non_saturating_loss(true_input, false_input)
        elif self.loss_name == "oasis":
            return self._oasis(true_input, false_input, segmap)
        else:
            raise ValueError("Loss type not supported")

    def forward(
        self,
        true_input: Union[
            Sequence[Sequence[TensorType[...]]],
            Sequence[TensorType[...]],
            TensorType[...],
        ],
        false_input: Union[
            Sequence[Sequence[TensorType[...]]],
            Sequence[TensorType[...]],
            TensorType[...],
        ],
        segmap: TensorType[...] = None,
    ) -> Union[Tuple[float, float], float]:
        if type(true_input) != type(false_input):
            raise TypeError(
                "True input and False input shoudl be either both lists or both tensors"
            )
        elif isinstance(true_input, list):
            if self.mode == "generator":
                loss = 0
                assert len(true_input) == len(false_input)

                num_disc = len(true_input)
                for i in range(num_disc):
                    if isinstance(true_input[i], list):
                        true_input_logits = true_input[i][-1]
                    else:
                        true_input_logits = true_input[i]
                    if isinstance(false_input[i], list):
                        false_input_logits = false_input[i][-1]
                    else:
                        false_input_logits = false_input[i]

                    loss += self.loss(true_input_logits, false_input_logits, segmap)
                return loss / num_disc
            elif self.mode == "discriminator":
                loss_true = 0
                loss_false = 0
                assert len(true_input) == len(false_input)
                num_disc = len(true_input)
                for i in range(num_disc):
                    if isinstance(true_input[i], list):
                        true_input_logits = true_input[i][-1]
                    else:
                        true_input_logits = true_input[i]
                    if isinstance(false_input[i], list):
                        false_input_logits = false_input[i][-1]
                    else:
                        false_input_logits = false_input[i]

                    losses = self.loss(true_input_logits, false_input_logits, segmap)
                    loss_true += losses[0]
                    loss_false += losses[1]
                return loss_true / num_disc, loss_false / num_disc

        else:
            return self.loss(true_input, false_input, segmap)


class PerceptualLoss(nn.Module):
    """ "
    Perceptual Loss based on VGG19 network
    """

    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.vgg = nn.ModuleDict(
            OrderedDict(
                {
                    "vgg_slice_1": torch.nn.Sequential(
                        *[vgg_pretrained_features[i] for i in range(2)]
                    ),
                    "vgg_slice_2": torch.nn.Sequential(
                        *[vgg_pretrained_features[i] for i in range(2, 7)]
                    ),
                    "vgg_slice_3": torch.nn.Sequential(
                        *[vgg_pretrained_features[i] for i in range(7, 12)]
                    ),
                    "vgg_slice_4": torch.nn.Sequential(
                        *[vgg_pretrained_features[i] for i in range(12, 21)]
                    ),
                    "vgg_slice_5": torch.nn.Sequential(
                        *[vgg_pretrained_features[i] for i in range(21, 30)]
                    ),
                }
            )
        )
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: TensorType["batch_size", "num_input_channels", "height", "width"],
        y: TensorType["batch_size", "num_input_channels", "height", "width"],
    ) -> float:
        loss = 0
        for i, slice_name in enumerate(self.vgg):
            x = self.vgg[slice_name](x)
            y = self.vgg[slice_name](y)
            loss += self.weights[i] * self.criterion(x, y)
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching loss across discriminator activations
    """

    def __init__(self):
        super().__init__()

        self.criterion = nn.L1Loss()

    def forward(
        self,
        true_feat: Sequence[Sequence[TensorType[...]]],
        false_feat: Sequence[Sequence[TensorType[...]]],
    ) -> float:
        loss = 0  # torch.FloatTensor(1).fill_(0).to(true_feat[0][0].device).requires_grad_()
        for i in range(len(true_feat)):
            for j in range(len(true_feat[i]) - 1):
                loss += self.criterion(false_feat[i][j], true_feat[i][j].detach())
        loss /= len(true_feat)
        return loss


class KLDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def kld_loss(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # if kl.dim() > 2:
        #     kl = kl.mean(1)
        return kl.sum()

    def forward(self, latents):
        assert isinstance(latents, Sequence)
        assert len(latents) % 2 == 0
        loss = 0
        for i in range(len(latents) // 2):
            loss += self.kld_loss(latents[2 * i], latents[2 * i + 1])
        return loss / (len(latents) // 2)


#### Regularizations ####
class R1Reg(nn.Module):
    def __init__(self, do_r1_each_step):
        super().__init__()
        self.counter = 0
        self.do_r1_each_step = do_r1_each_step

    def forward(self, real_images, real_disc_outputs):
        self.counter += 1
        if self.counter == self.do_r1_each_step and self.training:
            self.counter = 0
            r1_regu = 0
            for disc_num in range(len(real_disc_outputs)):
                r1_regu += self.r1_penalty(real_images, real_disc_outputs[disc_num][-1])
            return r1_regu / len(real_disc_outputs)

    def r1_penalty(self, real_images, real_disc_outputs):
        gradients = torch.autograd.grad(
            outputs=real_disc_outputs.sum(),
            inputs=real_images,
            create_graph=True,
            # grad_outputs=torch.ones_like(real_disc_outputs),
        )[0]
        gradients = rearrange(gradients, "b ... -> b (...)")
        return (gradients.pow(2).sum(1).mean()) / 2


class LabelMixReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_mix_criterium = nn.MSELoss()

    def forward(
        self,
        real_images,
        generated_images,
        segmentation_maps,
        real_disc,
        generated_disc,
    ):
        target_map = torch.argmax(segmentation_maps, dim=1, keepdim=True)
        all_classes = torch.unique(target_map)
        for c in all_classes:
            target_map[target_map == c] = torch.randint(0, 2, (1,), device=self.device)
        target_map = target_map.float()
        mixed_image = (
            target_map * real_images + (1 - target_map) * generated_images.detach()
        )
        mixed_disc = self.discriminator(mixed_image, segmentation_maps)
        mixed_D_output = target_map * real_disc + (1 - target_map) * generated_disc
        label_mix_reg = self.label_mix_criterium(mixed_disc, mixed_D_output)
        return label_mix_reg


def get_class_balancing(label):
    class_occurence = torch.sum(label, dim=(0, 2, 3))
    num_of_classes = (class_occurence > 0).sum()
    coefficients = torch.reciprocal(class_occurence)
    coefficients = torch.nan_to_num(coefficients, nan=0.0, posinf=0.0)
    coefficients = num_of_classes * coefficients / coefficients.sum()
    integers = torch.argmax(label, dim=1, keepdim=True)
    weight_map = coefficients[integers]
    if torch.isnan(weight_map).any():
        print("has nan")
    return weight_map
