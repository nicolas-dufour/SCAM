import torch
import torch.nn as nn
import pytorch_lightning as pl

from hydra.utils import instantiate

from models.loss import LossWrapper
from metrics.fid import FID, NoTrainInceptionV3
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR

from torchtyping import TensorType

from typing import Tuple, Union, Sequence
from pathlib import Path

from torch.nn.utils import clip_grad_norm_
import itertools


class Geppetto(pl.LightningModule):
    """
    This is the pl wrapper for the Geppetto model

    Parameters:
    -----------
        cfg: ConfigDict
            See config.yaml
    """

    def __init__(self, cfg):
        super().__init__()

        # Networks
        self.cfg = cfg
        if "disc_augments" in cfg:
            self.disc_augment = instantiate(cfg.disc_augments)
        else:
            self.disc_augment = nn.Identity()
        self.generator = instantiate(cfg.generator)
        self.generator.init_weight(cfg.change_init)
        self.discriminator = instantiate(cfg.discriminator)
        self.discriminator.init_weight(cfg.change_init)
        self.encoder = instantiate(cfg.encoder)
        self.encoder.init_weight(cfg.change_init)

        # Criterions
        self.loss_computer_gen = LossWrapper(self.cfg.losses, mode="generator")
        self.loss_computer_disc = LossWrapper(self.cfg.losses, mode="discriminator")
        # Metrics
        feature_extractor = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=[str(2048)]
        )

        self.train_reco_fid = FID(
            feature_extractor, Path(cfg.dataset_path) / Path("train/stats")
        )
        self.train_swap_fid = FID(
            feature_extractor, Path(cfg.dataset_path) / Path("train/stats")
        )
        self.train_reco_psnr = PSNR()

        self.val_reco_fid = FID(
            feature_extractor, Path(cfg.dataset_path) / Path("train/stats")
        )
        self.val_swap_fid = FID(
            feature_extractor, Path(cfg.dataset_path) / Path("train/stats")
        )
        self.val_reco_psnr = PSNR()

        self.test_reco_fid = FID(
            feature_extractor, Path(cfg.dataset_path) / Path("train/stats")
        )
        self.test_swap_fid = FID(
            feature_extractor, Path(cfg.dataset_path) / Path("train/stats")
        )
        self.test_reco_psnr = PSNR()

        # hparams

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        """
        A single training step
        """
        # self.log("step", self.trainer.global_step // 2, logger=True, on_step=True)
        real_images, segmentation_maps = batch

        opt_g, opt_d = self.optimizers()

        gen_loss, generated_images, gen_metrics_dict = self.generator_step(
            real_images, segmentation_maps, log_metrics=True, step_type="train"
        )
        for metric_elem in gen_metrics_dict:
            self.log(
                f"train/{metric_elem['name']}",
                metric_elem["value"],
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        self.log(
            "train/reco_fid",
            self.train_reco_fid,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/reco_psnr",
            self.train_reco_psnr,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/swap_fid",
            self.train_swap_fid,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        opt_g.zero_grad()
        self.manual_backward(gen_loss)

        if self.cfg.gradient_clip_val > 0:
            clip_grad_norm_(
                itertools.chain(self.generator.parameters(), self.encoder.parameters()),
                max_norm=self.cfg.gradient_clip_val,
            )
        opt_g.step()
        # yield gen_loss

        disc_loss, disc_metrics_dict = self.discriminator_step(
            real_images, generated_images, segmentation_maps
        )
        for metric_elem in disc_metrics_dict:
            self.log(
                f"train/{metric_elem['name']}",
                metric_elem["value"],
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        opt_d.zero_grad()
        self.manual_backward(disc_loss)
        if self.cfg.gradient_clip_val > 0:
            clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.cfg.gradient_clip_val,
            )
        opt_d.step()
        ##### Logging Metrics #####

    def validation_step(self, batch, batch_idx):
        """
        A single validation step
        """
        real_images, segmentation_maps = batch

        _, generated_images, gen_metrics_dict = self.generator_step(
            real_images, segmentation_maps, log_metrics=True, step_type="val"
        )

        for metric_elem in gen_metrics_dict:
            self.log(
                f"val/{metric_elem['name']}",
                metric_elem["value"],
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self.log(
            "val/reco_fid",
            self.val_reco_fid,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/reco_psnr",
            self.val_reco_psnr,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/swap_fid",
            self.val_swap_fid,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        _, disc_metrics_dict = self.discriminator_step(
            real_images, generated_images, segmentation_maps
        )
        for metric_elem in disc_metrics_dict:
            self.log(
                f"val/{metric_elem['name']}",
                metric_elem["value"],
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def test_step(self, batch, batch_idx):
        """
        A single validation step
        """
        real_images, segmentation_maps = batch

        _, _, _ = self.generator_step(
            real_images, segmentation_maps, log_metrics=True, step_type="test"
        )

        self.log(
            "test/reco_fid",
            self.test_reco_fid,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test/reco_psnr",
            self.test_reco_psnr,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test/swap_fid",
            self.test_swap_fid,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """
        Optimizers configuration
        """
        gen_opt_params = [
            {"params": self.generator.parameters()},
            {"params": self.encoder.parameters()},
        ]
        g_opt = instantiate(
            self.cfg.optim.gen_optim, params=itertools.chain(gen_opt_params)
        )
        d_opt = instantiate(
            self.cfg.optim.disc_optim,
            params=itertools.chain(self.discriminator.parameters()),
        )
        return [g_opt, d_opt], []

    def generator_step(
        self, real_images, segmentation_maps, log_metrics=False, step_type="train"
    ):
        batch_size = real_images.shape[0]

        #### Train

        ##### Reconstruction task
        if self.cfg.losses.gan_loss_on_swaps:
            permutation = torch.cat(
                [
                    torch.LongTensor([i for i in range(batch_size - batch_size // 2)]),
                    torch.randperm(batch_size // 2) + batch_size // 2,
                ]
            )
            reco_idx = batch_size - batch_size // 2
            swap_idx = reco_idx
            styles_codes_reco, content_code_reco = self.encoder(
                real_images[:reco_idx], segmentation_maps[:reco_idx]
            )
            with torch.no_grad():
                styles_codes_swap, content_code_swap = self.encoder(
                    real_images[reco_idx:], segmentation_maps[reco_idx:]
                )
            styles_codes = torch.cat([styles_codes_reco, styles_codes_swap], dim=0)
            content_code = torch.cat([content_code_reco, content_code_swap], dim=0)
            generated_images = self.generator(
                segmentation_maps, styles_codes[permutation], content_code
            )
            reco_images = generated_images[:reco_idx]
            swap_images = generated_images[reco_idx:]

        else:
            styles_codes, content_code = self.encoder(real_images, segmentation_maps)
            reco_images = self.generator(segmentation_maps, styles_codes, content_code)
            generated_images = reco_images
            reco_idx = batch_size
            swap_idx = 0
            with torch.no_grad():
                permutation = torch.randperm(batch_size)
                if self.cfg.losses.lambda_kld > 0 and self.cfg.encoder.use_vae:
                    permuted_style_codes = [
                        styles_codes[i][permutation] for i in range(len(styles_codes))
                    ]
                else:
                    permuted_style_codes = styles_codes[permutation]

                swap_images = self.generator(
                    segmentation_maps, permuted_style_codes, content_code
                )

        if log_metrics and step_type == "train":
            self.train_swap_fid(swap_images)

            self.train_reco_fid(reco_images)

            self.train_reco_psnr(reco_images, real_images[:reco_idx])
        if log_metrics and step_type == "val":

            self.val_swap_fid(swap_images)

            self.val_reco_fid(reco_images)

            self.val_reco_psnr(reco_images, real_images[:reco_idx])
        if log_metrics and step_type == "test":

            self.test_swap_fid(swap_images)

            self.test_reco_fid(reco_images)

            self.test_reco_psnr(reco_images, real_images[:reco_idx])

        ## Generator losses

        ### Compute generator losses

        real_disc, generated_disc = self.discriminate(
            real_images,
            generated_images,
            segmentation_maps,
        )

        gen_loss, gen_loss_metrics = self.loss_computer_gen(
            real_images,
            generated_images,
            segmentation_maps,
            real_disc,
            generated_disc,
            styles_codes,
            last_layer = self.generator.get_last_layer(),
        )
        return gen_loss, generated_images, gen_loss_metrics

    def discriminator_step(self, real_images, generated_images, segmentation_maps):
        real_images.requires_grad = True
        generated_images = generated_images.detach()
        ## Discriminator losses
        real_disc, generated_disc = self.discriminate(
            real_images,
            generated_images,
            segmentation_maps,
        )
        disc_loss, disc_loss_metrics = self.loss_computer_disc(
            real_images,
            generated_images,
            segmentation_maps,
            real_disc,
            generated_disc,
        )
        mean_real_discs_output = 0
        mean_generated_discs_output = 0
        for num_disc in range(len(real_disc)):
            mean_real_discs_output += real_disc[num_disc][-1].detach().mean()
            mean_generated_discs_output += generated_disc[num_disc][-1].detach().mean()
        mean_real_discs_output /= len(real_disc)
        mean_generated_discs_output /= len(generated_disc)
        disc_loss_metrics.append(
            {"name": "mean_real_discs_output", "value": mean_real_discs_output}
        )
        disc_loss_metrics.append(
            {
                "name": "mean_generated_discs_output",
                "value": mean_generated_discs_output,
            }
        )
        return disc_loss, disc_loss_metrics

    def encode(
        self,
        images: TensorType["batch_size", "num_channels", "height", "width"],
        segmentation_maps: TensorType[
            "batch_size", "num_segmap_labels", "height", "width"
        ],
    ) -> Tuple[
        TensorType["batch_size", "num_segmap_labels", "style_dim"],
        TensorType[
            "batch_size",
            "output_dim",
            "output_fmap_heigth",
            "output_fmap_width",
        ],
    ]:
        """

        Encode the given image and retrieve the style codes

        """
        styles_codes, content_code = self.encoder(images, segmentation_maps)
        return styles_codes, content_code

    def generate(
        self,
        segmentation_maps: TensorType[
            "batch_size", "num_segmap_labels", "height", "width"
        ],
        styles_codes: TensorType["batch_size", "num_segmap_labels", "style_dim"],
        content_code: TensorType["batch_size", "content_dim"],
    ) -> TensorType["batch_size", "num_channels", "height", "width"]:

        """

        Generate an image given a segmentation map and style-codes

        """
        generated_images = self.generator(segmentation_maps, styles_codes, content_code)
        return generated_images

    def encode_and_generate(
        self,
        images: TensorType["batch_size", "num_channels", "height", "width"],
        segmentation_maps: TensorType[
            "batch_size", "num_segmap_labels", "height", "width"
        ],
    ) -> TensorType["batch_size", "num_channels", "height", "width"]:
        """

        Given a style image encode it and reconstruct the input

        """
        styles_codes, content_code = self.encoder(images, segmentation_maps)
        generated_images = self.generator(segmentation_maps, styles_codes, content_code)
        return generated_images

    def encode_swap_and_generate(
        self,
        images: TensorType["batch_size", "num_channels", "height", "width"],
        segmentation_maps: TensorType[
            "batch_size", "num_segmap_labels", "height", "width"
        ],
        permutation: TensorType["batch_size"],
    ) -> TensorType["batch_size", "num_channels", "height", "width"]:
        """

        Given a style image encode it swap the codes and masks and generate a novel image

        """

        with torch.no_grad():
            styles_codes, content_code = self.encoder(images, segmentation_maps)
        if self.cfg.losses.lambda_kld > 0 and self.cfg.encoder.use_vae:
            permuted_style_codes = [
                styles_codes[0][permutation],
                styles_codes[1][permutation],
            ]
        else:
            permuted_style_codes = styles_codes[permutation]

        generated_images = self.generator(
            segmentation_maps, permuted_style_codes, content_code
        )
        return generated_images

    def discriminate(
        self,
        real_images: TensorType["batch_size", "num_channels", "height", "width"],
        generated_images: TensorType["batch_size", "num_channels", "height", "width"],
        segmentation_maps: TensorType["batch_size", "num_segmap_labels", "style_dim"],
    ) -> Tuple[
        Union[
            Sequence[
                Sequence[TensorType["batch_size", 1, "output_height", "output_width"]]
            ],
            Sequence[TensorType["batch_size", 1, "output_height", "output_width"]],
        ],
        Union[
            Sequence[
                Sequence[TensorType["batch_size", 1, "output_height", "output_width"]]
            ],
            Sequence[TensorType["batch_size", 1, "output_height", "output_width"]],
        ],
    ]:
        """

        Apply the discrimination process

        """

        batch_size = real_images.shape[0]

        real_images = self.disc_augment(real_images)

        generated_images = self.disc_augment(generated_images)

        real_images.requires_grad = True

        real_and_generated = torch.cat([real_images, generated_images], dim=0)
        real_and_generate_masks = torch.cat(
            [segmentation_maps, segmentation_maps], dim=0
        )
        real_and_generated = torch.cat(
            [real_and_generated, real_and_generate_masks], dim=1
        )

        discriminated_real_and_generated = self.discriminator(
            real_and_generated  # , real_and_generate_masks
        )
        if type(discriminated_real_and_generated) == list:
            real_disc = []
            generated_disc = []
            real_and_generated.requires_grad_(True)
            for disc_out in discriminated_real_and_generated:
                if self.cfg.discriminator.apply_grad_norm and self.training:
                    grad = torch.autograd.grad(
                        disc_out[-1],
                        [real_and_generated],
                        torch.ones_like(disc_out[-1]),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
                    grad_norm = grad_norm.view(
                        -1, *[1 for _ in range(len(disc_out[-1].shape) - 1)]
                    )
                    disc_out[-1] = disc_out[-1] / (grad_norm + torch.abs(disc_out[-1]))
                real_disc.append([feature_map[:batch_size] for feature_map in disc_out])
                generated_disc.append(
                    [feature_map[batch_size:] for feature_map in disc_out]
                )
        else:
            real_disc = discriminated_real_and_generated[:batch_size]
            generated_disc = discriminated_real_and_generated[batch_size:]
        return real_disc, generated_disc
