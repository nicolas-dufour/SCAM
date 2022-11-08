import logging
from pytorch_lightning import Callback
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from einops import rearrange
from utils.utils import AttentionVis, get_palette, remap_image_torch


log = logging.getLogger(__name__)


class ReInitOptimAfterSanity(Callback):
    def on_train_start(self, trainer, pl_module):
        optimizers, lr_schedulers, optimizer_frequencies = trainer.init_optimizers(
            pl_module
        )
        trainer.optimizers = optimizers
        trainer.lr_schedulers = lr_schedulers
        trainer.optimizer_frequencies = optimizer_frequencies


class LogAttention(Callback):
    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 25000 == 0:
            self.log_attention(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        self.log_attention(trainer, pl_module)

    def log_attention(self, trainer, pl_module):
        if self.ready:
            logger = trainer.logger
            experiment = logger.experiment
            log_encoder = (
                pl_module.cfg.encoder._target_
                == "models.encoder.SemanticAttentionTransformerEncoder"
            )

            attnn_viz = AttentionVis(log_encoder=log_encoder)

            # Create tokens palette
            np.random.seed(42)
            num_tokens = (
                (pl_module.cfg.generator.num_labels - 1)
                * pl_module.cfg.generator.num_labels_split
                + pl_module.cfg.generator.num_labels_bg
            )
            palette_tokens = torch.from_numpy(get_palette(num_tokens)).to(
                device=pl_module.device
            )
            # Create labels palette
            np.random.seed(10)
            palette_segmask = torch.from_numpy(
                get_palette(pl_module.cfg.generator.num_labels)
            ).to(device=pl_module.device)

            val_dataloader = DataLoader(
                trainer.datamodule.test_dataset,
                batch_size=self.num_samples,
                shuffle=True,
            )
            logs = dict()

            real_images, segmentation_maps = next(iter(val_dataloader))
            real_images = real_images.to(device=pl_module.device)
            segmentation_maps = segmentation_maps.to(device=pl_module.device)

            segmask_colorized = palette_segmask[segmentation_maps.argmax(1)]

            segmask_colorized = rearrange(segmask_colorized, "b h w c -> b c h w")
            with torch.no_grad():
                outputs = attnn_viz.encode_and_generate(
                    pl_module, real_images, segmentation_maps
                )
                output_images = outputs["output"]
                del outputs["output"]
                for attn_key, attn_val in outputs.items():

                    attentions = [
                        F.interpolate(
                            attention,
                            size=(output_images.shape[2], output_images.shape[3]),
                        )
                        for attention in attn_val
                    ]

                    attentions_colorized = [
                        palette_tokens[attention.argmax(1)] for attention in attentions
                    ]

                    attentions_colorized = [
                        rearrange(attention, "b h w c -> b c h w")
                        for attention in attentions_colorized
                    ]

                    attn_viz = rearrange(
                        [
                            remap_image_torch(real_images),
                            remap_image_torch(output_images),
                            segmask_colorized,
                            *attentions_colorized,
                        ],
                        "l b c h w -> b c h (l w)",
                    ).float()
                    attn_viz = [
                        wandb.Image(
                            attn_viz[i],
                            caption="Reference; Reconstruction; Segmask; Attn matrix argmax from first layers to last",
                        )
                        for i in range(attn_viz.shape[0])
                    ]
                    logs.update({f"Images/{attn_key}": attn_viz})
                experiment.log(logs, step=trainer.global_step)


class GANImageLog(Callback):
    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 25000 == 0:
            self.log_images(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        self.log_images(trainer, pl_module)

    def log_images(self, trainer, pl_module):
        if self.ready:
            logger = trainer.logger
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_dataloader = DataLoader(
                trainer.datamodule.test_dataset,
                batch_size=self.num_samples,
                shuffle=True,
            )

            real_images, segmentation_maps = next(iter(val_dataloader))
            real_images = real_images.to(device=pl_module.device)
            segmentation_maps = segmentation_maps.to(device=pl_module.device)

            with torch.no_grad():

                styles_codes, content_code = pl_module.encoder(
                    real_images, segmentation_maps
                )
                reco_images = pl_module.generator(
                    segmentation_maps, styles_codes, content_code
                )
                reco_images = rearrange(
                    [real_images, reco_images], "l b c h w -> b c h (l w)"
                )
                segmask_reco = rearrange(
                    [segmentation_maps, segmentation_maps], "l b c h w -> b c h (l w)"
                )
                segmask_reco = segmask_reco.argmax(dim=1).cpu().numpy()
                permutation = torch.randperm(self.num_samples)
                if pl_module.cfg.losses.lambda_kld > 0:
                    permuted_style_codes = [
                        styles_codes[i][permutation] for i in range(len(styles_codes))
                    ]
                else:
                    permuted_style_codes = styles_codes[permutation]

                swap_images = pl_module.generator(
                    segmentation_maps, permuted_style_codes, content_code
                )
                segmask_swap = rearrange(
                    [
                        segmentation_maps,
                        segmentation_maps[permutation],
                        segmentation_maps,
                    ],
                    "l b c h w -> b c h (l w)",
                )
                segmask_swap = segmask_swap.argmax(dim=1).cpu().numpy()
                swap_images = rearrange(
                    [real_images, real_images[permutation], swap_images],
                    "l b c h w -> b c h (l w)",
                )
            reco_images = [
                wandb.Image(
                    reco_images[i],
                    caption="Left: Real; Right: Reconstruction",
                    masks={
                        "Segmentation": {
                            "mask_data": segmask_reco[i],
                        }
                    },
                )
                for i in range(reco_images.shape[0])
            ]
            experiment.log(
                {"Images/Reconstruction": reco_images}, step=trainer.global_step
            )

            swap_images = [
                wandb.Image(
                    swap_images[i],
                    caption="Left: Semantic ref; Middle: style ref; Right: Swap",
                    masks={
                        "Segmentation": {
                            "mask_data": segmask_swap[i],
                        }
                    },
                )
                for i in range(swap_images.shape[0])
            ]
            experiment.log({"Images/Swap": swap_images}, step=trainer.global_step)


class StopIfNaN(Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.continuous_nan_batchs = 0

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        logs = trainer.callback_metrics
        i = 0
        found_metric = False
        while i < len(self.monitor) and not found_metric:
            if self.monitor[i] in logs.keys():
                current = logs[self.monitor[i]].squeeze()
                found_metric = True
            else:
                i += 1
        if not found_metric:
            raise ValueError("Asked metric not in logs")

        if not torch.isfinite(current):
            self.continuous_nan_batchs += 1
            if self.continuous_nan_batchs >= 5:
                trainer.should_stop = True
                log.info("Training interrupted because of NaN in {self.monitor}")
        else:
            self.continuous_nan_batchs = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx) -> None:
        valid_gradients = True
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break

        if not valid_gradients:
            log.warning(
                f"detected inf or nan values in gradients. not updating model parameters"
            )
            optimizer.zero_grad()
