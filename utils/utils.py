import numpy as np
import torch
from einops import rearrange


def remap_image_numpy(image):

    image_numpy = ((image + 1) / 2.0) * 255.0
    return np.clip(image_numpy, 0, 255).astype(int)


def remap_image_torch(image):

    image_torch = ((image + 1) / 2.0) * 255.0
    return torch.clip(image_torch, 0, 255).type(torch.uint8)


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    palette = np.random.randint(0, 255, size=(num_cls, 3))
    return palette


class AttentionVis:
    def __init__(self, log_encoder=True):
        self.latent_to_image_gen_masks = []
        self.image_to_latent_gen_masks = []
        self.log_encoder = log_encoder
        if self.log_encoder:
            self.encoder_masks = []

    def encoder_hook_fn(self, m, i, o):
        self.encoder_masks.append(o[1])

    def latent_to_image_gen_hook_fn(self, m, i, o):
        self.latent_to_image_gen_masks.append(o[1])

    def image_to_latent_gen_hook_fn(self, m, i, o):
        self.image_to_latent_gen_masks.append(o[1])

    def encode_and_generate(self, model, images, masks):
        ### activate attention
        num_gen_blocks = model.cfg.generator.num_up_layers
        gen_mod_blocks = [
            getattr(model.generator.backbone[f"SCAM_block_{i}"], f"mod_{j}")
            for i in range(num_gen_blocks)
            for j in range(2)
        ]

        for gen_mod_block in gen_mod_blocks:
            gen_mod_block.return_attention = True
        if self.log_encoder:
            num_enc_blocks = model.cfg.encoder.num_blocks
            enc_blocks = [
                model.encoder.backbone[f"block_{i}"]["cross_attention"]
                for i in range(num_enc_blocks)
            ]
            model.encoder.return_attention = True

        handles = []
        for mod in gen_mod_blocks:
            handle = mod.latent_to_image.register_forward_hook(
                self.latent_to_image_gen_hook_fn
            )
            handles.append(handle)
            handle = mod.image_to_latent.register_forward_hook(
                self.image_to_latent_gen_hook_fn
            )
            handles.append(handle)
        if self.log_encoder:
            for enc_block in enc_blocks:
                handle = enc_block.register_forward_hook(self.encoder_hook_fn)
                handles.append(handle)

        # ### retrieve attention
        output = model.encode_and_generate(images, masks)
        gen_mod_dims = [(mod.height, mod.width) for mod in gen_mod_blocks]
        if self.log_encoder:
            enc_dims = model.encoder.image_sizes
        # ### Remove hook
        for handle in handles:
            handle.remove()
        # ### deasctivate attention output
        for gen_mod_block in gen_mod_blocks:
            gen_mod_block.return_attention = False
        if self.log_encoder:
            model.encoder.return_attention = False

        latent_to_image_gen_attention_masks = [
            rearrange(
                attention_mask,
                "b heads (h w) c-> b c (heads h) w",
                h=h,
                w=w,
            )
            for (h, w), attention_mask in zip(
                gen_mod_dims, self.latent_to_image_gen_masks
            )
        ]
        image_to_latent_gen_attention_masks = [
            rearrange(
                attention_mask,
                "b heads c (h w)-> b c (heads h) w",
                h=h,
                w=w,
            )
            for (h, w), attention_mask in zip(
                gen_mod_dims, self.image_to_latent_gen_masks
            )
        ]
        if self.log_encoder:
            encoder_attention_masks = [
                rearrange(
                    attention_mask,
                    "b heads c (h w)-> b c (heads h) w",
                    h=h,
                    w=w,
                )
                for (h, w), attention_mask in zip(enc_dims, self.encoder_masks)
            ]
            self.encoder_masks = []
        self.latent_to_image_gen_masks = []
        self.image_to_latent_gen_masks = []

        if self.log_encoder:
            return {
                "output": output,
                "latent_to_image_gen_attn": latent_to_image_gen_attention_masks,
                "image_to_latent_gen_attn": image_to_latent_gen_attention_masks,
                "encoder_attn": encoder_attention_masks,
            }
        else:
            return {
                "output": output,
                "latent_to_image_gen_attn": latent_to_image_gen_attention_masks,
                "image_to_latent_gen_attn": image_to_latent_gen_attention_masks,
            }

    # def generate(self, model, style_codes, masks):
    #     ### activate attention
    #     model.generator.backbone.SCAM_block_5.mod_1.return_attention = True
    #     output = model.generator(masks, style_codes)
    #     self.height = model.generator.backbone.SCAM_block_5.mod_1.height
    #     self.width = model.generator.backbone.SCAM_block_5.mod_1.width
    #     ### register hook
    #     handle = model.generator.backbone.SCAM_block_5.mod_1.latent_to_image.register_forward_hook(
    #         self.hook_fn
    #     )
    #     ### retrieve attention
    #     output = model.generator(masks, style_codes)
    #     ### Remove hook
    #     handle.remove()
    #     ### deasctivate attention output
    #     model.generator.backbone.SCAM_block_4.mod_1.return_attention = False
    #     return output, self.attention_masks
