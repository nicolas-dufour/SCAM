import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchtyping import TensorType
from einops import repeat, rearrange
from models.utils_blocks.equallr import EqualLinear
from functools import partial


class MaskedAttention(nn.Module):
    """
    Masked Attention Module. Can be used for both self attention and cross attention

    to_dim: int,
        The dimension of the query token dim
    from_dim: int,
        The Dimension of the key token dim
    num_heads: int,
        Number of attention heads
    """

    def __init__(
        self,
        to_dim: int,
        from_dim: int,
        latent_dim: int,
        num_heads: int,
        use_equalized_lr: bool = False,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        self.latent_dim_k = latent_dim // num_heads
        self.num_heads = num_heads
        LinearLayer = (
            partial(EqualLinear, lr_mul=lr_mul) if use_equalized_lr else nn.Linear
        )

        # Mappings Query, Key Values
        self.q_linear = LinearLayer(to_dim, latent_dim)
        self.v_linear = LinearLayer(from_dim, latent_dim)
        self.k_linear = LinearLayer(from_dim, latent_dim)

        # Final output mapping
        self.out = nn.Sequential(LinearLayer(latent_dim, to_dim))

    def forward(
        self,
        X_to: TensorType["batch_size", "num_to_tokens", "to_dim"],
        X_from: TensorType["batch_size", "num_from_tokens", "from_dim"],
        mask_from: TensorType["batch_size", "num_to_tokens", "num_from_tokens"] = None,
        return_attention: bool = False,
    ):
        Q = rearrange(self.q_linear(X_to), " b t (h k) -> b h t k ", h=self.num_heads)
        K = rearrange(self.v_linear(X_from), " b t (h k) -> b h t k ", h=self.num_heads)
        V = rearrange(self.k_linear(X_from), " b t (h k) -> b h t k ", h=self.num_heads)

        attn = torch.einsum("bhtk,bhfk->bhtf", [Q, K]) / math.sqrt(self.latent_dim_k)

        if mask_from is not None:
            mask_from = mask_from.unsqueeze(1)
            attn = attn.masked_fill(mask_from == 0, -1e4)

        attn = F.softmax(attn, dim=-1)

        output = torch.einsum("bhtf,bhfk->bhtk", [attn, V])
        output = rearrange(output, "b h t k -> b t (h k)")
        output = self.out(output)

        if return_attention:
            return output, attn
        else:
            return output


class MaskedTransformer(nn.Module):
    def __init__(
        self,
        to_dim,
        to_len,
        from_dim,
        latent_dim,
        num_heads,
        use_equalized_lr=False,
        lr_mul=1,
    ):
        super().__init__()
        self.attention = MaskedAttention(
            to_dim,
            from_dim,
            latent_dim,
            num_heads,
            use_equalized_lr=use_equalized_lr,
            lr_mul=lr_mul,
        )

        LinearLayer = (
            partial(EqualLinear, lr_mul=lr_mul) if use_equalized_lr else nn.Linear
        )
        self.ln_1 = nn.LayerNorm((to_len, to_dim))
        self.fc = nn.Sequential(
            LinearLayer(to_dim, to_dim),
            nn.LeakyReLU(2e-1),
            LinearLayer(to_dim, to_dim),
            nn.LeakyReLU(2e-1),
        )
        self.ln_2 = nn.LayerNorm((to_len, to_dim))

    def forward(self, X_to, X_from, mask=None, return_attention=False):
        if return_attention:
            X_to_out, attn = self.attention(X_to, X_from, mask, return_attention)
        else:
            X_to_out = self.attention(X_to, X_from, mask, return_attention)
        X_to = self.ln_1(X_to_out + X_to)
        X_to_out = self.fc(X_to)
        X_to = self.ln_2(X_to_out + X_to)
        if return_attention:
            return X_to, attn
        else:
            return X_to


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Encoder
    Adapted from https://github.com/lucidrains/transganformer

    dim: int,
        Tokens dimension
    """

    def __init__(self, dim: int, emb_type: str = "add"):
        super().__init__()
        dim //= 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.emb_type = emb_type
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: TensorType["batch_size", "num_token", "dim"]):
        h = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device).type_as(
            self.inv_freq
        )
        w = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device).type_as(
            self.inv_freq
        )
        sinu_inp_h = torch.einsum("i , j -> i j", h, self.inv_freq)
        sinu_inp_w = torch.einsum("i , j -> i j", w, self.inv_freq)
        sinu_inp_h = repeat(sinu_inp_h, "h c -> () c h w", w=x.shape[-1])
        sinu_inp_w = repeat(sinu_inp_w, "w c -> () c h w", h=x.shape[-2])
        sinu_inp = torch.cat((sinu_inp_w, sinu_inp_h), dim=1)
        emb = torch.cat((sinu_inp.sin(), sinu_inp.cos()), dim=1)
        if self.emb_type == "add":
            x_emb = x + emb
        elif self.emb_type == "concat":
            emb = repeat(emb, "1 ... -> b ...", b=x.shape[0])
            x_emb = torch.cat([x, emb], dim=1)
        return x_emb


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding

    Parameters:
    -----------
        num_tokens_max: int,
            Max size of the sequence lenght
        dim_tokens: int,
            Size of the embedding dim
    """

    def __init__(self, num_tokens_max: int, dim_tokens: int):
        super().__init__()
        self.num_tokens_max = num_tokens_max
        self.dim_tokens = dim_tokens
        self.weights = nn.Parameter(torch.Tensor(num_tokens_max, dim_tokens))

    def forward(self, x: TensorType["batch_size", "num_tokens", "dim_tokens"]):
        _, num_tokens = x.shape[:2]
        assert num_tokens <= self.num_tokens_max
        return x + self.weights[:num_tokens].view(1, num_tokens, self.dim_tokens)
