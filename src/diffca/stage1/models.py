"""Stage 1 models: the image encoders that produce conditioning tokens, and the flow-matching U-Net.

Parameter names and registration order are those of the original code, so existing checkpoints
(including the positional EMA shadow parameters) load unchanged.
"""

import math
from abc import ABC, abstractmethod

import torch
from diffusers import UNet2DConditionModel
from torch import nn


# ---------------------------------------------------------------------------
# Perceiver-style cross-attention (used by the Cross-Query encoder)
# ---------------------------------------------------------------------------

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    """(batch, length, width) -> (batch, heads, length, width // heads)."""
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    return x.reshape(bs, heads, length, -1)


class PerceiverAttention(nn.Module):
    """Learned queries attend to the image features concatenated with themselves."""

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """x: image features (b, n1, D); latents: query tokens (b, n2, D)."""
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, length, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Scale q and k separately: more stable in fp16 than dividing afterwards.
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, length, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Generator and wrappers
# ---------------------------------------------------------------------------

class DiffusionGenerator(nn.Module):
    """Conditional U-Net predicting the flow-matching velocity. `t_cont` is in [0, 1]."""

    def __init__(self, unet_config: dict):
        super().__init__()
        self.unet = UNet2DConditionModel(**unet_config)

    def forward(self, x_noisy, t_cont, cond_tokens):
        return self.unet(
            sample=x_noisy,
            timestep=t_cont * 1000,
            encoder_hidden_states=cond_tokens,
            return_dict=False,
        )[0]


class TrainableWrapper(nn.Module):
    """The two trained parts (generator + encoder), wrapped together for Accelerate and EMA."""

    def __init__(self, diffusion_model, encoder):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.encoder = encoder

    def forward(self, x_noisy, dino_embeds, t, clean_latents, clean_images):
        context = self.encoder.encode(backbone_embeds=dino_embeds, latents=clean_latents, images=clean_images)
        return self.diffusion_model(x_noisy, t, context)


class AEModel(nn.Module):
    """Holds references to every Stage 1 component (generator, encoder, VAE, DINO backbone).

    During training and evaluation, `denoise_step` goes through the Accelerate-prepared
    `TrainableWrapper` set with `set_prepared_model`.
    """

    def __init__(self, diffusion_model: nn.Module, encoder: nn.Module, vae: nn.Module = None, backbone: nn.Module = None):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.encoder = encoder
        self.vae = vae
        self.backbone = backbone
        self.prepared_model = None

    def set_prepared_model(self, model):
        self.prepared_model = model

    def denoise_step(self, x_noisy, dino_embeds, t, clean_latents, clean_images):
        if self.prepared_model:
            return self.prepared_model(x_noisy, dino_embeds, t, clean_latents, clean_images)
        context = self.encoder.encode(backbone_embeds=dino_embeds, latents=clean_latents, images=clean_images)
        return self.diffusion_model(x_noisy, t, context)


# ---------------------------------------------------------------------------
# Encoders: image -> conditioning tokens Z (K x d)
# ---------------------------------------------------------------------------

class Encoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, backbone_embeds: torch.Tensor | None = None, latents: torch.Tensor | None = None,
               images: torch.Tensor | None = None) -> torch.Tensor:
        """Map backbone features and/or VAE latents to conditioning tokens."""


class CrossQueryEncoder(Encoder):
    """CQ: `n_tokens` learned queries cross-attend to the DINOv3 tokens (paper: Cross-Query extractor)."""

    def __init__(self, dim=1024, depth=4, dim_head=64, heads=12, n_tokens=32, embedding_dim=768, output_dim=128,
                 ff_mult=4, **kwargs):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_tokens, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )

    def encode(self, backbone_embeds: torch.Tensor, latents: torch.Tensor | None = None,
               images: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(backbone_embeds)

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class ColorEncoder(nn.Module):
    """Small CNN mapping the VAE latents to one global color token."""

    def __init__(self, num_input_channels: int = 3, output_embedding_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossQueryColorEncoder(CrossQueryEncoder):
    """CQC (the paper's encoder): `n_tokens - 1` cross-query tokens plus one color token, which comes first."""

    def __init__(self, num_input_channels=3, output_dim=128, **kwargs):
        kwargs["n_tokens"] = kwargs["n_tokens"] - 1  # one slot is taken by the color token
        super().__init__(output_dim=output_dim, **kwargs)
        self.color_encoder = ColorEncoder(num_input_channels, output_dim)

    def encode(self, backbone_embeds: torch.Tensor, latents: torch.Tensor,
               images: torch.Tensor | None = None) -> torch.Tensor:
        token_embeds = super().forward(backbone_embeds)
        color_embeds = self.color_encoder(latents)
        if color_embeds.ndim == 2:
            color_embeds = color_embeds.unsqueeze(1)
        return torch.cat([color_embeds, token_embeds], dim=1)


class DinoEncoder(Encoder):
    """Baseline ("DINOv3" / "CLIP" in the paper): a linear projection of every backbone token."""

    def __init__(self, output_dim: int, embedding_dim: int = 768, **kwargs):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, output_dim)

    def encode(self, backbone_embeds: torch.Tensor, latents: torch.Tensor | None = None,
               images: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(backbone_embeds)

    def forward(self, x):
        return self.proj(x)


class EncResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels) if bn else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(self.bn1(x)))
        h = self.conv2(self.bn2(h))
        return x + h


class EncDiffEncoder(Encoder):
    """Baseline: our re-implementation of the EncDiff encoder (CNN on the VAE latents -> scalars -> tokens)."""

    def __init__(self, d, context_dim, latent_unit, bn=True, num_channels=3, num_input_channels=None, **kwargs):
        super().__init__()
        if num_input_channels is not None:
            num_channels = num_input_channels
        self.context_dim = context_dim
        self.latent_unit = latent_unit

        # 64x64 latents -> 4x4 feature map. (No ReLU after the third BatchNorm, as in the trained models.)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, 4, 2, 1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 4, 2, 1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.Conv2d(d, d, 4, 2, 1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            EncResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            EncResBlock(d, d, bn=bn),
        )

        self.flatten_dim = d * 4 * 4
        self.to_scalars = nn.Linear(self.flatten_dim, self.latent_unit)

        # One independent MLP (1 -> 64 -> 128 -> context_dim) per scalar, as grouped 1x1 convolutions.
        self.parallel_mlps = nn.Sequential(
            nn.Conv1d(latent_unit, latent_unit * 64, kernel_size=1, groups=latent_unit),
            nn.ELU(True),
            nn.Conv1d(latent_unit * 64, latent_unit * 128, kernel_size=1, groups=latent_unit),
            nn.ELU(True),
            nn.Conv1d(latent_unit * 128, latent_unit * context_dim, kernel_size=1, groups=latent_unit),
        )

    def encode(self, backbone_embeds: torch.Tensor | None = None, latents: torch.Tensor | None = None,
               images: torch.Tensor | None = None) -> torch.Tensor:
        if latents is None:
            raise ValueError("EncDiffEncoder requires clean latents for encoding.")
        return self.forward(latents)

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # reshape: CPU convolutions may return channels-last tensors
        scalars = self.to_scalars(x)
        scalars = scalars.view(scalars.size(0), self.latent_unit, 1)
        tokens_flat = self.parallel_mlps(scalars)
        return tokens_flat.view(x.size(0), self.latent_unit, self.context_dim)


# Values of `model.image_encoder.type` in the Stage 1 configs.
ENCODERS = {
    "CrossQueryEncoder": CrossQueryEncoder,
    "CrossQueryColorEncoder": CrossQueryColorEncoder,
    "DinoEncoder": DinoEncoder,
    "EncDiffEncoder": EncDiffEncoder,
}
