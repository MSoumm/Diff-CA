"""Build the Stage 1 model from a config, and load a trained checkpoint (with its EMA weights) for inference."""

import copy
import json
import os

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, VQModel
from diffusers.training_utils import EMAModel

from diffca.stage1.models import ENCODERS, AEModel, DiffusionGenerator, TrainableWrapper
from diffca.utils import profile_module

EMA_DECAY = 0.9999
DEFAULT_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def resolve_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_vae(vae_conf: dict, require_config: bool = False):
    """Load the frozen VQ or KL autoencoder and set its latent scaling factor.

    The scaling factor comes from `vae.config_path` (the `ae_config.json` written by `preprocess.py`). Without it,
    training and evaluation (`require_config`) stop; the inference scripts use the VAE's own factor.
    """
    if "VQ" in vae_conf.get("type", ""):
        vae = VQModel.from_pretrained(vae_conf["pretrained_path"])
    else:
        vae = AutoencoderKL.from_pretrained(vae_conf["pretrained_path"])

    scaling_factor = None
    config_path = vae_conf.get("config_path")
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            scaling_factor = json.load(f).get("scaling_factor")
    if scaling_factor is None:
        if require_config:
            raise ValueError("vae.config_path must point to the ae_config.json written by preprocess.py "
                             "(it holds the latent scaling factor).")
        scaling_factor = getattr(vae.config, "scaling_factor", None)

    scaling_factor = float(scaling_factor)
    try:
        vae.config.scaling_factor = scaling_factor
    except Exception:
        pass
    try:
        vae.config["scaling_factor"] = scaling_factor
    except Exception:
        pass
    return vae, scaling_factor


def safe_encode(vae, images: torch.Tensor) -> torch.Tensor:
    """Unscaled latents: a sample of the posterior for a KL-VAE, the pre-quantization latents for a VQ model."""
    enc_out = vae.encode(images)
    if hasattr(enc_out, "latent_dist"):
        return enc_out.latent_dist.sample()
    if hasattr(enc_out, "latents"):
        return enc_out.latents
    return enc_out[0]


def build_model(config: dict, use_compile: bool, profile: bool = False, require_vae_config: bool = False):
    """Return (AEModel, TrainableWrapper, vae_scaling_factor).

    `profile` prints the parameter count and MACs of the encoder and of the generator, as during training (their
    random dummy inputs are part of the seeded RNG sequence of training).
    The generator and the encoder are wrapped in `torch.compile` when `use_compile` is set; checkpoints
    saved by the training script store their weights under `_orig_mod.` because of it.
    """
    unet_config = config["model"]["unet_config"]
    diffusion_model = DiffusionGenerator(unet_config=unet_config)

    encoder_conf = copy.deepcopy(config["model"]["image_encoder"])
    encoder_type = encoder_conf.pop("type")
    encoder = ENCODERS[encoder_type](num_input_channels=unet_config["in_channels"], **encoder_conf)

    if profile:
        if encoder_type == "EncDiffEncoder":
            dummy_input = torch.randn(1, 3, 64, 64)
        else:
            dummy_input = torch.randn(1, 261, 768)  # 261 DINOv3 tokens
        profile_module(model=copy.deepcopy(encoder), inputs=dummy_input, device="cpu")

    vae, scaling_factor = load_vae(config["vae"], require_config=require_vae_config)

    backbone = None
    backbone_path = config["model"]["image_encoder"].get("backbone_path", None)
    if backbone_path:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(backbone_path)

    model = AEModel(diffusion_model=diffusion_model, encoder=encoder, vae=vae, backbone=backbone)
    model.vae.eval()
    if model.backbone:
        model.backbone.requires_grad_(False)
        model.backbone.eval()

    if profile:
        image_encoder_conf = config["model"]["image_encoder"]
        dummy_x = torch.randn(1, unet_config["in_channels"], unet_config["sample_size"], unet_config["sample_size"])
        dummy_t = torch.tensor([0.5])
        dummy_conditioning = torch.randn(1, image_encoder_conf["n_tokens"], image_encoder_conf["output_dim"])
        profile_module(model=copy.deepcopy(model.diffusion_model), inputs=(dummy_x, dummy_t, dummy_conditioning),
                       device="cpu")

    if use_compile and hasattr(torch, "compile"):
        print("Compiling diffusion model and encoder...")
        model.diffusion_model = torch.compile(model.diffusion_model, mode="reduce-overhead")
        model.encoder = torch.compile(model.encoder, mode="reduce-overhead")

    trainable_parts = TrainableWrapper(diffusion_model=model.diffusion_model, encoder=model.encoder)
    return model, trainable_parts, scaling_factor


def load_trained_model(config: dict, checkpoint_dir: str, accelerator: Accelerator, use_compile: bool = True):
    """Build the model and load a training checkpoint (`checkpoint-<step>/`), with the EMA weights swapped in.

    Note: like any `accelerator.load_state`, this also restores the RNG states saved in the checkpoint.
    Returns (AEModel, prepared TrainableWrapper, vae_scaling_factor, weight_dtype).
    """
    weight_dtype = resolve_weight_dtype(accelerator)
    model, trainable_parts, scaling_factor = build_model(config, use_compile=use_compile)

    # The checkpoint holds an optimizer state; a dummy optimizer gives it somewhere to load.
    dummy_optimizer = torch.optim.AdamW(trainable_parts.parameters(), lr=1.0e-4)
    trainable_parts, dummy_optimizer = accelerator.prepare(trainable_parts, dummy_optimizer)
    ema_model = EMAModel(trainable_parts.parameters(), decay=EMA_DECAY)
    ema_model.to(next(trainable_parts.parameters()).device)
    accelerator.register_for_checkpointing(ema_model)
    accelerator.load_state(checkpoint_dir)
    ema_model.copy_to(trainable_parts.parameters())

    model.set_prepared_model(trainable_parts)
    model.vae.to(accelerator.device, dtype=weight_dtype)
    if model.backbone is not None:
        model.backbone.to(accelerator.device, dtype=weight_dtype)
    trainable_parts.eval()
    return model, trainable_parts, scaling_factor, weight_dtype


def find_checkpoint(save_dir: str, global_step: int | None) -> str | None:
    """`save_dir/checkpoint-<global_step>` if it exists, else the latest `checkpoint-*` folder."""
    if not save_dir:
        return None
    if global_step is not None:
        candidate = os.path.join(save_dir, f"checkpoint-{global_step}")
        if os.path.isdir(candidate):
            return candidate

    step_dirs = []
    for entry in os.listdir(save_dir):
        full_path = os.path.join(save_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
            try:
                step_dirs.append((int(entry.split("-")[-1]), full_path))
            except ValueError:
                continue
    if step_dirs:
        return max(step_dirs, key=lambda item: item[0])[1]
    return None
