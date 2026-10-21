"""Decode separator outputs to images with the Stage 1 generator: attribute swaps and salient-code interpolations."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from diffca.stage2.evaluate import collect_pair_names

DEFAULT_SAMPLING_STEPS = 20
IMAGE_SIZE = 256


def generate_from_tokens(diffusion_model, vae, tokens, starting_noise, steps, scaling_factor):
    """Euler integration of the flow from `starting_noise`, conditioned on `tokens`, then VAE decoding. In [-1, 1]."""
    dt = 1.0 / steps
    x_t = starting_noise.clone()
    bsz = tokens.shape[0]
    for i in range(steps):
        t_tensor = torch.full((bsz,), i / steps, device=x_t.device, dtype=x_t.dtype)
        x_t = x_t + diffusion_model(x_t, t_tensor, tokens) * dt
    decode_latents = (x_t / scaling_factor).to(dtype=next(vae.parameters()).dtype)
    return vae.decode(decode_latents).sample.clamp(-1, 1)


class _ImageGenerator:
    """Tokens -> images in [0, 1] at 256x256, with fresh noise unless `noise` is given."""

    def __init__(self, diffusion_model, vae, config, device, scaling_factor):
        self.diffusion_model = diffusion_model
        self.vae = vae
        self.device = device
        self.scaling_factor = scaling_factor
        self.steps = config.get("sampling", {}).get("num_inference_steps", DEFAULT_SAMPLING_STEPS)
        self.dtype = next(diffusion_model.parameters()).dtype
        self.in_channels = config["model"]["unet_config"]["in_channels"]
        self.sample_size = config["model"]["unet_config"]["sample_size"]

    def noise(self, batch_size):
        return torch.randn((batch_size, self.in_channels, self.sample_size, self.sample_size),
                           device=self.device, dtype=self.dtype)

    def __call__(self, tokens, noise=None):
        if noise is None:
            noise = self.noise(tokens.shape[0])
        images = generate_from_tokens(self.diffusion_model, self.vae, tokens.to(device=self.device, dtype=self.dtype),
                                      noise, self.steps, self.scaling_factor)
        images = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        return (images + 1) / 2


def generate_all_swapped_images(model, loaders, diffusion_model, vae, config, device, scaling_factor, output_dir,
                                use_ema=True):
    """For every evaluation pair (x, y) of every pair of classes, save to `output_dir/<i>_<j>/`:
    recon_cs_x/x_y.png, recon_cs_y/y_x.png (reconstructions from the separator), swap_x2y/x_y.png (common of x,
    salient of y) and swap_y2x/y_x.png."""
    if diffusion_model is None or vae is None:
        print("Missing required components for swapped image generation.")
        return
    os.makedirs(output_dir, exist_ok=True)
    generate = _ImageGenerator(diffusion_model, vae, config, device, scaling_factor)
    decode = model.model.decoder

    model.eval()
    with torch.no_grad():
        for pair_key, pair_loader in loaders.get("pairs", {}).items():
            dirs = {name: os.path.join(output_dir, pair_key, name) for name in ("recon_cs_x", "recon_cs_y", "swap_x2y", "swap_y2x")}
            for path in dirs.values():
                os.makedirs(path, exist_ok=True)

            batch_indices_iter = iter(pair_loader.batch_sampler)
            for (X_a, _, _), (X_b, _, _) in tqdm(pair_loader, desc=f"Generating swapped images ({pair_key})"):
                pair_entries = collect_pair_names(pair_loader.dataset, next(batch_indices_iter, None))
                if not pair_entries:
                    continue
                out_a = model(X_a.to(device), use_ema=use_ema)["standard_outputs"]
                out_b = model(X_b.to(device), use_ema=use_ema)["standard_outputs"]
                cls_a, context_a = out_a["cls"].unsqueeze(1), out_a["content_tokens"]
                cls_b, context_b = out_b["cls"].unsqueeze(1), out_b["content_tokens"]

                images = {
                    "recon_cs_x": generate(decode(context_a, cls_a, use_ema=use_ema)),
                    "recon_cs_y": generate(decode(context_b, cls_b, use_ema=use_ema)),
                    "swap_x2y": generate(decode(context_a, cls_b, use_ema=use_ema)),
                    "swap_y2x": generate(decode(context_b, cls_a, use_ema=use_ema)),
                }
                for i, (_, name_x, name_y) in enumerate(pair_entries):
                    for key, imgs in images.items():
                        # Files are named <sample giving the common tokens>_<other sample>.png
                        x_first = key in ("recon_cs_x", "swap_x2y")
                        filename = f"{name_x}_{name_y}.png" if x_first else f"{name_y}_{name_x}.png"
                        save_image(imgs[i], os.path.join(dirs[key], filename))


def generate_interpolation_grid(model, loaders, diffusion_model, vae, config, device, scaling_factor, output_dir,
                                use_ema=True, max_grids=3, negative_class=0, positive_class=2, num_negatives=3,
                                num_steps=5):
    """Grids decoding the common tokens of `num_negatives` samples of `negative_class` with the salient code
    (1 - a) h_neg + a h_pos, a = 1/num_steps ... 1, for a sample of `positive_class`. Reseeds the RNGs (42)."""
    if diffusion_model is None or vae is None or config is None:
        print("Missing required components for interpolation generation.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    generate = _ImageGenerator(diffusion_model, vae, config, device, scaling_factor)
    alphas = [(i + 1) / num_steps for i in range(num_steps)]

    dataset = (loaders["full"] if isinstance(loaders, dict) else loaders).dataset
    neg_indices, pos_indices = [], []
    for idx in range(len(dataset)):
        try:
            _, _, sublabel = dataset[idx]
        except ValueError:
            continue
        if int(sublabel) == negative_class:
            neg_indices.append(idx)
        if int(sublabel) == positive_class:
            pos_indices.append(idx)
    if negative_class == positive_class and neg_indices and not pos_indices:
        pos_indices = list(neg_indices)
    if not neg_indices or not pos_indices:
        print("Not enough negatives or positives for interpolation generation.")
        return []

    saved_paths = []
    model.eval()
    with torch.no_grad():
        print("Interpolating ... ")
        chosen_pos_indices = np.random.choice(pos_indices, max_grids, replace=len(pos_indices) < max_grids).tolist()
        for grid_idx, pos_idx in enumerate(chosen_pos_indices):
            chosen_neg_indices = np.random.choice(neg_indices, num_negatives, replace=len(neg_indices) < num_negatives).tolist()
            X_neg = torch.stack([dataset[idx][0] for idx in chosen_neg_indices], dim=0).to(device)
            out_neg = model(X_neg, use_ema=use_ema)["standard_outputs"]
            X_pos = dataset[pos_idx][0].unsqueeze(0).to(device)
            cls_pos = model(X_pos, use_ema=use_ema)["standard_outputs"]["cls"]

            save_image(generate(X_neg), os.path.join(output_dir, f"originals_neg_{grid_idx}.png"), nrow=num_negatives)
            save_image(generate(X_pos), os.path.join(output_dir, f"original_pos_{grid_idx}.png"))

            noise = generate.noise(num_negatives)
            # Decoded with the online decoder (identical to the EMA one when ema_decay = 0, as in the paper configs).
            images_by_alpha = [
                generate(model.decoder(out_neg["content_tokens"], (1.0 - alpha) * out_neg["cls"] + alpha * cls_pos), noise=noise)
                for alpha in alphas
            ]
            grid = torch.cat([torch.cat([images[row:row + 1] for images in images_by_alpha], dim=0)
                              for row in range(num_negatives)], dim=0)
            save_path = os.path.join(output_dir, f"interpolation_grid_{grid_idx}.png")
            save_image(grid, save_path, nrow=num_steps)
            saved_paths.append(save_path)
    return saved_paths
