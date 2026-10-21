"""Encode every train/test image into its conditioning tokens Z (one `<image>.pt` per image), the input of Stage 2.

    python -m diffca.stage1.extract_features --config configs/stage1/ffhq_cqc.yaml \
        --checkpoint <stage1_output>/checkpoint-300000 --output_dir <stage1_output>/extracted_features/step-300000

Extra `--section.key value` arguments override config entries (e.g. `--data.batch_size 32`).
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffca.stage1.build import load_trained_model, safe_encode
from diffca.stage1.dataset import HybridDataset
from diffca.stage1.models import AEModel
from diffca.utils import load_config, seed_everything


def parse_overrides(raw_args: list[str]) -> dict[str, object]:
    """`--a.b value` / `--a.b=value` pairs -> {"a.b": parsed value}."""
    overrides: dict[str, object] = {}
    idx = 0
    while idx < len(raw_args):
        token = raw_args[idx]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument '{token}'. Overrides must use --key value.")
        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
        else:
            if idx + 1 >= len(raw_args):
                raise ValueError(f"Missing value for override '{token}'.")
            value = raw_args[idx + 1]
            idx += 1
        overrides[key] = yaml.safe_load(value)
        idx += 1
    return overrides


def apply_overrides(config: dict, overrides: dict[str, object]) -> None:
    for key, value in overrides.items():
        path = key.split(".")
        cursor = config
        for part in path[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(f"Override path '{key}' does not exist in config.")
            cursor = cursor[part]
        if path[-1] not in cursor:
            raise KeyError(f"Override key '{key}' does not exist in config.")
        cursor[path[-1]] = value


def prepare_conditions(batch: dict, model: AEModel, weight_dtype: torch.dtype):
    """(clean latents, backbone tokens, clean images) for a HybridDataset batch, cached or online."""
    device = next(model.parameters()).device
    encoder_dtype = next(model.encoder.parameters()).dtype
    if batch.get("is_cached", [False])[0]:
        clean_latents = batch["latent"].to(device=device, dtype=encoder_dtype)
        dino_embeds = batch["dino"].to(device=device, dtype=encoder_dtype)
        clean_images = batch.get("pixel_values_clean")
        if clean_images is not None:
            clean_images = clean_images.to(device=device, dtype=weight_dtype)
        return clean_latents, dino_embeds, clean_images

    images_clean = batch["pixel_values_clean"].to(device=device, dtype=weight_dtype)
    images_dino = batch["pixel_values_dino"].to(device=device, dtype=weight_dtype)
    clean_latents = safe_encode(model.vae, images_clean)
    clean_latents = clean_latents * model.vae.config.scaling_factor
    clean_latents = clean_latents.to(dtype=encoder_dtype)

    if model.backbone is None:
        raise ValueError("Backbone required to compute DINO embeddings but is missing.")
    dino_embeds = model.backbone(images_dino.to(model.backbone.dtype)).last_hidden_state
    dino_embeds = dino_embeds.to(dtype=encoder_dtype)
    return clean_latents, dino_embeds, images_clean


def extract_split(split_name, dataset, model, output_dir, weight_dtype, batch_size, num_workers):
    split_out_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_out_dir, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    offset = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Extracting {split_name}"):
            clean_latents, dino_embeds, clean_images = prepare_conditions(batch, model, weight_dtype)
            context = model.encoder.encode(backbone_embeds=dino_embeds, latents=clean_latents, images=clean_images)
            bsz = context.shape[0]
            for i, image_path in enumerate(dataset.image_paths[offset: offset + bsz]):
                torch.save(context[i].detach().cpu(), os.path.join(split_out_dir, f"{Path(image_path).stem}.pt"))
            offset += bsz


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the Stage 1 conditioning tokens of every image.")
    parser.add_argument("--config", type=str, required=True, help="Path to the Stage 1 config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a Stage 1 checkpoint-<step> directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write the features.")
    args, unknown_args = parser.parse_known_args()

    config = load_config(args.config)
    if unknown_args:
        print("Applying command-line overrides to config...")
        apply_overrides(config, parse_overrides(unknown_args))

    if config["augmentations"]["enable"] and config["data"]["use_cached_features"]:
        raise ValueError("Augmentations enabled but use_cached_features is TRUE.")

    seed_everything(config["seed"])
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "no"))
    model, _, scaling_factor, weight_dtype = load_trained_model(config, args.checkpoint, accelerator)
    config["vae"]["scaling_factor"] = float(scaling_factor)

    os.makedirs(args.output_dir, exist_ok=True)
    # Note: the train split goes through the training augmentations enabled in the config.
    datasets = {"train": HybridDataset(config, is_train=True), "test": HybridDataset(config, is_train=False)}
    for split, dataset in datasets.items():
        extract_split(split, dataset, model, args.output_dir, weight_dtype,
                      batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"])


if __name__ == "__main__":
    main()
