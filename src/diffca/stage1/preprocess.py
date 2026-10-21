"""Pre-compute VAE latents and backbone tokens (DINOv3 or CLIP) for the cached-feature Stage 1 configs.

    python -m diffca.stage1.preprocess --config configs/preprocess/ffhq.yaml

Writes `<output_path>/<split>/ae_latents/*.npy` and `<split>/dino_features/*.npy` (or `clip_features/`, or one
`features.h5` per split with `use_h5`), all in float16, and `<output_path>/ae_config.json`: the VAE config with
`scaling_factor = 1 / std` of the training latents. The Stage 1 configs point to that file in `vae.config_path`.
"""

import argparse
import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, VQModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

from diffca.utils import load_config

METRICS_MAX_IMAGES = 1000  # SSIM/LPIPS of the VAE reconstructions are shown on the first images only


@dataclass
class PreprocessConfig:
    train_path: str = None
    test_path: str = None
    output_path: str = None
    batch_size: int = 1

    ae_type: str = "AutoencoderKL"  # or "VQModel"
    ae_path: str = "stabilityai/sd-vae-ft-mse"
    save_ae: bool = True
    display_metrics: bool = True    # show PSNR/SSIM/LPIPS of the VAE reconstructions

    save_dino: bool = True
    dino_type: str = "DINOv3"       # or "CLIP"
    dino_path: str = "facebook/dinov3-vitb16"

    use_h5: bool = False            # one .h5 file per split instead of one .npy per image and feature


def _get_processor_target_size(processor) -> int:
    """Square input size expected by an image processor."""
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
        if "height" in size and "width" in size:
            return int(size["height"])
    if isinstance(size, (int, float)):
        return int(size)
    crop = getattr(processor, "crop_size", None)
    if isinstance(crop, dict) and "height" in crop:
        return int(crop["height"])
    return 224


class WelfordOnlineStats:
    """Running mean and std over all latent values (batches merged with Chan's formula, in float32)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        x = x.to(torch.float32)
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        batch_count = x.numel()

        new_n = self.n + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / new_n
        self.M2 = self.M2 + (batch_var * batch_count) + (delta**2) * self.n * batch_count / new_n
        self.n = new_n

    @property
    def std(self):
        if self.n < 2:
            return 1.0
        return np.sqrt(self.M2 / self.n)


class ImageListDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in {".png", ".jpg", ".jpeg"}:
                    self.files.append(os.path.join(dirpath, filename))
        self.files = sorted(self.files)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        return self.transform(img), os.path.basename(path)


def save_batch(data_dict, output_dir, use_h5, h5_file_handle=None, filenames=None):
    """Save every tensor of `data_dict` as float16."""
    if use_h5:
        for key, tensor in data_dict.items():
            data_np = tensor.detach().cpu().to(torch.float16).numpy()
            if key not in h5_file_handle:
                h5_file_handle.create_dataset(key, data=data_np, maxshape=(None,) + data_np.shape[1:], chunks=True)
            else:
                dset = h5_file_handle[key]
                dset.resize((dset.shape[0] + data_np.shape[0]), axis=0)
                dset[-data_np.shape[0]:] = data_np
        if filenames:
            if "filenames" not in h5_file_handle:
                h5_file_handle.create_dataset("filenames", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
            dset = h5_file_handle["filenames"]
            dset.resize((dset.shape[0] + len(filenames)), axis=0)
            dset[-len(filenames):] = filenames
    else:
        for i, fname in enumerate(filenames):
            name_no_ext = os.path.splitext(fname)[0]
            for key, tensor in data_dict.items():
                save_subdir = os.path.join(output_dir, key)
                os.makedirs(save_subdir, exist_ok=True)
                np.save(os.path.join(save_subdir, f"{name_no_ext}.npy"), tensor[i].detach().cpu().to(torch.float16).numpy())


def backbone_tokens(imgs, dino_processor, dino_model, device):
    """All last-layer tokens of the backbone, for AE-normalized images in [-1, 1]."""
    imgs_01 = (imgs + 1) / 2.0
    with torch.no_grad():
        target = _get_processor_target_size(dino_processor)
        if imgs_01.shape[-1] != target or imgs_01.shape[-2] != target:
            imgs_rs = F.interpolate(imgs_01, size=(target, target), mode="bicubic", align_corners=False)
        else:
            imgs_rs = imgs_01
        if not hasattr(dino_processor, "norm_mean_tensor"):
            dino_processor.norm_mean_tensor = torch.tensor(dino_processor.image_mean, device=device, dtype=imgs_rs.dtype).view(1, 3, 1, 1)
            dino_processor.norm_std_tensor = torch.tensor(dino_processor.image_std, device=device, dtype=imgs_rs.dtype).view(1, 3, 1, 1)
        pixel_values = (imgs_rs - dino_processor.norm_mean_tensor) / dino_processor.norm_std_tensor
        return dino_model(pixel_values=pixel_values).last_hidden_state


def process_split(config, split_name, data_path, ae_model, dino_processor, dino_model, device, stats_calculator=None):
    print(f"\n--- Processing Split: {split_name} ---")
    split_out_dir = os.path.join(config.output_path, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    h5_handle = None
    if config.use_h5:
        h5_handle = h5py.File(os.path.join(split_out_dir, "features.h5"), "w")

    dataset = ImageListDataset(data_path)
    print(f"Processing {len(dataset)} images from {data_path}...")
    if len(dataset) == 0:
        print(f"No images found in {data_path}. Skipping split.")
        return 0.0

    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=24, shuffle=False, pin_memory=True)

    lpips_fn = None
    if config.display_metrics and split_name == "train":
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device).to(torch.bfloat16)

    running_psnr = running_ssim = running_lpips = 0.0
    count_metrics = 0
    feature_key = "clip_features" if config.dino_type.upper() == "CLIP" else "dino_features"

    pbar = tqdm(loader)
    for i, (imgs, filenames) in enumerate(pbar):
        imgs = imgs.to(device, dtype=torch.bfloat16)
        batch_save_data = {}

        with torch.no_grad():
            if config.ae_type == "VQModel":
                latents = ae_model.encode(imgs).latents
            else:
                latents = ae_model.encode(imgs).latent_dist.mode()

            if stats_calculator is not None:
                stats_calculator.update(latents)
            if config.save_ae:
                batch_save_data["ae_latents"] = latents

            if config.display_metrics or stats_calculator is not None:
                recon = ae_model.decode(latents).sample
                running_psnr += peak_signal_noise_ratio(recon.float(), imgs.float(), data_range=2.0).item()
                if config.display_metrics and (i * config.batch_size) < METRICS_MAX_IMAGES and split_name == "train":
                    running_ssim += structural_similarity_index_measure(recon.float(), imgs.float(), data_range=2.0).item()
                    running_lpips += lpips_fn(recon.clamp(-1, 1), imgs.clamp(-1, 1)).item()
                    count_metrics += 1
                pbar.set_postfix({
                    "PSNR": f"{running_psnr / (i + 1):.2f}",
                    "SSIM": f"{running_ssim / max(1, count_metrics):.3f}" if config.display_metrics else "",
                    "LPIPS": f"{running_lpips / max(1, count_metrics):.3f}" if config.display_metrics else "",
                })

        if config.save_dino and dino_model is not None:
            batch_save_data[feature_key] = backbone_tokens(imgs, dino_processor, dino_model, device)

        if batch_save_data:
            save_batch(batch_save_data, split_out_dir, config.use_h5, h5_handle, filenames)

    if h5_handle:
        h5_handle.close()
    return running_psnr / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents and backbone tokens.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = PreprocessConfig(**load_config(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing with config: {config}")

    print(f"Loading AE: {config.ae_path} (BF16)...")
    ae_cls = VQModel if config.ae_type == "VQModel" else AutoencoderKL
    ae_model = ae_cls.from_pretrained(config.ae_path, torch_dtype=torch.bfloat16).to(device).eval()

    dino_model = None
    dino_processor = None
    if config.save_dino:
        print(f"Loading encoder: {config.dino_type} @ {config.dino_path} (BF16)...")
        dino_processor = AutoImageProcessor.from_pretrained(config.dino_path)
        backbone_cls = CLIPVisionModel if config.dino_type.upper() == "CLIP" else AutoModel
        dino_model = backbone_cls.from_pretrained(config.dino_path, torch_dtype=torch.bfloat16).to(device).eval()

    if config.train_path:
        stats = WelfordOnlineStats()
        process_split(config, "train", config.train_path, ae_model, dino_processor, dino_model, device, stats_calculator=stats)

        scaling_factor = 1.0 / stats.std
        print(f"\nComputed Latent Std: {stats.std:.4f}")
        print(f"Recommended Scaling Factor: {scaling_factor:.4f}")

        ae_config_out = ae_model.config
        ae_config_out["scaling_factor"] = float(scaling_factor)
        with open(os.path.join(config.output_path, "ae_config.json"), "w") as f:
            json.dump(ae_config_out, f, indent=4)

    if config.test_path:
        process_split(config, "test", config.test_path, ae_model, dino_processor, dino_model, device, stats_calculator=None)


if __name__ == "__main__":
    main()
