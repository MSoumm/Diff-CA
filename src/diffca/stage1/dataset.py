"""Stage 1 image dataset: raw images, optionally with pre-computed VAE latents and backbone features."""

import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from diffca.utils import create_transform_from_config

IMAGE_SIZE = 256
BRATS_SLICES = (85, 100)  # BraTS: only axial slices 85..100 (inclusive) are used

VAE_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
DINO_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path):
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def list_images(img_dir, brats=False):
    image_paths = []
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if brats:
                slice_index = int(fname.split(".")[-2].split("_")[-1])
                if not BRATS_SLICES[0] <= slice_index <= BRATS_SLICES[1]:
                    continue
            image_paths.append(os.path.join(root, fname))
    return sorted(image_paths)


def _load_h5_cache(cache_file, image_basenames, feature_key):
    print("Detected H5 cache.")
    with h5py.File(cache_file, "r") as f:
        latents = torch.from_numpy(f["ae_latents"][:])
        features = torch.from_numpy(f[feature_key][:])
        if "filenames" not in f:
            return latents, features
        cached_names = [n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n) for n in f["filenames"][:]]
        name_to_idx = {name: i for i, name in enumerate(cached_names)}
        try:
            indices = [name_to_idx[name] for name in image_basenames]
        except KeyError as exc:
            raise KeyError(f"Missing cached feature for image: {exc}") from exc
        return latents[indices], features[indices]


def _load_npy_folder(directory, filenames):
    """Load `directory/<name>.npy` for every name, as bfloat16, with a thread pool."""

    def _load(name):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return torch.from_numpy(np.load(path)).to(torch.bfloat16)
        return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(_load, filenames), total=len(filenames),
                            desc=f"Loading {os.path.basename(directory)}"))
    missing = sum(r is None for r in results)
    if missing:
        raise FileNotFoundError(f"Missing {missing} cached files under {directory}")
    return torch.stack(results)


class HybridDataset(Dataset):
    """Images for Stage 1.

    Two modes (`data.use_cached_features`):
      - cached: VAE latents and backbone tokens are read from the output of `preprocess.py`
        (an .h5 file, or a folder with `ae_latents/` and `dino_features/` or `clip_features/`);
      - online: they are computed during training from the returned pixels, which allows augmentations.
    """

    def __init__(self, config, is_train=True, max_samples=None):
        self.use_cached_features = config["data"].get("use_cached_features", False)
        self.cache_images_to_ram = config["data"].get("cache_images_to_ram", False)
        self.subfolder = "train" if is_train else "test"
        self.latent_scale = config.get("vae", {}).get("scaling_factor", None)
        if self.latent_scale is None:
            raise ValueError("VAE scaling factor must be specified in config under 'vae.scaling_factor'")

        dataset_path = config["data"]["dataset_path"]
        if isinstance(dataset_path, dict):
            img_dir = dataset_path.get(self.subfolder)
        else:
            img_dir = os.path.join(dataset_path, self.subfolder)

        self.image_paths = list_images(img_dir, brats="brats" in config["experiment_id"])
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        self.image_basenames = [os.path.basename(p) for p in self.image_paths]

        if self.use_cached_features:
            self._load_cached_features(config)

        self.images_ram = None
        if self.cache_images_to_ram and not self.use_cached_features:
            print(f"Loading {len(self.image_paths)} images into RAM for online processing...")
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.images_ram = list(tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths)))

        self.geometric_aug = None
        geom_cfg = config.get("augmentations", {}).get("geometric", {})
        if is_train and geom_cfg.get("enable", False) and not self.use_cached_features:
            self.geometric_aug = transforms.Compose([
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.9, 1.0), ratio=(1.0, 1.0),
                                             interpolation=Image.Resampling.BICUBIC),
            ])

        self.color_aug = None
        if is_train and config["augmentations"]["enable"]:
            self.color_aug = create_transform_from_config(config["augmentations"]["transforms"])

        self.vae_norm = VAE_NORM    # -> [-1, 1], input of the VAE
        self.dino_norm = DINO_NORM  # ImageNet statistics, input of the backbone

    def _load_cached_features(self, config):
        is_clip = config.get("data", {}).get("type", "DINO").upper() == "CLIP"
        feature_name = "clip_features" if is_clip else "dino_features"
        cache_root = config["data"]["cache_paths"][self.subfolder]
        print(f"Loading cached features from {cache_root}...")

        if os.path.isfile(cache_root) and cache_root.endswith(".h5"):
            self.latents, self.dino = _load_h5_cache(cache_root, self.image_basenames, feature_name)
        elif os.path.isdir(cache_root):
            print("Detected NPY folder cache. Loading into RAM (threaded)...")
            print("Using CLIP features." if is_clip else "Using DINO features.")
            filenames = [os.path.splitext(name)[0] + ".npy" for name in self.image_basenames]
            self.latents = _load_npy_folder(os.path.join(cache_root, "ae_latents"), filenames)
            self.dino = _load_npy_folder(os.path.join(cache_root, feature_name), filenames)
        else:
            raise ValueError(f"Cache path {cache_root} is neither an H5 file nor a valid directory.")

        if self.latent_scale != 1.0:
            self.latents = self.latents * self.latent_scale
        print(f"Loaded {len(self.latents)} samples into RAM.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.images_ram is not None:
            img_clean = self.images_ram[idx]
        else:
            img_clean = load_image(self.image_paths[idx])

        if self.geometric_aug is not None:
            img_clean = self.geometric_aug(img_clean)
        img_aug = self.color_aug(img_clean) if self.color_aug is not None else None

        if self.use_cached_features:
            # Cached latents/features bypass the augmentations.
            sample = {
                "pixel_values_clean": self.vae_norm(img_clean),
                "latent": self.latents[idx],
                "dino": self.dino[idx],
                "is_cached": True,
            }
        else:
            # The backbone sees the color-augmented image, so that color is carried by the color token.
            dino_source = img_aug if img_aug is not None else img_clean
            sample = {
                "pixel_values_clean": self.vae_norm(img_clean),
                "pixel_values_dino": self.dino_norm(dino_source),
                "is_cached": False,
            }
        if img_aug is not None:
            sample["pixel_values_aug"] = self.vae_norm(img_aug)
        return sample
