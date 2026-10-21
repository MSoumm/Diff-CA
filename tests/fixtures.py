"""Tiny models and synthetic datasets shared by the equivalence tests (CPU, a few seconds to build)."""

import copy
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image

N_TOKENS, TOKEN_DIM = 32, 128

# FFHQ-Features "faceAttributes": glasses type, smile, head yaw, gender. Covers every label branch,
# including the excluded cases (swimming goggles, 5 < |yaw| <= 12).
_GLASSES = ["NoGlasses", "ReadingGlasses", "Sunglasses", "NoGlasses", "SwimmingGoggles"]
_YAWS = [0.0, -3.0, 4.5, 20.0, -25.0, 9.0, -15.0, 30.0, 2.0, -40.0, 11.0, 13.0]


def _pattern(k):
    """A fixed random token pattern: class signal that survives the per-token LayerNorm of the probes."""
    return torch.randn(N_TOKENS, TOKEN_DIM, generator=torch.Generator().manual_seed(100 + k))


def face_attributes(i):
    return [{"faceAttributes": {
        "glasses": _GLASSES[i % len(_GLASSES)],
        "smile": (i * 37 % 100) / 100.0 if i % 3 else float(i * 7 % 100),  # both [0, 1] and [0, 100] scales
        "headPose": {"yaw": _YAWS[i % len(_YAWS)], "pitch": 0.0, "roll": 0.0},
        "gender": "male" if i % 2 else "female",
    }}]


def _save_image(path, rng):
    arr = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).resize((256, 256), Image.Resampling.NEAREST).save(path)


def build_tiny_backbone(path):
    """Random 1-layer DINOv3 ViT-B/16 (261 tokens of dim 768 at 256 px) with an image processor config."""
    from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast, DINOv3ViTModel
    torch.manual_seed(0)
    config = DINOv3ViTConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=12, intermediate_size=256,
                             patch_size=16, num_register_tokens=4, image_size=256)
    DINOv3ViTModel(config).save_pretrained(path)
    DINOv3ViTImageProcessorFast(size={"height": 256, "width": 256}).save_pretrained(path)


def build_tiny_vq(path):
    from diffusers import VQModel
    torch.manual_seed(1)
    VQModel(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 3,
            up_block_types=("UpDecoderBlock2D",) * 3, block_out_channels=(8, 8, 8), layers_per_block=1,
            latent_channels=3, num_vq_embeddings=64, norm_num_groups=4, vq_embed_dim=3, sample_size=256).save_pretrained(path)


def build_tiny_kl(path):
    from diffusers import AutoencoderKL
    torch.manual_seed(2)
    AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                  up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8), layers_per_block=1,
                  latent_channels=4, norm_num_groups=4, sample_size=256).save_pretrained(path)


def _write_ae_config(vae_cls, vae_path, out_path, scaling):
    config = dict(vae_cls.load_config(vae_path))
    config["scaling_factor"] = scaling
    with open(out_path, "w") as f:
        json.dump(config, f)


def build_image_data(root):
    rng = np.random.default_rng(0)
    # FFHQ-like: images + attribute JSON files
    for split, n in (("train", 40), ("test", 24)):
        os.makedirs(f"{root}/ffhq/{split}", exist_ok=True)
        for i in range(n):
            idx = i if split == "train" else 1000 + i
            _save_image(f"{root}/ffhq/{split}/{idx:05d}.png", rng)
    os.makedirs(f"{root}/ffhq_json", exist_ok=True)
    for idx in list(range(40)) + list(range(1000, 1024)):
        if idx in (7, 1011):  # missing JSON: kept with label 0
            continue
        with open(f"{root}/ffhq_json/{idx:05d}.json", "w") as f:
            json.dump(face_attributes(idx), f)
    # AFHQ-like images (online Stage 1 path): label from the file name
    for split, n in (("train", 24), ("val", 12)):
        os.makedirs(f"{root}/afhq/{split}", exist_ok=True)
        for i in range(n):
            kind = ["cat", "dog", "wild"][i % 3]
            _save_image(f"{root}/afhq/{split}/flickr_{kind}_{split}_{i:04d}.png", rng)
    # BraTS-like slices (Stage 1 keeps slices 85..100)
    os.makedirs(f"{root}/brats/train", exist_ok=True)
    for case in range(2):
        for s in (80, 85, 90, 99, 100, 105):
            _save_image(f"{root}/brats/train/BraTS-GLI-{case:05d}-000_slice_{s:03d}.png", rng)


def build_cached_features(root):
    """VAE latents and backbone tokens (npy, float16) for the cached Stage 1 path, plus an H5 copy of the test split."""
    import h5py
    from diffusers import VQModel
    from transformers import AutoModel
    vq = VQModel.from_pretrained(f"{root}/vq").eval()
    dino = AutoModel.from_pretrained(f"{root}/dino").eval()
    norm_vae = lambda a: torch.from_numpy(a).permute(2, 0, 1).float() / 127.5 - 1
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for split in ("train", "test"):
        names = sorted(os.listdir(f"{root}/ffhq/{split}"))
        os.makedirs(f"{root}/cache/{split}/ae_latents", exist_ok=True)
        os.makedirs(f"{root}/cache/{split}/dino_features", exist_ok=True)
        lat_all, feat_all = [], []
        for name in names:
            arr = np.asarray(Image.open(f"{root}/ffhq/{split}/{name}").convert("RGB"))
            with torch.no_grad():
                lat = vq.encode(norm_vae(arr)[None]).latents[0]
                feat = dino(((torch.from_numpy(arr).permute(2, 0, 1).float() / 255 - mean) / std)[None]).last_hidden_state[0]
            stem = os.path.splitext(name)[0]
            np.save(f"{root}/cache/{split}/ae_latents/{stem}.npy", lat.half().numpy())
            np.save(f"{root}/cache/{split}/dino_features/{stem}.npy", feat.half().numpy())
            lat_all.append(lat.half().numpy())
            feat_all.append(feat.half().numpy())
        with h5py.File(f"{root}/cache/{split}.h5", "w") as f:
            f.create_dataset("ae_latents", data=np.stack(lat_all))
            f.create_dataset("dino_features", data=np.stack(feat_all))
            f.create_dataset("filenames", data=np.array(names, dtype=object), dtype=h5py.special_dtype(vlen=str))


def build_token_data(root):
    """Random 32x128 token files for the AFHQ and BraTS separator datasets, with their label files."""
    g = torch.Generator().manual_seed(3)
    os.makedirs(f"{root}/tokens_afhq/train", exist_ok=True)
    os.makedirs(f"{root}/tokens_afhq/test", exist_ok=True)
    csv_lines = []
    for split, n in (("train", 72), ("test", 36)):
        for i in range(n):
            kind = ["cat", "dog", "wild", "cat"][i % 4]
            name = f"flickr_{kind}_{split}_{i:04d}"
            scores = [(i * k) % 7 / 7.0 for k in (1, 2, 3)]
            breed = 0 if kind != "cat" else 1 + int(np.argmax(scores))
            tokens = torch.randn(N_TOKENS, TOKEN_DIM, generator=g) + 1.5 * _pattern(10 + breed)  # depends on the breed
            torch.save(tokens, f"{root}/tokens_afhq/{split}/{name}.pt")
            if kind == "cat" and i % 9 != 5:  # some cats have no breed scores (skipped)
                a, b, c = scores
                csv_lines.append(f"{name}.jpg, tabby cat {a:.3f}, Egyptian cat {b:.3f}, tiger cat {c:.3f}, lynx 0.01")
    with open(f"{root}/afhq_breeds.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n\n")

    for split, n_cases in (("train", 6), ("test", 3)):
        os.makedirs(f"{root}/tokens_brats/{split}", exist_ok=True)
        lines = []
        for case in range(n_cases):
            for s in range(84, 102):
                name = f"BraTS-GLI-{case:05d}-000_slice_{s:03d}"
                label = int((case + s) % 3 == 0)
                torch.save(torch.randn(N_TOKENS, TOKEN_DIM, generator=g) + label, f"{root}/tokens_brats/{split}/{name}.pt")
                if s != 90 or case != 1:  # one slice without a label (skipped)
                    lines.append(f"{name}.png\t{label}" if s % 2 else f"{name}.png {label}")
        with open(f"{root}/brats_{split}_labels.txt", "w") as f:
            f.write("\n".join(lines + ["", "malformed_line", "x.png notanint"]) + "\n")


def build_ffhq_tokens(root):
    """Random 32x128 token files named like the FFHQ images (labels come from the attribute JSON files)."""
    g = torch.Generator().manual_seed(4)
    for split, ids in (("train", range(0, 40)), ("test", range(1000, 1024))):
        os.makedirs(f"{root}/tokens_ffhq/{split}", exist_ok=True)
        for idx in ids:
            attrs = face_attributes(idx)[0]["faceAttributes"]
            smile = attrs["smile"] / (100.0 if attrs["smile"] > 1 else 1.0)
            tokens = (torch.randn(N_TOKENS, TOKEN_DIM, generator=g) + 1.5 * _pattern(_GLASSES.index(attrs["glasses"]))
                      + attrs["headPose"]["yaw"] / 20.0 * _pattern(5) + 2.0 * (smile >= 0.5) * _pattern(6))
            torch.save(tokens, f"{root}/tokens_ffhq/{split}/{idx:05d}.pt")
    # an unreadable feature file (skipped with a warning when loaded in RAM)
    with open(f"{root}/tokens_ffhq/train/00039.pt", "wb") as f:
        f.write(b"not a tensor")


TINY_UNET = {
    "sample_size": 64, "in_channels": 3, "out_channels": 3, "norm_eps": 1.0e-05, "layers_per_block": 1,
    "norm_num_groups": 8, "act_fn": "silu", "block_out_channels": [16, 32], "cross_attention_dim": 128,
    "attention_head_dim": 8, "dropout": 0.1,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"], "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D"],
}
TINY_CQ = {"backbone_type": "DINOv3", "frozen_backbone": True, "n_tokens": 32, "embedding_dim": 768,
           "output_dim": 128, "dim": 64, "depth": 2, "dim_head": 16, "heads": 2, "ff_mult": 2}


def stage1_config(root, name, encoder="CrossQueryColorEncoder", online=False):
    """A Stage 1 config with the structure of the paper configs, scaled down."""
    unet = copy.deepcopy(TINY_UNET)
    if online:  # AFHQ-like: KL VAE (4 latent channels at 32x32), online features, all augmentations
        unet.update(sample_size=32, in_channels=4, out_channels=4)
    encoder_conf = {"type": encoder, **TINY_CQ, "backbone_path": f"{root}/dino"}
    if encoder == "DinoEncoder":
        encoder_conf = {"type": "DinoEncoder", "n_tokens": 261, "embedding_dim": 768, "output_dim": 128}
    elif encoder == "EncDiffEncoder":
        encoder_conf = {"type": "EncDiffEncoder", "d": 16, "context_dim": 128, "latent_unit": 32, "n_tokens": 32,
                        "output_dim": 128, "backbone_path": f"{root}/dino"}
    config = {
        "experiment_id": f"tiny_{name}",
        "output_dir": f"{root}/runs/{name}",
        "mixed_precision": "no",
        "seed": 42,
        "data": {
            "dataset_path": {"train": f"{root}/afhq/train", "test": f"{root}/afhq/val"} if online
            else {"train": f"{root}/ffhq/train", "test": f"{root}/ffhq/test"},
            "batch_size": 8, "num_workers": 0, "quick_eval_size": 4, "fid_eval_size": 8,
            "use_cached_features": not online,
            "cache_paths": {"train": f"{root}/cache/train", "test": f"{root}/cache/test"},
        },
        "augmentations": {
            "enable": online,
            "drop_dino": online,
            "geometric": {"enable": online},
            "transforms": [
                {"name": "RandomGrayscale", "params": {"p": 0.5}},
                {"name": "ColorJitter", "params": {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.2},
                 "random_apply_p": 0.8},
            ],
        },
        "model": {"unet_config": unet, "image_encoder": encoder_conf},
        "vae": {
            "type": "AutoencoderKL" if online else "VQModel",
            "config_path": f"{root}/{'kl' if online else 'vq'}_ae_config.json",
            "pretrained_path": f"{root}/{'kl' if online else 'vq'}",
        },
        "training": {"num_steps": 6, "base_lr": 1.0e-3, "max_grad_norm": 1.0, "weight_decay": 0.01,
                     "save_every_steps": 3, "quick_eval_every_steps": 4, "fid_eval_every_steps": 6},
        "sampling": {"num_inference_steps": 3},
    }
    return config


def write_yaml(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path


def build_all(root):
    """Build every fixture under `root` (skipped when already built)."""
    marker = f"{root}/.built"
    if os.path.exists(marker):
        if not os.path.exists(f"{root}/tokens_ffhq"):
            build_ffhq_tokens(root)
        return root
    os.makedirs(root, exist_ok=True)
    build_tiny_backbone(f"{root}/dino")
    build_tiny_vq(f"{root}/vq")
    build_tiny_kl(f"{root}/kl")
    from diffusers import AutoencoderKL, VQModel
    _write_ae_config(VQModel, f"{root}/vq", f"{root}/vq_ae_config.json", 1.37)
    _write_ae_config(AutoencoderKL, f"{root}/kl", f"{root}/kl_ae_config.json", 0.71)
    build_image_data(root)
    build_cached_features(root)
    build_token_data(root)
    build_ffhq_tokens(root)
    open(marker, "w").close()
    return root
