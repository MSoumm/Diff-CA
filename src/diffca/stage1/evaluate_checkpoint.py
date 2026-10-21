"""Reconstruction metrics of a trained Stage 1 checkpoint (PSNR, SSIM, LPIPS, DISTS, FID, FD-DINO, KID).

    python -m diffca.stage1.evaluate_checkpoint --config configs/stage1/ffhq_cqc.yaml \
        --checkpoint <stage1_output>/checkpoint-300000 --eval_samples 1000

Every batch starts from the same noise, drawn once after loading the checkpoint. FD-DINO uses the DINOv3 backbone of `--backbone_path`, else of `model.image_encoder.backbone_path`, else
facebook/dinov3-vitb16-pretrain-lvd1689m (as for the DINOv3 / CLIP baselines, whose configs name no backbone).
"""

import argparse

import torch
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from diffca.stage1.build import DEFAULT_BACKBONE, EMA_DECAY, build_model, resolve_weight_dtype
from diffca.stage1.dataset import HybridDataset
from diffca.stage1.evaluate import evaluate
from diffca.utils import load_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Evaluate the reconstructions of a Stage 1 checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the Stage 1 config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint-<step> directory.")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Number of test images (default: one batch, data.batch_size).")
    parser.add_argument("--inference_steps", type=int, default=None,
                        help="Sampling steps (default: sampling.num_inference_steps).")
    parser.add_argument("--backbone_path", type=str, default=None, help="DINOv3 weights, for FD-DINO.")
    parser.add_argument("--no_ema", action="store_true", help="Evaluate the online weights instead of the EMA.")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["augmentations"]["enable"] and config["data"]["use_cached_features"]:
        raise ValueError("CRITICAL: Augmentations are enabled but use_cached_features is TRUE. "
                         "Geometric augmentations break cached features. Set use_cached_features=False.")
    encoder_conf = config["model"]["image_encoder"]
    encoder_conf["backbone_path"] = args.backbone_path or encoder_conf.get("backbone_path") or DEFAULT_BACKBONE

    seed_everything(config["seed"])
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device
    weight_dtype = resolve_weight_dtype(accelerator)

    model, trainable_parts, scaling_factor = build_model(config, use_compile=True, require_vae_config=True)
    config["vae"]["scaling_factor"] = scaling_factor
    trainable_parts.eval()

    eval_samples = args.eval_samples or config["data"]["batch_size"]
    test_dataset = HybridDataset(config, is_train=False, max_samples=eval_samples)
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Please check the dataset path and contents.")
    eval_batch_size = min(config["data"]["batch_size"], len(test_dataset))
    eval_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False,
                             num_workers=config["data"]["num_workers"], pin_memory=True)
    trainable_parts, eval_loader = accelerator.prepare(trainable_parts, eval_loader)

    # Training checkpoints always contain the EMA, so it is registered even when evaluating the online weights.
    ema_model = EMAModel(trainable_parts.parameters(), decay=EMA_DECAY)
    ema_model.to(next(trainable_parts.parameters()).device)
    accelerator.register_for_checkpointing(ema_model)

    model.vae.to(device, dtype=weight_dtype)
    if model.backbone:
        model.backbone.to(device, dtype=weight_dtype)
    model.set_prepared_model(trainable_parts)

    accelerator.load_state(args.checkpoint)  # also restores the RNG state saved with the checkpoint
    if accelerator.is_main_process:
        print(f"Loaded checkpoint from {args.checkpoint}")
    try:
        global_step = int(args.checkpoint.strip("/").split("-")[-1])
    except ValueError:
        global_step = 0

    unet_conf = config["model"]["unet_config"]
    starting_noise = torch.randn((eval_batch_size, unet_conf["in_channels"], unet_conf["sample_size"], unet_conf["sample_size"]),
                                 device=device, dtype=weight_dtype)
    evaluate(
        model=model, ema_model=ema_model, trainable_wrapper=trainable_parts, accelerator=accelerator,
        dataloader=eval_loader, global_step=global_step,
        inference_steps=args.inference_steps or config["sampling"]["num_inference_steps"],
        starting_noise=starting_noise, fid=True, save_path=None, display=True, use_ema=not args.no_ema,
    )


if __name__ == "__main__":
    main()
