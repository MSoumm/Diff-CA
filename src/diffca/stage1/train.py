"""Stage 1: train the image-conditioned latent flow-matching model (encoder + U-Net).

    python -m diffca.stage1.train --config configs/stage1/ffhq_cqc.yaml [--resume_from <output_dir>/checkpoint-<step>]
"""

import argparse
import os
import shutil
from datetime import datetime

import torch
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from diffca.stage1.build import EMA_DECAY, build_model, resolve_weight_dtype, safe_encode
from diffca.stage1.dataset import HybridDataset
from diffca.stage1.evaluate import evaluate
from diffca.utils import load_config, seed_everything

COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_GRAY = "\033[90m"
COLOR_RESET = "\033[0m"

DROP_DINO_END_T = 0.3  # drop_dino: the backbone tokens are dropped with probability max(0, 1 - t / 0.3)
AUG_CUTOFF_T = 0.3     # augmented latents are used as the flow input with probability min(1, t / 0.3)
OFFSET_NOISE = 0.1


def append_metrics_log(save_dir, global_step, tag, metrics):
    if not metrics:
        return
    items = [f"step={global_step}", f"tag={tag}"] + [f"{key}={metrics[key]:.6f}" for key in sorted(metrics)]
    with open(os.path.join(save_dir, "metrics.txt"), "a") as f:
        f.write(" ".join(items) + "\n")


def save_config_copy(config_path, save_dir):
    dest_config = os.path.join(save_dir, "config_saved.yaml")
    if os.path.exists(dest_config):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_config = os.path.join(save_dir, f"config_saved_{timestamp}.yaml")
    if config_path != dest_config:
        shutil.copy(config_path, dest_config)


def flow_matching_loss(model, batch, config, weight_dtype, device):
    """One flow-matching loss evaluation, with minibatch optimal-transport pairing of noise and data.

    The flow goes from noise (t=0) to data (t=1): x_t = (1 - t) x_0 + t x_1, target velocity x_1 - x_0.
    """
    vae = model.vae
    images_aug = None

    if batch.get("is_cached", [False])[0]:
        clean_latents = batch["latent"].to(weight_dtype)
        dino_embeds = batch["dino"].to(weight_dtype)
        clean_images = batch.get("pixel_values", None)  # not provided by HybridDataset, so None
        if clean_images is not None:
            clean_images = clean_images.to(weight_dtype)
    else:
        images_clean = batch["pixel_values_clean"].to(weight_dtype)
        images_aug = batch.get("pixel_values_aug")
        if images_aug is not None:
            images_aug = images_aug.to(weight_dtype)
        images_dino = batch["pixel_values_dino"].to(weight_dtype)
        clean_latents = safe_encode(vae, images_clean) * vae.config.scaling_factor
        dino_embeds = model.backbone(images_dino).last_hidden_state  # from the color-augmented image
        clean_images = images_clean

    bsz = clean_latents.shape[0]
    t = torch.rand((bsz,), device=device, dtype=clean_latents.dtype)

    if config.get("augmentations", {}).get("drop_dino", False):
        p_drop_dino = torch.clamp(1.0 - (t / DROP_DINO_END_T), min=0.0, max=1.0)
        rand_vals = torch.rand(bsz, device=device)
        drop_mask = (rand_vals < p_drop_dino).to(clean_latents.dtype).view(bsz, 1, 1)
        dino_embeds = dino_embeds * (1.0 - drop_mask)

    x_0 = torch.randn_like(clean_latents)
    offset_noise = torch.randn(x_0.shape[0], x_0.shape[1], 1, 1, device=x_0.device)
    x_0 = x_0 + OFFSET_NOISE * offset_noise

    # Optimal transport: reorder the data so that each noise sample is paired with a close latent.
    with torch.no_grad():
        dists = torch.cdist(x_0.reshape(x_0.shape[0], -1), clean_latents.reshape(clean_latents.shape[0], -1))
        _, col_ind = linear_sum_assignment(dists.cpu().numpy())
    clean_latents = clean_latents[col_ind]
    dino_embeds = dino_embeds[col_ind]
    if clean_images is not None:
        clean_images = clean_images[col_ind]
    if images_aug is not None:
        images_aug = images_aug[col_ind]

    # Time-dependent augmentation of the flow input (decided per slot, after the OT reordering).
    latents_for_input = clean_latents
    if images_aug is not None:
        p_aug = torch.where(t >= AUG_CUTOFF_T, torch.tensor(1.0, device=device), t / AUG_CUTOFF_T)
        use_aug_mask = torch.rand((bsz,), device=device) < p_aug
        if use_aug_mask.any():
            aug_latents = safe_encode(vae, images_aug[use_aug_mask]) * vae.config.scaling_factor
            latents_for_input = clean_latents.clone()
            latents_for_input[use_aug_mask] = aug_latents

    t_view = t.view(bsz, *([1] * (clean_latents.ndim - 1)))
    x_t = (1 - t_view) * x_0 + t_view * latents_for_input
    flow_target = clean_latents - x_0

    model_pred = model.denoise_step(x_noisy=x_t, dino_embeds=dino_embeds, t=t,
                                    clean_latents=clean_latents, clean_images=clean_images)
    return F.mse_loss(model_pred, flow_target)


def save_final_weights(accelerator, trainable_parts, ema_model, save_dir):
    """Save the last training state, plus the EMA weights as plain state dicts."""
    accelerator.save_state(os.path.join(save_dir, "checkpoint-final"))

    ema_model.store(trainable_parts.parameters())
    ema_model.copy_to(trainable_parts.parameters())
    # unwrap=False: unwrapping a wrapper of compiled modules fails with accelerate 1.8 (the keys are the same).
    state_dict = accelerator.get_state_dict(trainable_parts, unwrap=False)
    normalized_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    torch.save(normalized_state, os.path.join(save_dir, "pytorch_model_ema.bin"))
    for prefix, filename in (("diffusion_model.", "diffusion_model_ema.bin"), ("encoder.", "encoder_ema.bin")):
        part = {k.replace(prefix, "", 1): v for k, v in normalized_state.items() if k.startswith(prefix)}
        torch.save(part, os.path.join(save_dir, filename))
    ema_model.restore(trainable_parts.parameters())


def main():
    parser = argparse.ArgumentParser(description="Stage 1: train the conditional flow-matching model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the Stage 1 config file")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint-<step> directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["augmentations"]["enable"] and config["data"]["use_cached_features"]:
        raise ValueError(
            f"{COLOR_RED}CRITICAL: Augmentations are enabled but use_cached_features is TRUE. "
            f"Geometric augmentations break cached features. Set use_cached_features=False.{COLOR_RESET}"
        )

    seed_everything(config["seed"])
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    save_dir = config["output_dir"]
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        save_config_copy(args.config, save_dir)
        print(f"\n{COLOR_GRAY}Experiment:{COLOR_RESET} {config['experiment_id']}")
        print(f"{COLOR_GRAY}Output Dir:{COLOR_RESET} {save_dir}\n")

    weight_dtype = resolve_weight_dtype(accelerator)

    # --- Model, optimizer ---
    model, trainable_parts, scaling_factor = build_model(config, use_compile=True, profile=True, require_vae_config=True)
    config["vae"]["scaling_factor"] = scaling_factor

    optimizer = torch.optim.AdamW(
        trainable_parts.parameters(),
        lr=float(config["training"]["base_lr"]),
        weight_decay=config["training"]["weight_decay"],
    )

    # --- Data ---
    data_conf = config["data"]
    train_dataset = HybridDataset(config, is_train=True)
    test_dataset = HybridDataset(config, is_train=False, max_samples=data_conf.get("fid_eval_size", None))
    print(f"\n{COLOR_GRAY}Train Dataset Length:{COLOR_RESET} {len(train_dataset)}")
    print(f"{COLOR_GRAY}Test Dataset Length:{COLOR_RESET} {len(test_dataset)}\n")
    if len(train_dataset) == 0:
        raise ValueError(f"{COLOR_RED}Train dataset is empty. Please check the dataset path and contents.{COLOR_RESET}")
    if len(test_dataset) == 0:
        raise ValueError(f"{COLOR_RED}Test dataset is empty. Please check the dataset path and contents.{COLOR_RESET}")

    train_dataloader = DataLoader(train_dataset, batch_size=data_conf["batch_size"], shuffle=True,
                                  num_workers=data_conf["num_workers"], pin_memory=True)
    quick_indices = range(min(len(test_dataset), data_conf.get("quick_eval_size", 1000)))
    quick_eval_loader = DataLoader(Subset(test_dataset, quick_indices), batch_size=data_conf["batch_size"],
                                   shuffle=False, num_workers=4)
    fid_indices = range(min(len(test_dataset), data_conf.get("fid_eval_size", 10000)))
    full_eval_loader = DataLoader(Subset(test_dataset, fid_indices), batch_size=data_conf["batch_size"],
                                  shuffle=False, num_workers=8)

    trainable_parts, optimizer, train_dataloader, quick_eval_loader, full_eval_loader = accelerator.prepare(
        trainable_parts, optimizer, train_dataloader, quick_eval_loader, full_eval_loader
    )

    ema_model = EMAModel(trainable_parts.parameters(), decay=EMA_DECAY)
    ema_model.to(next(trainable_parts.parameters()).device)
    accelerator.register_for_checkpointing(ema_model)  # so that the EMA survives a resume

    model.vae.to(accelerator.device, dtype=weight_dtype)
    if model.backbone:
        model.backbone.to(accelerator.device, dtype=weight_dtype)
    model.set_prepared_model(trainable_parts)

    # --- Resume ---
    global_step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        ema_model.to(next(trainable_parts.parameters()).device)
        print(f"\n{COLOR_GREEN}Loaded checkpoint from {args.resume_from}{COLOR_RESET}")
        try:
            global_step = int(args.resume_from.strip("/").split("-")[-1])
            print(f"{COLOR_GRAY}Resumed from step:{COLOR_RESET} {global_step}")
        except ValueError:
            global_step = 0
            print(f"{COLOR_YELLOW}Warning: Could not parse step from resume path, starting at step 0.{COLOR_RESET}")
        cfg_lr = float(config["training"]["base_lr"])
        opt_lr = optimizer.param_groups[0].get("lr", cfg_lr)
        if abs(opt_lr - cfg_lr) > 1e-12:
            print(f"{COLOR_YELLOW}Warning: checkpoint lr={opt_lr:.6g} differs from config lr={cfg_lr:.6g}. Overwriting.{COLOR_RESET}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg_lr
        print()

    # --- Training loop ---
    train_conf = config["training"]
    unet_conf = config["model"]["unet_config"]
    max_steps = train_conf["num_steps"]
    inference_steps = config["sampling"]["num_inference_steps"]
    fixed_noise = None  # starting noise of the FID evaluations, drawn once
    if accelerator.is_main_process:
        print(f"{COLOR_GRAY}Starting Training... {max_steps} Steps.{COLOR_RESET}\n")

    epoch = 0
    while global_step < max_steps:
        model.train()
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # The whole step, evaluations included, runs under autocast, as for the paper models.
            with accelerator.autocast():
                loss = flow_matching_loss(model, batch, config, weight_dtype, device)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parts.parameters(), train_conf["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    ema_model.step(trainable_parts.parameters())
                    global_step += 1
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

                    if global_step % train_conf["save_every_steps"] == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            print(f"{COLOR_GREEN}Saved checkpoint:{COLOR_RESET} {save_path}")
                        accelerator.wait_for_everyone()

                    if global_step % train_conf["quick_eval_every_steps"] == 0:
                        if accelerator.is_main_process:
                            print(f"{COLOR_YELLOW}Running Quick Eval (LPIPS/SSIM) on {len(quick_eval_loader.dataset)} images...{COLOR_RESET}")
                        quick_metrics = evaluate(
                            model=model, ema_model=ema_model, trainable_wrapper=trainable_parts,
                            accelerator=accelerator, dataloader=quick_eval_loader, global_step=global_step,
                            inference_steps=inference_steps, starting_noise=None, fid=False,
                            save_path=os.path.join(save_dir, f"quick-eval-step-{global_step}"),
                        )
                        if accelerator.is_main_process:
                            append_metrics_log(save_dir, global_step, "quick", quick_metrics)
                        model.train()

                    if global_step % train_conf["fid_eval_every_steps"] == 0:
                        if accelerator.is_main_process:
                            print(f"{COLOR_YELLOW}Running Full Eval (FID) on {len(full_eval_loader.dataset)} images...{COLOR_RESET}")
                        if fixed_noise is None:
                            fixed_noise = torch.randn(
                                (data_conf["batch_size"], unet_conf["in_channels"], unet_conf["sample_size"], unet_conf["sample_size"]),
                                device=device,
                            )
                        full_metrics = evaluate(
                            model=model, ema_model=ema_model, trainable_wrapper=trainable_parts,
                            accelerator=accelerator, dataloader=full_eval_loader, global_step=global_step,
                            inference_steps=inference_steps, starting_noise=fixed_noise, fid=True,
                            save_path=os.path.join(save_dir, f"full-eval-step-{global_step}"),
                        )
                        if accelerator.is_main_process:
                            append_metrics_log(save_dir, global_step, "fid", full_metrics)
                        model.train()

                if global_step >= max_steps:
                    break
        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\n{COLOR_GREEN}Training finished. Saving final model...{COLOR_RESET}")
        save_final_weights(accelerator, trainable_parts, ema_model, save_dir)
        print(f"{COLOR_GREEN}Final model saved to{COLOR_RESET} {save_dir}")

    accelerator.end_training()
    print(f"{COLOR_GRAY}Finished script.{COLOR_RESET}")


if __name__ == "__main__":
    main()
