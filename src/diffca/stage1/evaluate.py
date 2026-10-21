"""Stage 1 reconstruction metrics (LPIPS, SSIM, DISTS, RMSE, PSNR, FID, FD-DINO, KID), multi-GPU safe."""

import os

import torch
from accelerate import Accelerator
from torch import nn
from torchmetrics.functional.image import deep_image_structure_and_texture_similarity
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchmetrics.image.kid import KernelInceptionDistance, poly_mmd
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from diffca.utils import print0

COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_GRAY = "\033[90m"
COLOR_RESET = "\033[0m"


class DinoEvalWrapper(nn.Module):
    """Feature extractor for FD-DINO: the backbone's pooled output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param = next(self.model.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        return self.model(x).pooler_output


def compute_multigpu_fid(accelerator: Accelerator, fid_metric: FrechetInceptionDistance):
    """FID from the feature statistics summed over all processes."""
    num_real = accelerator.reduce(fid_metric.real_features_num_samples, reduction="sum")
    real_sum = accelerator.reduce(fid_metric.real_features_sum, reduction="sum")
    real_cov = accelerator.reduce(fid_metric.real_features_cov_sum, reduction="sum")

    num_fake = accelerator.reduce(fid_metric.fake_features_num_samples, reduction="sum")
    fake_sum = accelerator.reduce(fid_metric.fake_features_sum, reduction="sum")
    fake_cov = accelerator.reduce(fid_metric.fake_features_cov_sum, reduction="sum")

    if not accelerator.is_main_process:
        return 0.0

    real_sum, real_cov = real_sum.double(), real_cov.double()
    fake_sum, fake_cov = fake_sum.double(), fake_cov.double()

    mean_real = (real_sum / num_real).unsqueeze(0)
    mean_fake = (fake_sum / num_fake).unsqueeze(0)

    cov_real = (real_cov - num_real * mean_real.t().mm(mean_real)) / (num_real - 1)
    cov_fake = (fake_cov - num_fake * mean_fake.t().mm(mean_fake)) / (num_fake - 1)

    if torch.isnan(cov_real).any() or torch.isnan(cov_fake).any():
        print0(f"{COLOR_YELLOW}NaN detected in covariance matrices. Returning FID 0.{COLOR_RESET}", accelerator=accelerator)
        return 0.0

    return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0).contiguous(), cov_fake.contiguous()).item()


def compute_multigpu_kid(accelerator: Accelerator, kid_metric: KernelInceptionDistance):
    """KID (mean, std) from the features gathered over all processes."""
    if not kid_metric.real_features or not kid_metric.fake_features:
        return 0.0, 0.0

    real_features = torch.cat([feat.detach() for feat in kid_metric.real_features], dim=0)
    fake_features = torch.cat([feat.detach() for feat in kid_metric.fake_features], dim=0)
    real_features = accelerator.gather_for_metrics(real_features)
    fake_features = accelerator.gather_for_metrics(fake_features)

    if not accelerator.is_main_process:
        return 0.0, 0.0

    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    subset_size = min(kid_metric.subset_size, n_real, n_fake)

    if subset_size < 2:
        print0(f"{COLOR_YELLOW}Not enough samples for KID. Returning KID 0.{COLOR_RESET}", accelerator=accelerator)
        return 0.0, 0.0
    if subset_size != kid_metric.subset_size:
        print0(f"{COLOR_YELLOW}Adjusting KID subset_size from {kid_metric.subset_size} to {subset_size}.{COLOR_RESET}",
               accelerator=accelerator)

    kid_scores = []
    for _ in range(kid_metric.subsets):
        perm = torch.randperm(n_real, device=real_features.device)
        f_real = real_features[perm[:subset_size]]
        perm = torch.randperm(n_fake, device=fake_features.device)
        f_fake = fake_features[perm[:subset_size]]
        kid_scores.append(poly_mmd(f_real, f_fake, kid_metric.degree, kid_metric.gamma, kid_metric.coef))

    kid_scores = torch.stack(kid_scores)
    return kid_scores.mean().item(), kid_scores.std(unbiased=False).item()


class _RunningStats:
    """Streaming mean/std of per-image scores, reduced over processes at the end."""

    def __init__(self, device):
        self.sum = torch.zeros((), device=device, dtype=torch.float32)
        self.sumsq = torch.zeros((), device=device, dtype=torch.float32)
        self.count = torch.zeros((), device=device, dtype=torch.float32)

    def update(self, values):
        if values is None:
            return
        if values.dim() == 0:
            values = values.unsqueeze(0)
        values = values.detach().float()
        self.sum += values.sum()
        self.sumsq += (values * values).sum()
        self.count += values.numel()

    def running_mean(self):
        return (self.sum / self.count).item()

    def gather(self, accelerator):
        total_sum = accelerator.reduce(self.sum, reduction="sum")
        total_sumsq = accelerator.reduce(self.sumsq, reduction="sum")
        total_count = accelerator.reduce(self.count, reduction="sum")
        if not accelerator.is_main_process:
            return 0.0, 0.0
        count = total_count.item()
        if count == 0:
            return 0.0, 0.0
        mean = (total_sum / count).item()
        var = (total_sumsq / count) - (mean * mean)
        std = torch.sqrt(torch.clamp(var, min=0.0)).item()
        return mean, std


def sample_euler(velocity_fn, x_t, inference_steps, device, dtype):
    """Integrate the flow from t=0 (noise) to t=1 (data) with `inference_steps` Euler steps."""
    bsz = x_t.shape[0]
    dt = 1.0 / inference_steps
    for i in range(inference_steps):
        t_tensor = torch.full((bsz,), i / inference_steps, device=device, dtype=dtype)
        x_t = x_t + velocity_fn(x_t, t_tensor) * dt
    return x_t


def _save_visuals(save_path, model, vae, real_images, gen_images, images, clean_latents, dino_embeds, x_t_init,
                  inference_steps, accelerator):
    """First batch only: reconstructions, and reconstructions with the color token cyclically permuted."""
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    n_vis = min(4, gen_images.shape[0])
    real_vis = (real_images[:n_vis] + 1) / 2
    gen_vis = (gen_images[:n_vis] + 1) / 2
    diff_vis = (real_images[:n_vis] - gen_images[:n_vis]).abs() / 2
    save_image(torch.cat([real_vis, gen_vis, diff_vis], dim=0),
               os.path.join(save_path, "images", "recon_absdiff_grid.png"), nrow=n_vis)

    if dino_embeds is None:
        return
    with accelerator.autocast():
        context_tokens = model.encoder.encode(
            backbone_embeds=dino_embeds[:n_vis], latents=clean_latents[:n_vis], images=images[:n_vis]
        ).to(clean_latents.dtype)

        perm_tokens = context_tokens.clone()
        for idx in range(n_vis):
            perm_tokens[idx, 0] = context_tokens[(idx + 1) % n_vis, 0]

        x_perm = sample_euler(lambda x, t: model.diffusion_model(x, t, perm_tokens), x_t_init[:n_vis].clone(),
                              inference_steps, accelerator.device, clean_latents.dtype)
        perm_images = vae.decode(x_perm / getattr(vae.config, "scaling_factor", 0.18215)).sample
        perm_images = perm_images.clamp(-1, 1).float()
        perm_vis = (perm_images + 1) / 2

    swap_diff_vis = (gen_images[:n_vis] - perm_images[:n_vis]).abs() / 2
    save_image(torch.cat([real_vis, gen_vis, perm_vis, swap_diff_vis], dim=0),
               os.path.join(save_path, "images", "color_token_permutation_grid.png"), nrow=n_vis)


def run_evaluation_on_model(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    inference_steps: int,
    starting_noise: torch.Tensor,
    fid=True,
    save_path: str = None,
    display=False,
):
    """Reconstruct every image (encode -> sample from noise -> decode) and score it against the input."""
    vae = model.vae
    backbone = model.backbone
    device = accelerator.device

    def get_eval_images(batch):
        if "pixel_values" in batch:
            return batch["pixel_values"]
        if "pixel_values_clean" in batch:
            return batch["pixel_values_clean"]
        raise KeyError(f"{COLOR_RED}Batch missing expected image key: 'pixel_values' or 'pixel_values_clean'{COLOR_RESET}")

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

    stats = {name: _RunningStats(device) for name in ("lpips_vgg", "lpips_alex", "ssim", "dists", "rmse", "psnr")}
    dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.inference_mode():
        lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="none").to(device)
        lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="none").to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0, reduction="none").to(device)

        if fid:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device).float()
            fd_dino_metric = None
            if backbone is not None:
                fd_dino_metric = FrechetInceptionDistance(feature=DinoEvalWrapper(backbone), normalize=True).to(device).float()
            kid_metric = KernelInceptionDistance(feature=2048).to(device)

        for step, batch in enumerate(pbar):
            with accelerator.autocast():
                images = get_eval_images(batch).to(device=device, dtype=vae.dtype)

                enc_out = vae.encode(images)
                clean_latents = enc_out.latent_dist.mode() if hasattr(enc_out, "latent_dist") else enc_out.latents
                clean_latents = clean_latents * getattr(vae.config, "scaling_factor")

                dino_embeds = None
                if "dino" in batch:
                    dino_embeds = batch["dino"].to(device=device, dtype=clean_latents.dtype)
                elif backbone:
                    dino_images = batch.get("pixel_values_dino", images)
                    dino_embeds = backbone(dino_images.to(device=device, dtype=backbone.dtype)).last_hidden_state

                bsz = clean_latents.shape[0]
                if starting_noise is not None:
                    x_t = starting_noise[:bsz].to(clean_latents.device).to(clean_latents.dtype).clone()
                else:
                    x_t = torch.randn_like(clean_latents)
                x_t_init = x_t.clone()

                # `model` holds the EMA weights here (see `evaluate`).
                x_t = sample_euler(
                    lambda x, t: model.denoise_step(x_noisy=x, dino_embeds=dino_embeds, t=t,
                                                    clean_latents=clean_latents, clean_images=images),
                    x_t, inference_steps, device, clean_latents.dtype,
                )
                gen_images = vae.decode(x_t / getattr(vae.config, "scaling_factor", 0.18215)).sample

            gen_images = gen_images.clamp(-1, 1).float()
            real_images = images.float()

            if save_path and step == 0:
                _save_visuals(save_path, model, vae, real_images, gen_images, images, clean_latents, dino_embeds,
                              x_t_init, inference_steps, accelerator)

            stats["lpips_vgg"].update(lpips_vgg(gen_images, real_images))
            stats["lpips_alex"].update(lpips_alex(gen_images, real_images))
            stats["ssim"].update(ssim_metric(gen_images, real_images))
            stats["dists"].update(deep_image_structure_and_texture_similarity(gen_images, real_images))
            mse = torch.mean((gen_images - real_images) ** 2, dim=[1, 2, 3])
            stats["rmse"].update(torch.sqrt(mse))
            stats["psnr"].update(10 * torch.log10(4.0 / mse))

            if display:
                pbar.set_postfix({
                    "LPIPS_ALEX": f"{stats['lpips_alex'].running_mean():.4f}",
                    "DISTS": f"{stats['dists'].running_mean():.4f}",
                    "PSNR": f"{stats['psnr'].running_mean():.4f}",
                })

            if fid:
                real_01 = (real_images + 1) / 2
                gen_01 = (gen_images + 1) / 2
                fid_metric.update(real_01, real=True)
                fid_metric.update(gen_01, real=False)
                if fd_dino_metric is not None:
                    fd_dino_metric.update(dino_norm(real_01), real=True)
                    fd_dino_metric.update(dino_norm(gen_01), real=False)
                kid_metric.update((real_01 * 255).clamp(0, 255).to(torch.uint8), real=True)
                kid_metric.update((gen_01 * 255).clamp(0, 255).to(torch.uint8), real=False)

    results = {}
    for name in ("lpips_vgg", "lpips_alex", "ssim", "dists", "rmse", "psnr"):
        results[name], results[f"{name}_std"] = stats[name].gather(accelerator)

    if fid:
        try:
            results["fid"] = compute_multigpu_fid(accelerator, fid_metric)
            if fd_dino_metric is not None:
                results["FD-Dino"] = compute_multigpu_fid(accelerator, fd_dino_metric)
            results["kid"], results["kid_std"] = compute_multigpu_kid(accelerator, kid_metric)
            fid_metric.reset()
            if fd_dino_metric is not None:
                fd_dino_metric.reset()
            kid_metric.reset()
        except Exception as e:
            print0(f"{COLOR_RED}FID/KID Error: {e}{COLOR_RESET}", accelerator=accelerator)
            results["fid"] = 0.0
            results["FD-Dino"] = 0.0
            results["kid"] = 0.0
            results["kid_std"] = 0.0

    return results if accelerator.is_main_process else {}


def evaluate(model, ema_model, trainable_wrapper, accelerator, dataloader, global_step,
             inference_steps=25, starting_noise=None, fid=True, save_path=None, display=False,
             use_ema: bool = True):
    """Evaluate with the EMA weights swapped in (and the online weights restored afterwards)."""
    if accelerator.is_main_process:
        ema_tag = " (EMA)" if use_ema else ""
        print(f"\n{COLOR_GRAY}***** Evaluation at Step {global_step}{ema_tag} *****{COLOR_RESET}\n")

    if use_ema:
        if ema_model is None:
            raise ValueError("EMA requested but ema_model is None.")
        ema_model.store(trainable_wrapper.parameters())
        ema_model.copy_to(trainable_wrapper.parameters())

    results = run_evaluation_on_model(
        accelerator=accelerator,
        model=model,
        dataloader=dataloader,
        inference_steps=inference_steps,
        starting_noise=starting_noise,
        fid=fid,
        save_path=save_path,
        display=display,
    )

    if use_ema:
        ema_model.restore(trainable_wrapper.parameters())

    if accelerator.is_main_process:
        print0(f"{COLOR_GREEN}--- Results ---{COLOR_RESET}", accelerator=accelerator)
        for k, v in results.items():
            print0(f"{k}: {v:.4f}", accelerator=accelerator)
        print0("", accelerator=accelerator)

    return results
