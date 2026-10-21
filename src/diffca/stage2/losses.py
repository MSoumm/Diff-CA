"""Stage 2 losses: reconstruction, pinning, cycle consistency, and the adversary with self-tuning gradient reversal.

Total loss = lambda_recon * L_rec
           + lambda_norm_pos * mean_{Y=1} exp(-||h||^2)    (keeps h away from 0 on target samples)
           + lambda_norm_neg * mean_{Y=0} ||h||^2           (pins h to 0 on background samples)
           + lambda_cycle_context * L_cyc(common) + lambda_cycle_cls * L_cyc(salient)
           + lambda_adv * BCE(adversary(GRL(common tokens)), Y)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ADV_ACC_EMA = 0.05  # weight of the current batch in the running adversary accuracy
GRL_FACTOR = 0.999  # multiplicative step of the self-tuning GRL strength
GRL_RANGE = (0.001, 10.0)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.lambda_), None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


def get_lambda(step, start, end, lambda_start=0, lambda_final=1):
    """Linear ramp from `lambda_start` (before `start`) to `lambda_final` (after `end`)."""
    if step < start:
        return lambda_start
    elif step > end:
        return lambda_final
    else:
        return min(lambda_final, lambda_start + (lambda_final - lambda_start) * (step - start) / (end - start))


def warmup_schedule(steps_per_epoch: int, total_epochs: int):
    """(end of warmup, end of the linear ramp) in steps: 5% and 10% of training."""
    warmup_steps = max(1, int(math.ceil(steps_per_epoch * (total_epochs / 20))))
    linear_end = max(warmup_steps + 1, int(math.ceil(steps_per_epoch * (total_epochs / 10))))
    return warmup_steps, linear_end


def compute_reconstruction_loss(reconstructed_tokens, original_src, pos_mask, neg_mask):
    """MSE, averaged per class; the two class means are weighted by their (detached) squares."""
    reconstruction_loss_per_sample = F.mse_loss(reconstructed_tokens, original_src, reduction="none").mean(dim=[1, 2])

    if pos_mask.any() and neg_mask.any():
        pos_recon_loss_val = reconstruction_loss_per_sample[pos_mask].mean()
        neg_recon_loss_val = reconstruction_loss_per_sample[neg_mask].mean()

        pos_recon_loss_detached = pos_recon_loss_val.detach()**2
        neg_recon_loss_detached = neg_recon_loss_val.detach()**2
        total_recon_loss_detached = pos_recon_loss_detached + neg_recon_loss_detached

        if total_recon_loss_detached > 1e-8:
            weight_pos = pos_recon_loss_detached / total_recon_loss_detached
            weight_neg = neg_recon_loss_detached / total_recon_loss_detached
            loss_recon = weight_pos * pos_recon_loss_val + weight_neg * neg_recon_loss_val
        else:
            loss_recon = reconstruction_loss_per_sample.mean()
    else:
        loss_recon = reconstruction_loss_per_sample.mean()

    return loss_recon, reconstruction_loss_per_sample


def compute_cls_loss(cls, pos_mask, neg_mask):
    """Pinning losses on the salient code h: exp(-||h||^2) on Y=1 ("loss_cls"), ||h||^2 on Y=0 ("loss_norm_neg")."""
    cls_norms = torch.linalg.norm(cls, dim=1)
    loss_norm_neg_per_sample = (cls**2).sum(dim=1)
    loss_cls_per_sample = torch.exp(-cls_norms**2 / 1)

    loss_norm_neg = loss_norm_neg_per_sample[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=cls.device)
    loss_cls = loss_cls_per_sample[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=cls.device)
    return loss_cls, loss_norm_neg, cls_norms, loss_cls_per_sample, loss_norm_neg_per_sample


def compute_cycle_loss(cycle_out):
    """MSE between the codes of each sample and the codes re-encoded after the swap (targets detached):
    the common tokens of a sample should survive the swap, and its salient code should move to its partner."""
    if not cycle_out:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), None, None

    def interleave(loss_a, loss_b):
        per_sample = torch.zeros(loss_a.shape[0] * 2, device=loss_a.device, dtype=loss_a.dtype)
        per_sample[0::2] = loss_a
        per_sample[1::2] = loss_b
        return per_sample

    orig_a, orig_b = cycle_out["original_a"], cycle_out["original_b"]
    cyc_a, cyc_b = cycle_out["cycled_a"], cycle_out["cycled_b"]

    loss_ctx_a = F.mse_loss(orig_a["content"].detach(), cyc_a["content"], reduction="none").mean(dim=[1, 2])
    loss_ctx_b = F.mse_loss(orig_b["content"].detach(), cyc_b["content"], reduction="none").mean(dim=[1, 2])
    loss_cycle_context_per_sample = interleave(loss_ctx_a, loss_ctx_b)
    loss_cycle_context = loss_cycle_context_per_sample.mean()

    loss_cls_a = F.mse_loss(orig_a["cls"].detach(), cyc_b["cls"], reduction="none").mean(dim=1)
    loss_cls_b = F.mse_loss(orig_b["cls"].detach(), cyc_a["cls"], reduction="none").mean(dim=1)
    loss_cycle_cls_per_sample = interleave(loss_cls_a, loss_cls_b)
    loss_cycle_cls = loss_cycle_cls_per_sample.mean()

    return loss_cycle_context, loss_cycle_cls, loss_cycle_context_per_sample, loss_cycle_cls_per_sample


def compute_adversarial_loss(context_tokens, labels, joint_adversary, total_step, is_training,
                             low_target_acc=0.85, high_target_acc=0.85, steps_per_epoch: int = 55, total_epochs: int = 1):
    """BCE of the adversary on the common tokens, behind a gradient reversal layer of strength lambda.

    Self-tuning lambda (state in the adversary's buffers, updated during training only):
      - warmup: lambda ramps linearly from 0 to 1 between 5% and 10% of training;
      - adaptive mode (from 10% of training, or earlier once the running accuracy drops below 0.9 after 5%):
        lambda /= 0.999 while the running accuracy is above `high_target_acc` (the adversary wins: push harder),
        lambda *= 0.999 while it is below `low_target_acc`; lambda is clamped to [0.001, 10].
    """
    reversed_context_tokens = grad_reverse(context_tokens, lambda_=joint_adversary.lambda_val)
    adversarial_logits = joint_adversary(reversed_context_tokens).squeeze(-1)

    loss_adversarial_per_sample = F.binary_cross_entropy_with_logits(adversarial_logits, labels.float(), reduction="none")
    loss_adversarial = loss_adversarial_per_sample.mean()

    with torch.no_grad():
        preds = (torch.sigmoid(adversarial_logits) > 0.5).float()
        batch_adv_accuracy = (preds == labels.float()).float().mean()
        if is_training:
            joint_adversary.ema_acc = ADV_ACC_EMA * batch_adv_accuracy.item() + (1 - ADV_ACC_EMA) * joint_adversary.ema_acc

    if is_training:
        warmup_steps, linear_end = warmup_schedule(steps_per_epoch, total_epochs)
        if total_step >= linear_end or (not joint_adversary.adaptive_mode and joint_adversary.ema_acc < 0.9
                                        and total_step > warmup_steps):
            joint_adversary.adaptive_mode = torch.tensor(True, device=joint_adversary.adaptive_mode.device)

        if joint_adversary.adaptive_mode:
            if joint_adversary.ema_acc < low_target_acc:
                joint_adversary.lambda_val *= GRL_FACTOR
            elif joint_adversary.ema_acc > high_target_acc:
                joint_adversary.lambda_val /= GRL_FACTOR
            joint_adversary.lambda_val.clamp_(*GRL_RANGE)
        else:
            joint_adversary.lambda_val = torch.tensor(
                get_lambda(start=warmup_steps, end=linear_end, step=total_step, lambda_final=1),
                device=joint_adversary.lambda_val.device,
            )

    return loss_adversarial, batch_adv_accuracy, loss_adversarial_per_sample


def cycle_loss_criterion(
    training_outputs: dict,
    joint_adversary: nn.Module,
    total_step: int = 0,
    is_training: bool = True,
    steps_per_epoch: int = 55,
    total_epochs: int = 1,
    lambda_recon: float = 1.0,
    lambda_norm_pos: float = 0.1,
    lambda_norm_neg: float = 0.1,
    lambda_cycle_context: float = 1.0,
    lambda_cycle_cls: float = 1.0,
    lambda_adv: float = 1.0,
    low_target_acc: float = 0.85,
    high_target_acc: float = 0.85,
    **kwargs,
):
    """Total separator loss. Returns (loss, components) when training, (loss, components, extras) otherwise.

    Extra keyword arguments (the rest of the training arguments) are ignored.
    """
    std_out = training_outputs["standard_outputs"]
    cycle_out = training_outputs.get("cycle_outputs")
    labels = std_out["labels"]
    pos_mask, neg_mask = (labels == 1), (labels == 0)

    loss_recon, reconstruction_loss_per_sample = compute_reconstruction_loss(
        std_out["reconstructed_tokens"], std_out["original_src"], pos_mask, neg_mask
    )
    loss_cls, loss_norm_neg, cls_norms, loss_cls_per_sample, loss_norm_neg_per_sample = compute_cls_loss(
        std_out["cls"], pos_mask, neg_mask
    )
    loss_cycle_context, loss_cycle_cls, loss_cycle_context_per_sample, loss_cycle_cls_per_sample = compute_cycle_loss(cycle_out)

    if lambda_adv > 0:
        loss_adversarial, batch_adv_accuracy, loss_adversarial_per_sample = compute_adversarial_loss(
            std_out["content_tokens"], labels, joint_adversary, total_step, is_training,
            low_target_acc=low_target_acc, high_target_acc=high_target_acc,
            steps_per_epoch=steps_per_epoch, total_epochs=total_epochs,
        )
    else:
        loss_adversarial, batch_adv_accuracy, loss_adversarial_per_sample = torch.tensor(0.0), torch.tensor(0.0), None

    if lambda_cycle_context > 0:
        # Ramp from 1 to lambda_cycle_context between 5% and 10% of training (constant when it is 1).
        warmup_steps, linear_end = warmup_schedule(steps_per_epoch, total_epochs)
        lambda_cycle_context = get_lambda(step=total_step, start=warmup_steps, end=linear_end,
                                          lambda_start=1, lambda_final=lambda_cycle_context)

    total_loss = (
        lambda_recon * loss_recon
        + lambda_norm_pos * loss_cls
        + lambda_norm_neg * loss_norm_neg
        + lambda_cycle_context * loss_cycle_context
        + lambda_cycle_cls * loss_cycle_cls
        + lambda_adv * loss_adversarial
    )

    loss_components = {
        "total_loss": total_loss,
        "loss_recon": loss_recon,
        "loss_cls": loss_cls,
        "loss_norm_neg": loss_norm_neg,
        "loss_cycle_context": loss_cycle_context,
        "loss_cycle_cls": loss_cycle_cls,
        "loss_adversarial": loss_adversarial,
        # Per-sample losses, logged per class
        "reconstruction_loss_per_sample": reconstruction_loss_per_sample,
        "loss_cls_per_sample": loss_cls_per_sample,
        "loss_norm_neg_per_sample": loss_norm_neg_per_sample,
        "loss_cycle_context_per_sample": loss_cycle_context_per_sample,
        "loss_cycle_cls_per_sample": loss_cycle_cls_per_sample,
        "loss_adversarial_per_sample": loss_adversarial_per_sample,
        # Monitoring
        "recon_rmse": torch.sqrt(reconstruction_loss_per_sample.detach()).mean().item(),
        "positive_mean_norms": cls_norms[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0),
        "negative_mean_norms": cls_norms[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0),
        "lambda_GRL_discriminator": joint_adversary.lambda_val.item(),
        "batch_adv_accuracy": batch_adv_accuracy.item(),
        "adv_accuracy_ema": joint_adversary.ema_acc.item(),
    }
    for key, value in loss_components.items():
        if isinstance(value, torch.Tensor):
            loss_components[key] = value.detach()

    if is_training:
        return total_loss, loss_components
    return total_loss, loss_components, {"cls_norms": cls_norms.detach()}
