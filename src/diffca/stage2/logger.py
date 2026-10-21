"""Per-step and per-epoch logging of the separator losses (stdout + wandb), split by class where relevant."""

from collections import defaultdict

import torch
import wandb

BATCH_LEVEL_KEYS = ["total_loss", "loss_recon", "loss_cycle_context", "loss_cycle_cls", "loss_cls", "loss_adversarial"]
PER_SAMPLE_KEYS = {
    "reconstruction_loss": "reconstruction_loss_per_sample",
    "loss_cls": "loss_cls_per_sample",
    "loss_cycle_cls": "loss_cycle_cls_per_sample",
    "loss_adversarial": "loss_adversarial_per_sample",
    "loss_cycle_context": "loss_cycle_context_per_sample",
}
METRIC_KEYS = ["lambda_GRL_discriminator", "batch_adv_accuracy", "adv_accuracy_ema"]
STEP_SCALAR_KEYS = BATCH_LEVEL_KEYS + METRIC_KEYS + ["main_lr", "adversary_lr"]
NORM_KEYS = [("positive_mean_norms", True), ("negative_mean_norms", False)]  # (key, averaged over Y=1 samples)


def _scalar(value):
    return value.item() if isinstance(value, torch.Tensor) else value


def _truncate(mask, tensor):
    """The cycle losses only cover an even number of samples: truncate the mask to match."""
    if mask.shape[0] == tensor.shape[0]:
        return mask
    return mask[:min(mask.shape[0], tensor.shape[0])]


class TrainingLogger:
    def __init__(self, accelerator, train_loader_len):
        self.accelerator = accelerator
        self.train_loader_len = train_loader_len
        self.reset_epoch_states()

    def reset_epoch_states(self):
        self.agg = {True: defaultdict(float), False: defaultdict(float)}  # keyed by is_training
        self.pos_count = {True: 0, False: 0}
        self.neg_count = {True: 0, False: 0}

    def log_step(self, components, epoch, step, pos_mask, neg_mask):
        if not self.accelerator.is_main_process:
            return
        log_dict = {f"train/step/{key}": _scalar(components[key]) for key in STEP_SCALAR_KEYS if key in components}
        for log_key, tensor_key in PER_SAMPLE_KEYS.items():
            tensor = components.get(tensor_key)
            if tensor is None:
                continue
            pos, neg = _truncate(pos_mask, tensor), _truncate(neg_mask, tensor)
            if pos.any():
                log_dict[f"train/step/{log_key}_pos"] = tensor[pos].mean().item()
            if neg.any():
                log_dict[f"train/step/{log_key}_neg"] = tensor[neg].mean().item()
        wandb.log(log_dict, step=epoch * self.train_loader_len + step)

    def aggregate_losses(self, components, batch_size, pos_mask, neg_mask, is_training):
        agg = self.agg[is_training]
        num_pos, num_neg = pos_mask.sum().item(), neg_mask.sum().item()
        self.pos_count[is_training] += num_pos
        self.neg_count[is_training] += num_neg

        for key in BATCH_LEVEL_KEYS:
            if key in components:
                agg[key] += components[key].item() * batch_size

        for log_key, tensor_key in PER_SAMPLE_KEYS.items():
            tensor = components.get(tensor_key)
            if tensor is None:
                continue
            pos, neg = pos_mask, neg_mask
            if tensor.shape[0] != pos_mask.shape[0]:
                min_len = min(tensor.shape[0], pos_mask.shape[0])
                pos, neg, tensor = pos_mask[:min_len], neg_mask[:min_len], tensor[:min_len]
            agg[f"{log_key}_pos"] += tensor[pos].sum().item()
            agg[f"{log_key}_neg"] += tensor[neg].sum().item()

        for key in METRIC_KEYS:
            if key in components:
                agg[key] += _scalar(components[key]) * batch_size

        for key, positive in NORM_KEYS:
            count = num_pos if positive else num_neg
            if key in components and count > 0:
                agg[key] += components[key].item() * count

    def log_epoch(self, epoch, is_training):
        if not self.accelerator.is_main_process:
            return
        mode = "train" if is_training else "val"
        agg = self.agg[is_training]
        pos_count, neg_count = self.pos_count[is_training], self.neg_count[is_training]
        num_samples = pos_count + neg_count

        print(f"--- Epoch {epoch + 1} {mode.capitalize()} Loss Components (Per Sample) ---")
        log_dict = {}

        def print_and_log(name, value, is_metric=False):
            print(f"  - {'Metric' if is_metric else 'Average'} {name}: {value:.4f}")
            log_dict[f"{mode}/{name}"] = value

        for key in BATCH_LEVEL_KEYS:
            if key in agg:
                print_and_log(key, agg[key] / num_samples)
        for key in PER_SAMPLE_KEYS:
            if f"{key}_pos" in agg and pos_count > 0:
                print_and_log(f"{key}_pos", agg[f"{key}_pos"] / pos_count)
            if f"{key}_neg" in agg and neg_count > 0:
                print_and_log(f"{key}_neg", agg[f"{key}_neg"] / neg_count)
        for key in METRIC_KEYS:
            if key in agg:
                print_and_log(key, agg[key] / num_samples, is_metric=True)
        for key, positive in NORM_KEYS:
            count = pos_count if positive else neg_count
            if key in agg and count > 0:
                print_and_log(key, agg[key] / count, is_metric=True)

        wandb.log(log_dict, step=(epoch + 1) * self.train_loader_len)
        if is_training:
            self.reset_epoch_states()
