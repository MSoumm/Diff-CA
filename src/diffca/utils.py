"""Helpers shared by both stages: seeding, configs, augmentations, profiling, LR schedule."""

import math
import os
import random

import numpy as np
import torch
import yaml
from thop import clever_format, profile
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def print0(to_print, accelerator=None):
    """Print on the main process only."""
    if accelerator is None:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(to_print)
    elif accelerator.is_main_process:
        accelerator.print(to_print)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_transform_from_config(config: list) -> transforms.Compose:
    """Build a torchvision Compose from a list of {name, params, random_apply_p} entries."""
    transform_list = []
    for aug_config in config:
        transform = getattr(transforms, aug_config["name"])(**aug_config.get("params", {}))
        if "random_apply_p" in aug_config:
            transform = transforms.RandomApply([transform], p=aug_config["random_apply_p"])
        transform_list.append(transform)
    return transforms.Compose(transform_list)


def profile_module(model: nn.Module, inputs, device: str = "cpu"):
    """Print the parameter count and MACs (thop) of `model` on `inputs`.

    Note: thop reports MACs; FLOPs = 2 x MACs.
    """
    model = model.to(device)
    model.eval()

    if isinstance(inputs, torch.Tensor):
        inputs = (inputs.to(device),)
    elif isinstance(inputs, (tuple, list)):
        inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs)
    else:
        print("Warning: 'inputs' must be a Tensor or Tuple of Tensors for thop.")
        return

    print(f"\nProfiling {model.__class__.__name__}...")
    try:
        macs, params = profile(model, inputs=inputs, verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
        print("-" * 40)
        print(f"Model: {model.__class__.__name__}")
        print(f"Total Parameters: {params_str}")
        print(f"Total MACs (FLOPs/2): {macs_str}")
        print("-" * 40)
        return macs, params
    except Exception as e:
        print(f"Profiling failed: {e}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters (Fallback): {total_params:,}")
        return 0, total_params


class WarmupCosineDecayScheduler(_LRScheduler):
    """Linear warmup followed by cosine decay to `min_lr`."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            scale = step / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_scale for base_lr in self.base_lrs]


class MultiOptimizer:
    """Steps several optimizers together and exposes their combined `param_groups`."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state)
