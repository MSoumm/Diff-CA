"""Stage 2: train the separator on the extracted Stage 1 tokens, or decode swaps with a trained one.

Training:
    python -m diffca.stage2.train --config_yaml configs/stage2/glasses.yaml
Swaps and interpolations with a trained separator (needs config_file/save_dir, the Stage 1 model):
    python -m diffca.stage2.train --config_yaml configs/stage2/glasses.yaml \
        --resume_from_checkpoint <output_dir>/disentangler_checkpoints/epoch_<n> --test_epoch --generate_all_images --interpolate
"""

import json
import logging
import os
import random
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from diffca.stage1.build import find_checkpoint, load_trained_model
from diffca.stage2.classifiers import build_referee
from diffca.stage2.config import get_config
from diffca.stage2.datasets import (AFHQ_SUBLABELS, AfhqFeatureDataset, BratsFeatureDataset, FeatureAttributeDataset,
                                    create_evaluation_dataloaders)
from diffca.stage2.evaluate import run_evaluation_protocol
from diffca.stage2.generate import generate_all_swapped_images, generate_interpolation_grid
from diffca.stage2.logger import TrainingLogger
from diffca.stage2.losses import cycle_loss_criterion
from diffca.stage2.model import AdversaryMLP, CycleConsistencyWrapper, TokenDisentangler
from diffca.stage2.projections import generate_projections
from diffca.utils import MultiOptimizer, WarmupCosineDecayScheduler, load_config, profile_module, seed_everything

logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

NUM_WORKERS = 24
VAL_BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 64
WARMUP_EPOCHS = 100  # linear LR warmup, then cosine decay to 0.1 x the base LR
DROPOUT = 0.1
ADVERSARY_HIDDEN, ADVERSARY_DROPOUT = 512, 0.2

GREEN, YELLOW, RESET = "\033[92m", "\033[93m", "\033[0m"


class BestMetricsManager:
    """Best value (and epoch) of every evaluation metric; registered with Accelerate so it is checkpointed."""

    CLASS_NAMES = {
        "glasses": ["no_glasses", "eyeglasses", "sunglasses"],
        "smile": ["no_smile", "smile"],
        "headPose": ["straight", "left", "right"],
        "afhq": ["dog", "tabby", "Egyptian_cat", "other_cat"],
        "brats": ["class_0", "class_1"],
    }

    def __init__(self, attribute="glasses"):
        inf = float("inf")
        self.state = {
            "Acc_AddEye": (0.0, -1), "Acc_AddSun": (0.0, -1), "Acc_RemoveEye": (0.0, -1), "Acc_RemoveSun": (0.0, -1),
            "Acc_SunToEye": (0.0, -1), "Acc_EyeToSun": (0.0, -1),
            "rmse_combined": (inf, -1), "Harmonic_Swap": (0.0, -1),
            "harmonic_recon": (0.0, -1), "rmse_score": (0.0, -1),
            "max_rmse": (inf, -1), "min_recon": (0.0, -1), "min_Swap": (0.0, -1),
            "max_Swap_MAE": (inf, -1), "mae_recon_yaw": (inf, -1),
        }
        if attribute == "headPose":
            for name in ("AddLeft", "AddRight", "RemoveLeft", "RemoveRight", "RightToLeft", "LeftToRight"):
                self.state[f"Acc_{name}"] = (0.0, -1)
            for name in ("AddLeft", "AddRight", "RemoveLeft", "RemoveRight", "RightToLeft", "LeftToRight"):
                self.state[f"Mae_{name}"] = (inf, -1)
        for name in self.CLASS_NAMES.get(attribute, ["class_0", "class_1", "class_2"]):
            self.state[f"rmse_{name}"] = (inf, -1)
            self.state[f"acc_{name}"] = (0.0, -1)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict


def _to_serializable(val):
    if isinstance(val, (int, float, bool)) or val is None:
        return val
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    return str(val)


def build_datasets(args):
    """(train, validation) datasets of the attribute, with the features loaded in RAM."""
    if args.attribute == "brats":
        return (BratsFeatureDataset(args.train_pt_folder, args.labels_train, load_features_in_ram=True),
                BratsFeatureDataset(args.test_pt_folder, args.labels_test, load_features_in_ram=True))
    if args.attribute == "afhq":
        return (AfhqFeatureDataset(args.train_pt_folder, sublabels_csv_path=args.sublabels_csv, load_features_in_ram=True),
                AfhqFeatureDataset(args.test_pt_folder, sublabels_csv_path=args.sublabels_csv, load_features_in_ram=True))
    return (FeatureAttributeDataset(args.train_pt_folder, args.json_folder, args.attribute, load_features_in_ram=True),
            FeatureAttributeDataset(args.test_pt_folder, args.json_folder, args.attribute, load_features_in_ram=True))


def print_split_stats(name, dataset, attribute):
    label_counter = Counter(sample["label"].item() for sample in dataset.samples)
    sublabel_counter = Counter(sample["sublabel"].item() for sample in dataset.samples)
    print(f"{YELLOW}{name} label counts: {dict(label_counter)}{RESET}")
    if attribute == "headPose":
        bins = Counter()
        for yaw, count in sublabel_counter.items():
            yaw = float(yaw)
            bin_name = "straight" if -8.0 <= yaw <= 8.0 else "left" if yaw < -12.0 else "right" if yaw > 12.0 else None
            if bin_name:
                bins[bin_name] += count
        print(f"{YELLOW}{name} sublabel counts: {dict(bins)}{RESET}")
    else:
        names = {"afhq": AFHQ_SUBLABELS, "glasses": {0: "no_glasses", 1: "eyeglasses", 2: "sunglasses"}}.get(attribute, {})
        print(f"{YELLOW}{name} sublabel counts: { {names.get(k, k): v for k, v in sublabel_counter.items()} }{RESET}")


def make_loaders(args, train_dataset, val_dataset):
    """Training batches are drawn with replacement, with equal probability for both labels."""
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    class_weights = 1.0 / torch.bincount(train_labels.long()).float()
    sampler = WeightedRandomSampler(weights=class_weights[train_labels.long()], num_samples=len(train_labels),
                                    replacement=True)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def worker_init(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=NUM_WORKERS,
                              generator=generator, worker_init_fn=worker_init)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            generator=generator, worker_init_fn=worker_init)
    return train_loader, val_loader


def nan_hook(module, inputs, output):
    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
        raise ValueError(f"NaN detected in the output of {module.__class__.__name__}")


def init_wandb(args):
    keys = ["attribute", "referee_classifier_path", "model_type", "num_encoder_layers", "num_decoder_layers",
            "learning_rate", "adversary_lr", "lambda_recon", "lambda_norm_pos", "lambda_norm_neg",
            "lambda_cycle_context", "lambda_cycle_cls", "lambda_adv", "max_grad_norm", "ema_decay", "cycle_ema_decay",
            "low_target_acc", "high_target_acc", "low_rank", "json_folder", "output_dir", "config_file", "save_dir",
            "global_step", "sublabels_csv", "labels_train", "labels_test", "test_epoch", "interpolate"]
    wandb.init(project=args.wandb_project, mode=args.wandb_mode, id=args.wandb_id or wandb.util.generate_id(),
               reinit=True, resume="never", config={key: getattr(args, key) for key in keys})


def train(args=None):
    if args is None:
        args = get_config()
    seed_everything(args.seed)

    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="wandb")
    device = accelerator.device

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump({k: _to_serializable(v) for k, v in vars(args).items()}, f, indent=2)
    final_evaluation_dir = None
    if args.generate_all_images:
        final_evaluation_dir = os.path.join(args.output_dir, "final_evaluation")
        os.makedirs(final_evaluation_dir, exist_ok=True)

    if accelerator.is_main_process:
        init_wandb(args)

    # --- Data ---
    train_dataset, val_dataset = build_datasets(args)
    eval_val_subset = Subset(val_dataset, range(len(val_dataset)))
    print(f"{GREEN} Dataset size: {len(train_dataset)}{RESET}")
    print(f"{GREEN} Validation Dataset size: {len(val_dataset)}{RESET}")
    sample_feature, _, _ = train_dataset[0]
    num_tokens, input_dim = sample_feature.shape[0], sample_feature.shape[-1]

    referee_classifier = None
    if args.referee_classifier_path:
        referee_classifier = build_referee(args.attribute, num_tokens=num_tokens).to(device)
        referee_classifier.load_state_dict(torch.load(args.referee_classifier_path, map_location=device))
        referee_classifier.eval()
        if accelerator.is_main_process:
            print(f"\n{GREEN}Successfully loaded referee classifier.{RESET}\n")

    print_split_stats("Train", train_dataset, args.attribute)
    print_split_stats("Val", val_dataset, args.attribute)
    train_loader, val_loader = make_loaders(args, train_dataset, val_dataset)

    # --- Separator, adversary, optimizers ---
    disentangler = TokenDisentangler(
        d_model=input_dim, num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
        dropout=DROPOUT, ema_decay=args.ema_decay, cycle_ema_decay=args.cycle_ema_decay, low_rank=args.low_rank,
    )
    model = CycleConsistencyWrapper(disentangler)
    if accelerator.is_main_process:  # parameter count / MACs (its dummy input is part of the seeded RNG sequence)
        profile_module(model=deepcopy(model), inputs=torch.randn(1, 32, input_dim), device="cpu")
    joint_adversary = AdversaryMLP(num_tokens=32, token_dim=input_dim, hidden_dim=ADVERSARY_HIDDEN,
                                   dropout_rate=ADVERSARY_DROPOUT)

    optimizer_main = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    optimizer_adversary = optim.AdamW(joint_adversary.parameters(), lr=args.adversary_lr, betas=(0.9, 0.99))
    optimizer = MultiOptimizer(optimizer_main, optimizer_adversary)
    steps = len(train_loader)  # before accelerator.prepare, as in the paper runs
    scheduler = WarmupCosineDecayScheduler(optimizer_main, warmup_steps=steps * WARMUP_EPOCHS,
                                           total_steps=args.epochs * steps, min_lr=0.1 * args.learning_rate)
    adv_scheduler = WarmupCosineDecayScheduler(optimizer_adversary, warmup_steps=steps * WARMUP_EPOCHS,
                                               total_steps=args.epochs * steps, min_lr=0.1 * args.adversary_lr)

    # --- Stage 1 model, to decode tokens to images ---
    stage1_model, stage1_config, scaling_factor = None, None, None
    if args.config_file and args.save_dir and (not args.no_generate_images or args.test_epoch):
        if accelerator.is_main_process:
            print("--- Setting up AEModel for evaluation ---")
        stage1_config = load_config(args.config_file)
        checkpoint_dir = find_checkpoint(args.save_dir, args.global_step)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No intermediate checkpoint-* folder found in {args.save_dir}")
        # A separate Accelerator, so that the Stage 1 checkpoint does not register with the training one.
        # (Loading it also restores its RNG states; every epoch reseeds anyway.)
        stage1_model, _, scaling_factor, _ = load_trained_model(
            stage1_config, checkpoint_dir, Accelerator(mixed_precision=args.mixed_precision), use_compile=True)
        print(f"\n{GREEN}Loaded AEModel from {checkpoint_dir} for evaluation.{RESET}\n")
    diffusion_model = stage1_model.diffusion_model if stage1_model is not None else None
    vae = stage1_model.vae if stage1_model is not None else None

    model, joint_adversary, optimizer, train_loader, val_loader, scheduler, adv_scheduler = accelerator.prepare(
        model, joint_adversary, optimizer, train_loader, val_loader, scheduler, adv_scheduler
    )
    steps_per_epoch = len(train_loader)  # after prepare: per process
    best_metrics_manager = BestMetricsManager(args.attribute)
    accelerator.register_for_checkpointing(best_metrics_manager)

    # --- Resume ---
    start_epoch = 0
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        optimizer.param_groups[1]["lr"] = args.adversary_lr  # until the next scheduler step
        try:
            start_epoch = int(args.resume_from_checkpoint.split("_")[-1]) + 1  # ".../epoch_10" -> 11
            if accelerator.is_main_process:
                print(f"Resuming training from epoch {start_epoch}")
        except (ValueError, IndexError):
            if accelerator.is_main_process:
                print(f"Could not parse epoch from checkpoint path: {args.resume_from_checkpoint}. Starting from epoch 0.")
            start_epoch = 0

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.model.decoder.register_forward_hook(nan_hook)
    logger = TrainingLogger(accelerator, steps_per_epoch)
    loss_kwargs = vars(args)

    # With --test_epoch, evaluate once even from the last checkpoint (start_epoch == epochs).
    end_epoch = max(args.epochs, start_epoch + 1) if args.test_epoch else args.epochs
    try:
        for epoch in range(start_epoch, end_epoch):
            seed_everything(args.seed + epoch)  # reproducible when resuming
            model.train()
            joint_adversary.train()
            logger.reset_epoch_states()

            # --- Training ---
            if not args.test_epoch:
                for step, (features, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]")):
                    optimizer.zero_grad()
                    features, labels = features.to(device), labels.to(device)
                    output = model(features, labels=labels, cycle=True)
                    loss, components = cycle_loss_criterion(
                        output, joint_adversary=joint_adversary, total_step=epoch * steps_per_epoch + step,
                        is_training=True, steps_per_epoch=steps_per_epoch, total_epochs=args.epochs, **loss_kwargs,
                    )
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(list(model.parameters()) + list(joint_adversary.parameters()),
                                                    args.max_grad_norm)
                    components["main_lr"] = optimizer.param_groups[0]["lr"]
                    components["adversary_lr"] = optimizer.param_groups[1]["lr"]
                    optimizer.step()
                    scheduler.step()
                    adv_scheduler.step()
                    unwrapped_model.update_ema()
                    unwrapped_model.update_cycle_ema()

                    pos_mask, neg_mask = labels == 1, labels == 0
                    logger.log_step(components, epoch, step, pos_mask, neg_mask)
                    logger.aggregate_losses(components, features.size(0), pos_mask, neg_mask, is_training=True)
                logger.log_epoch(epoch, is_training=True)

            # --- Validation ---
            model.eval()
            joint_adversary.eval()
            all_labels, all_sublabels, all_cls, all_context_tokens = [], [], [], []
            correct_cls = total_samples = 0
            tp = tn = fp = fn = 0
            with torch.no_grad():
                for features, labels, sublabels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Validation]"):
                    features, labels = features.to(device), labels.to(device)
                    output = model(features, labels, cycle=True)
                    _, components, _ = cycle_loss_criterion(
                        output, joint_adversary=joint_adversary, total_step=(epoch + 1) * steps_per_epoch,
                        is_training=False, steps_per_epoch=steps_per_epoch, total_epochs=args.epochs, **loss_kwargs,
                    )
                    pos_mask, neg_mask = labels == 1, labels == 0
                    logger.aggregate_losses(components, features.size(0), pos_mask, neg_mask, is_training=False)

                    std_out = output["standard_outputs"]
                    all_labels.append(labels.cpu())
                    all_sublabels.append(sublabels.cpu())
                    all_cls.append(std_out["cls"].cpu())
                    all_context_tokens.append(std_out["content_tokens"].mean(dim=1).cpu())

                    # Untrained linear head on h (see TokenDisentangler): printed for reference only.
                    correct_cls += ((torch.sigmoid(std_out["predicted_y"]) >= 0.5) == labels).sum().item()
                    total_samples += labels.size(0)

                    # Adversary on the common tokens: how much label information they still carry.
                    predicted = joint_adversary(std_out["content_tokens"]).squeeze(-1) >= 0
                    tp += ((predicted == 1) & (labels == 1)).sum().item()
                    tn += ((predicted == 0) & (labels == 0)).sum().item()
                    fp += ((predicted == 1) & (labels == 0)).sum().item()
                    fn += ((predicted == 0) & (labels == 1)).sum().item()
            logger.log_epoch(epoch, is_training=False)

            if accelerator.is_main_process:
                log_dict = {}
                eps = 1e-8
                cls_accuracy = correct_cls / total_samples if total_samples > 0 else 0
                context_accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
                recall_positive = tp / (tp + fn + eps)
                recall_negative = tn / (tn + fp + eps)
                log_dict["val/f1_score_context"] = 2 * (recall_positive * recall_negative) / (recall_positive + recall_negative + eps)
                log_dict["val/balanced_accuracy_context"] = (recall_positive + recall_negative) / 2.0
                log_dict["val/cls_accuracy"] = cls_accuracy
                log_dict["val/context_accuracy"] = context_accuracy
                print(f"Validation Accuracy (CLS Token): {cls_accuracy:.4f}")
                print(f"Validation Accuracy (Context Tokens): {context_accuracy:.4f}")
                print(f"Balanced Accuracy (Context Tokens): {log_dict['val/balanced_accuracy_context']:.4f}")
                print(f"F1 Score (Context Tokens): {log_dict['val/f1_score_context']:.4f}")

                # --- Evaluation protocol, swaps, projections ---
                eval_loaders = create_evaluation_dataloaders(eval_val_subset, val_dataset, EVAL_BATCH_SIZE,
                                                             attribute=args.attribute)
                eval_log_dict, best_metrics_manager.state = run_evaluation_protocol(
                    unwrapped_model, eval_loaders, referee_classifier, device, best_metrics_manager.state, epoch,
                    use_ema=True, attribute=args.attribute,
                    pair_names_dir=final_evaluation_dir if args.generate_all_images else None,
                )
                log_dict.update(eval_log_dict)
                if args.attribute == "headPose":
                    best_mae, best_epoch = best_metrics_manager.state.get("max_Swap_MAE", (float("inf"), -1))
                    print(f"Best max yaw MAE so far: {best_mae:.4f} at epoch {best_epoch}" if best_epoch >= 0
                          else "Best max yaw MAE so far: N/A")
                    log_dict["eval/best_max_yaw_mae"] = best_mae

                if args.interpolate:
                    generate_interpolation_grid(
                        unwrapped_model, eval_loaders, diffusion_model, vae, stage1_config, device, scaling_factor,
                        os.path.join(args.output_dir, str(epoch), "interpolations"), use_ema=True,
                        max_grids=100, num_negatives=1, negative_class=1, positive_class=1, num_steps=5,
                    )
                if args.generate_all_images:
                    generate_all_swapped_images(
                        unwrapped_model, eval_loaders, diffusion_model, vae, stage1_config, device, scaling_factor,
                        os.path.join(final_evaluation_dir, "swapped_images"), use_ema=True,
                    )
                for method, enabled in (("umap", args.umap), ("pca", args.pca)):
                    if enabled:
                        generate_projections(method, all_labels, all_sublabels, all_cls, all_context_tokens, epoch,
                                             args.output_dir, args.seed)

                if log_dict:
                    wandb.log(log_dict, step=(epoch + 1) * steps_per_epoch)

            if args.test_epoch:
                break
            epoch_dir = os.path.join(args.output_dir, "disentangler_checkpoints", f"epoch_{epoch}")
            accelerator.save_state(output_dir=epoch_dir)
            if accelerator.is_main_process:
                print(f"Checkpoint for epoch {epoch} saved to {epoch_dir}")
    finally:
        if wandb.run is not None:
            wandb.finish()

    print("Training complete.")
    return best_metrics_manager.state


if __name__ == "__main__":
    train()
