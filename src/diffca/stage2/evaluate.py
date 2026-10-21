"""Evaluation protocol of the separator, in token space, with the referee classifier.

S_recon: per-class reconstruction RMSE, and how often the referee gives the same class to Z and Z_hat.
S_swap:  the salient code of a sample of class j is decoded with the common tokens of a sample of class i;
         the swap succeeds when the referee predicts class j (removal: class 0). For head pose, the yaw
         error of the swap is also reported.
"""

import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

CLASS_NAMES = {
    "glasses": {0: "no_glasses", 1: "eyeglasses", 2: "sunglasses"},
    "headPose": {0: "straight", 1: "left", 2: "right"},
}


def yaw_to_class(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw bins of the referee: 0 straight (|yaw| <= 10), 1 left (< -10), 2 right (> 10)."""
    classes = torch.full_like(yaw, -1, dtype=torch.long)
    classes[(yaw >= -10.0) & (yaw <= 10.0)] = 0
    classes[yaw < -10.0] = 1
    classes[yaw > 10.0] = 2
    return classes


def normalize_attribute(attribute):
    if isinstance(attribute, str) and attribute.lower() == "headpose":
        return "headPose"
    return attribute


def class_name_map(attribute, class_ids=None):
    if attribute in CLASS_NAMES:
        return dict(CLASS_NAMES[attribute])
    if class_ids is None:
        return {}
    return {int(class_id): f"class_{int(class_id)}" for class_id in class_ids}


def default_class_names(attribute):
    if attribute in CLASS_NAMES:
        return list(CLASS_NAMES[attribute].values())
    return ["class_0", "class_1", "class_2"]


def predict_classes(referee_classifier, features, attribute):
    """(raw prediction, class id): yaw and yaw bin for head pose, argmax class otherwise."""
    if attribute == "headPose":
        preds = referee_classifier(features)
        if preds.ndim > 1:
            preds = preds.squeeze(-1)
        return preds, yaw_to_class(preds)
    class_ids = torch.argmax(referee_classifier(features), dim=1)
    return class_ids, class_ids


# ---------------------------------------------------------------------------
# File names of the evaluation pairs
# ---------------------------------------------------------------------------

def _resolve_sample_name(dataset, idx):
    base_dataset, resolved_idx = dataset, idx
    while hasattr(base_dataset, "dataset") and hasattr(base_dataset, "indices"):
        resolved_idx = base_dataset.indices[resolved_idx]
        base_dataset = base_dataset.dataset
    if not hasattr(base_dataset, "samples"):
        return None
    return Path(base_dataset.samples[resolved_idx]["feature_file"]).stem


def collect_pair_names(dataset, batch_indices):
    """[(group, name_a, name_b)] for a batch of a pair loader; `group` indexes the parts of a ConcatDataset."""
    if batch_indices is None:
        return []
    batch_indices = batch_indices.tolist() if torch.is_tensor(batch_indices) else list(batch_indices)
    if isinstance(dataset, ConcatDataset):
        entries = []
        for idx in batch_indices:
            dataset_idx = bisect_right(dataset.cumulative_sizes, idx)
            local_idx = idx - (0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1])
            for _, name1, name2 in collect_pair_names(dataset.datasets[dataset_idx], [local_idx]):
                entries.append((dataset_idx, name1, name2))
        return entries
    if hasattr(dataset, "indices1") and hasattr(dataset, "indices2"):
        entries = []
        for idx in batch_indices:
            name1 = _resolve_sample_name(dataset.dataset, dataset.indices1[idx % dataset.i1_sizes])
            name2 = _resolve_sample_name(dataset.dataset, dataset.indices2[idx % dataset.i2_sizes])
            if name1 is not None and name2 is not None:
                entries.append((0, name1, name2))
        return entries
    return []


class _PairNameWriter:
    """Writes the names of the evaluated pairs to <dir>/<i>_<j>.txt (used with --generate_all_images)."""

    def __init__(self, pair_names_dir, keys):
        os.makedirs(pair_names_dir, exist_ok=True)
        self.files = {key: open(os.path.join(pair_names_dir, f"{key}.txt"), "w") for key in keys}

    def __bool__(self):
        return bool(self.files)

    def write(self, loader, batch_indices, keys_by_group):
        for group_idx, name1, name2 in collect_pair_names(loader.dataset, batch_indices):
            key = keys_by_group.get(group_idx)
            if key:
                self.files[key].write(f"{name1} {name2}\n")

    def close(self):
        for handle in self.files.values():
            handle.close()


def _batch_indices(loader, writer):
    """Iterator over the sample indices of each batch, only needed when writing pair names."""
    return iter(loader.batch_sampler) if writer else None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def _reconstruction_scores(model, loader, referee_classifier, attribute, class_id_to_name, device, use_ema):
    rmses = defaultdict(list)
    recon_acc_stats = defaultdict(lambda: {"matches": 0, "total": 0})
    recon_mae_values = []
    with torch.no_grad():
        for features, _, sublabels in tqdm(loader, desc="Evaluation: Reconstruction"):
            features = features.to(device)
            reconstructed_tokens = model(features, use_ema=use_ema)["standard_outputs"]["reconstructed_tokens"]
            rmse = torch.sqrt(torch.mean((features - reconstructed_tokens)**2, dim=[1, 2]))

            if referee_classifier:
                if attribute == "headPose":
                    preds_recon, preds_recon_class = predict_classes(referee_classifier, reconstructed_tokens, attribute)
                else:
                    preds_original_class, _ = predict_classes(referee_classifier, features, attribute)
                    preds_recon_class, _ = predict_classes(referee_classifier, reconstructed_tokens, attribute)
                    matches = preds_original_class == preds_recon_class

            for i, sublabel in enumerate(sublabels):
                sublabel_item = sublabel.item()
                if attribute == "headPose":
                    class_id = yaw_to_class(torch.tensor(sublabel_item)).item()
                else:
                    class_id = int(sublabel_item)
                class_name = class_id_to_name.setdefault(class_id, f"class_{class_id}")

                rmses[class_name].append(rmse[i].item())
                if referee_classifier:
                    recon_acc_stats[class_name]["total"] += 1
                    if attribute == "headPose":
                        if preds_recon_class[i].item() == class_id:
                            recon_acc_stats[class_name]["matches"] += 1
                        recon_mae_values.append(abs(preds_recon[i].item() - sublabel_item))
                    elif matches[i]:
                        recon_acc_stats[class_name]["matches"] += 1
    return rmses, recon_acc_stats, recon_mae_values


class _SwapCounter:
    """Accuracy (and, for head pose, absolute yaw error) of named swap directions, in insertion order."""

    def __init__(self):
        self.correct, self.total, self.errors = {}, {}, {}

    def add(self, name, correct, total, errors=None):
        self.correct[name] = self.correct.get(name, 0) + correct
        self.total[name] = self.total.get(name, 0) + total
        if errors is not None:
            self.errors.setdefault(name, []).extend(errors)

    def accuracy(self, name):
        return self.correct.get(name, 0) / self.total[name] if self.total.get(name, 0) > 0 else 0

    def mae(self, name):
        return float(np.mean(self.errors[name])) if self.errors.get(name) else 0.0


def _head_pose_swaps(encode, swap, referee, loaders, writer, device):
    """Add/remove a left or right pose (straight <-> turned pairs), and left <-> right."""
    counter = _SwapCounter()
    zero_one_iter = _batch_indices(loaders["zero_one"], writer)
    for (X_straight, _, yaw_straight), (X_pose, _, yaw_pose) in tqdm(loaders["zero_one"], disable=True):
        if zero_one_iter is not None:
            writer.write(loaders["zero_one"], next(zero_one_iter, None), {0: "0_1", 1: "0_2"})
        cls_straight, context_straight = encode(X_straight)
        cls_pose, context_pose = encode(X_pose)
        yaw_straight, yaw_pose = yaw_straight.to(device), yaw_pose.to(device)
        preds_add = referee(swap(context_straight, cls_pose)).squeeze(-1)
        preds_remove = referee(swap(context_pose, cls_straight)).squeeze(-1)

        target_class = yaw_to_class(yaw_pose)
        for side, class_id in (("Left", 1), ("Right", 2)):
            mask = target_class == class_id
            if mask.any():
                n = mask.sum().item()
                counter.add(f"Add{side}", (yaw_to_class(preds_add[mask]) == class_id).sum().item(), n,
                            (preds_add[mask] - yaw_pose[mask]).abs().cpu().tolist())
                counter.add(f"Remove{side}", (yaw_to_class(preds_remove[mask]) == 0).sum().item(), n,
                            (preds_remove[mask] - yaw_straight[mask]).abs().cpu().tolist())

    style_iter = _batch_indices(loaders["style"], writer)
    for (X_left, _, yaw_left), (X_right, _, yaw_right) in tqdm(loaders["style"], disable=True):
        if style_iter is not None:
            writer.write(loaders["style"], next(style_iter, None), {0: "1_2"})
        cls_left, context_left = encode(X_left)
        cls_right, context_right = encode(X_right)
        yaw_left, yaw_right = yaw_left.to(device), yaw_right.to(device)
        preds_r2l = referee(swap(context_right, cls_left)).squeeze(-1)
        counter.add("RightToLeft", (yaw_to_class(preds_r2l) == 1).sum().item(), preds_r2l.size(0),
                    (preds_r2l - yaw_left).abs().cpu().tolist())
        preds_l2r = referee(swap(context_left, cls_right)).squeeze(-1)
        counter.add("LeftToRight", (yaw_to_class(preds_l2r) == 2).sum().item(), preds_l2r.size(0),
                    (preds_l2r - yaw_right).abs().cpu().tolist())

    order = ["AddLeft", "AddRight", "RemoveLeft", "RemoveRight", "RightToLeft", "LeftToRight"]
    accuracies = {f"Acc_{name}": counter.accuracy(name) for name in order}
    counts = {f"Acc_{name}": counter.total.get(name, 0) for name in order}
    maes = {f"Mae_{name}": counter.mae(name) for name in order}
    return accuracies, counts, maes


def _glasses_swaps(encode, swap, referee, loaders, writer):
    """Add/remove eyeglasses or sunglasses (no glasses <-> glasses pairs), and eyeglasses <-> sunglasses."""
    counter = _SwapCounter()
    zero_one_iter = _batch_indices(loaders["zero_one"], writer)
    for (X_no, _, _), (X_glass, _, sublabel_glass) in tqdm(loaders["zero_one"], disable=True):
        if zero_one_iter is not None:
            writer.write(loaders["zero_one"], next(zero_one_iter, None), {0: "0_1", 1: "0_2"})
        cls_no, context_no = encode(X_no)
        cls_glass, context_glass = encode(X_glass)
        preds_add = torch.argmax(referee(swap(context_no, cls_glass)), dim=1)
        preds_remove = torch.argmax(referee(swap(context_glass, cls_no)), dim=1)
        for kind, class_id in (("Eye", 1), ("Sun", 2)):
            mask = sublabel_glass == class_id
            if mask.any():
                n = mask.sum().item()
                counter.add(f"Add{kind}", (preds_add[mask] == class_id).sum().item(), n)
                counter.add(f"Remove{kind}", (preds_remove[mask] == 0).sum().item(), n)

    style_iter = _batch_indices(loaders["style"], writer)
    for (X_eye, _, _), (X_sun, _, _) in tqdm(loaders["style"], disable=True):
        if style_iter is not None:
            writer.write(loaders["style"], next(style_iter, None), {0: "1_2"})
        cls_eye, context_eye = encode(X_eye)
        cls_sun, context_sun = encode(X_sun)
        preds_s2e = torch.argmax(referee(swap(context_sun, cls_eye)), dim=1)
        counter.add("SunToEye", (preds_s2e == 1).sum().item(), preds_s2e.size(0))
        preds_e2s = torch.argmax(referee(swap(context_eye, cls_sun)), dim=1)
        counter.add("EyeToSun", (preds_e2s == 2).sum().item(), preds_e2s.size(0))

    order = ["AddEye", "AddSun", "RemoveEye", "RemoveSun", "SunToEye", "EyeToSun"]
    accuracies = {f"Acc_{name}": counter.accuracy(name) for name in order}
    counts = {f"Acc_{name}": counter.total.get(name, 0) for name in order}
    return accuracies, counts, {}


def _pairwise_swaps(encode, swap, referee, loaders, writer, attribute):
    """For every pair of classes (i, j): i -> j and j -> i."""
    accuracies, counts = {}, {}
    for pair_key, pair_loader in loaders.get("pairs", {}).items():
        try:
            class_i, class_j = (int(part) for part in pair_key.split("_", 1))
        except ValueError:
            continue
        counter = _SwapCounter()
        pair_iter = _batch_indices(pair_loader, writer)
        for (X_i, _, _), (X_j, _, _) in tqdm(pair_loader, disable=True):
            if pair_iter is not None:
                writer.write(pair_loader, next(pair_iter, None), {0: pair_key})
            cls_i, context_i = encode(X_i)
            cls_j, context_j = encode(X_j)
            preds_i_to_j, _ = predict_classes(referee, swap(context_i, cls_j), attribute)
            counter.add("i_to_j", (preds_i_to_j == class_j).sum().item(), preds_i_to_j.size(0))
            preds_j_to_i, _ = predict_classes(referee, swap(context_j, cls_i), attribute)
            counter.add("j_to_i", (preds_j_to_i == class_i).sum().item(), preds_j_to_i.size(0))
        for direction, name in (("i_to_j", f"Acc_{class_i}_to_{class_j}"), ("j_to_i", f"Acc_{class_j}_to_{class_i}")):
            if counter.total.get(direction, 0) > 0:
                accuracies[name] = counter.accuracy(direction)
                counts[name] = counter.total[direction]
    return accuracies, counts, {}


def run_evaluation_protocol(model, loaders, referee_classifier, device, best_metrics, epoch, use_ema=False,
                            attribute="glasses", pair_names_dir=None):
    """Compute S_recon and S_swap, update `best_metrics` ({name: (best value, epoch)}), return (log_dict, best_metrics)."""
    attribute = normalize_attribute(attribute)
    model.eval()
    if referee_classifier:
        referee_classifier.eval()

    log_dict = {}
    class_ids = loaders.get("class_ids")
    class_id_to_name = class_name_map(attribute, class_ids)
    three_classes = class_ids == [0, 1, 2] and "zero_one" in loaders and "style" in loaders
    allow_swap = attribute == "headPose" or ("pairs" in loaders and loaders["pairs"])

    writer = None
    if pair_names_dir and allow_swap:
        keys = ["0_1", "0_2", "1_2"] if attribute == "headPose" or three_classes else list(loaders.get("pairs", {}))
        writer = _PairNameWriter(pair_names_dir, keys)

    # --- S_recon ---
    rmses, recon_acc_stats, recon_mae_values = _reconstruction_scores(
        model, loaders["full"], referee_classifier, attribute, class_id_to_name, device, use_ema)

    if class_ids:
        class_names = [class_id_to_name.get(class_id, f"class_{class_id}") for class_id in class_ids]
    elif rmses:
        class_names = sorted(rmses.keys())
    else:
        class_names = default_class_names(attribute)
    rmse_by_class = [np.mean(rmses[name]) if rmses[name] else 0 for name in class_names]
    acc_by_class = [recon_acc_stats[name]["matches"] / recon_acc_stats[name]["total"]
                    if recon_acc_stats[name]["total"] > 0 else 0 for name in class_names]

    rmse_combined = float(np.mean(rmse_by_class)) if rmse_by_class else 0.0
    ref_rmse = 0.5
    max_rmse = max(rmse_by_class + [ref_rmse]) if rmse_by_class else ref_rmse
    rmse_score = 1 / (1 + rmse_combined / ref_rmse)
    if acc_by_class:
        harmonic_recon = len(acc_by_class) / sum(1 / (acc + 1e-8) for acc in acc_by_class)
        min_recon = min(acc_by_class)
    else:
        harmonic_recon = min_recon = 0.0

    for name, rmse_val, acc_val in zip(class_names, rmse_by_class, acc_by_class):
        best_metrics[f"rmse_{name}"] = min(best_metrics.get(f"rmse_{name}", (float("inf"), -1)), (rmse_val, epoch))
        best_metrics[f"acc_{name}"] = max(best_metrics.get(f"acc_{name}", (0.0, -1)), (acc_val, epoch))
    best_metrics["rmse_combined"] = min(best_metrics["rmse_combined"], (rmse_combined, epoch))
    best_metrics["max_rmse"] = min(best_metrics["max_rmse"], (max_rmse, epoch))
    best_metrics["harmonic_recon"] = max(best_metrics["harmonic_recon"], (harmonic_recon, epoch))
    best_metrics["min_recon"] = max(best_metrics["min_recon"], (min_recon, epoch))
    best_metrics["rmse_score"] = max(best_metrics["rmse_score"], (rmse_score, epoch))

    print("\n--- Evaluation Protocol Metrics ---")
    print("S_recon:")
    for name, rmse_val, acc_val in zip(class_names, rmse_by_class, acc_by_class):
        best_rmse, best_acc = best_metrics[f"rmse_{name}"], best_metrics[f"acc_{name}"]
        print(f"  - {name}: RMSE {rmse_val:.4f} (best: {best_rmse[0]:.4f} at epoch {best_rmse[1]}), "
              f"Accuracy {acc_val:.4f} (best: {best_acc[0]:.4f} at epoch {best_acc[1]})")
    print(f"  - Combined (mean): {rmse_combined:.4f}")
    print(f"  - RMSE-based Score: {rmse_score:.4f} (best: {best_metrics['rmse_score'][0]:.4f} at epoch {best_metrics['rmse_score'][1]})")
    if attribute == "headPose":
        mae_recon = float(np.mean(recon_mae_values)) if recon_mae_values else 0.0
        best_metrics["mae_recon_yaw"] = min(best_metrics["mae_recon_yaw"], (mae_recon, epoch))
        print(f"  - Reconstruction Yaw MAE: {mae_recon:.4f} (best: {best_metrics['mae_recon_yaw'][0]:.4f} at epoch {best_metrics['mae_recon_yaw'][1]})")

    for name, rmse_val, acc_val in zip(class_names, rmse_by_class, acc_by_class):
        log_dict[f"eval/S_recon_rmse_{name}"] = rmse_val
        log_dict[f"eval/S_recon_acc_{name}"] = acc_val
    log_dict["eval/S_recon_rmse_combined"] = rmse_combined
    log_dict["eval/S_recon_harmonic"] = harmonic_recon
    log_dict["eval/S_recon_rmse_score"] = rmse_score
    if attribute == "headPose":
        log_dict["eval/S_recon_mae_yaw"] = float(np.mean(recon_mae_values)) if recon_mae_values else 0.0

    if not referee_classifier:
        print("Referee classifier not available. Skipping S_swap calculation.")
        if writer:
            writer.close()
        return log_dict, best_metrics
    if not allow_swap:
        print("Swap evaluation not defined for this attribute. Skipping S_swap calculation.")
        return log_dict, best_metrics

    # --- S_swap ---
    def encode(X):
        out = model(X.to(device), use_ema=use_ema)["standard_outputs"]
        return out["cls"], out["content_tokens"]

    def swap(content_tokens, cls):
        return model.model.decoder(content_tokens, cls.unsqueeze(1), use_ema=use_ema)

    with torch.no_grad():
        if attribute == "headPose":
            accuracies, accuracy_counts, maes = _head_pose_swaps(encode, swap, referee_classifier, loaders, writer, device)
        elif attribute == "glasses" and three_classes:
            accuracies, accuracy_counts, maes = _glasses_swaps(encode, swap, referee_classifier, loaders, writer)
        else:
            accuracies, accuracy_counts, maes = _pairwise_swaps(encode, swap, referee_classifier, loaders, writer, attribute)

    if accuracies:
        harmonic_swap = len(accuracies) / sum((1 / (acc + 1e-8)) for acc in accuracies.values())
        accuracies["Harmonic_Swap"] = harmonic_swap
        best_metrics["Harmonic_Swap"] = max(best_metrics["Harmonic_Swap"], (harmonic_swap, epoch))
        accuracies["min_Swap"] = min(accuracies.values())

        print("S_swap (Accuracy):")
        for name, acc in accuracies.items():
            best_metrics[name] = max(best_metrics.get(name, (0.0, -1)), (acc, epoch))
            print(f"  - {name}: {acc:.4f} (best: {best_metrics[name][0]:.4f} at epoch {best_metrics[name][1]}) "
                  f"[n={accuracy_counts.get(name, 0)}]")
            log_dict[f"eval/{name}"] = acc
        print("-" * 30)

    if maes:
        max_swap_mae = max(maes.values())
        best_metrics["max_Swap_MAE"] = min(best_metrics.get("max_Swap_MAE", (float("inf"), -1)), (max_swap_mae, epoch))
        log_dict["eval/max_Swap_MAE"] = max_swap_mae
        print(f"Max swap MAE: {max_swap_mae:.4f} (best: {best_metrics['max_Swap_MAE'][0]:.4f} at epoch {best_metrics['max_Swap_MAE'][1]})")
        print("S_swap (Yaw MAE):")
        for name, mae in maes.items():
            best_metrics[name] = min(best_metrics.get(name, (float("inf"), -1)), (mae, epoch))
            print(f"  - {name}: {mae:.4f} (best: {best_metrics[name][0]:.4f} at epoch {best_metrics[name][1]})")
            log_dict[f"eval/{name}"] = mae
        print("-" * 30)

    if writer:
        writer.close()
    return log_dict, best_metrics
