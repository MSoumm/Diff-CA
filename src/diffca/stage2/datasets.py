"""Stage 2 datasets: the extracted conditioning tokens (`<image>.pt`, from `stage1.extract_features`) with labels.

Every dataset returns (tokens, label, sublabel). `label` is the binary background (0) / target (1) label used
to train the separator; `sublabel` is the finer class used for evaluation (glasses type, yaw, cat breed, ...).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

# Glasses (FFHQ-Features "glasses" field). Swimming goggles are excluded.
GLASSES_SUBLABELS = {"ReadingGlasses": 1, "Sunglasses": 2}
# Head pose: |yaw| <= 5 is the background class, |yaw| > 12 the target class, 5 < |yaw| <= 12 is excluded.
YAW_BACKGROUND, YAW_EXCLUDED_MAX = 5.0, 12.0
# BraTS: axial slices strictly between 85 and 100.
BRATS_SLICES = (85, 100)
AFHQ_SUBLABELS = {0: "dog", 1: "tabby", 2: "Egyptian_cat", 3: "other_cat"}


def _safe_load_feature_tensor(pt_path: str):
    try:
        return torch.load(pt_path, map_location="cpu"), None
    except Exception as exc:
        return None, str(exc)


class _FeatureDataset(Dataset):
    """Shared storage: `self.samples` is a list of dicts with `feature_file`, `label`, `sublabel` (and optionally
    the loaded `feature_tensor`)."""

    def __init__(self, pt_folder_path: str, load_features_in_ram: bool, preload_workers: int):
        super().__init__()
        self.pt_folder_path = pt_folder_path
        self.load_features_in_ram = load_features_in_ram
        self.preload_workers = preload_workers
        self.samples = []

    def _feature_files(self):
        return sorted(f for f in os.listdir(self.pt_folder_path) if f.endswith(".pt"))

    def _preload_features(self):
        """Load every feature tensor into RAM (threaded); unreadable files are dropped with a warning."""
        if not self.load_features_in_ram:
            return
        num_workers = max(1, int(self.preload_workers))
        print(f"Loading {len(self.samples)} feature tensors using {num_workers} workers...")
        pt_paths = [os.path.join(self.pt_folder_path, sample["feature_file"]) for sample in self.samples]
        loaded_samples = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_safe_load_feature_tensor, pt_paths)
            for sample, (tensor, error) in tqdm(zip(self.samples, results), total=len(self.samples)):
                if tensor is None:
                    print(f"Warning: Could not load feature file {sample['feature_file']}. Error: {error}. Skipping sample.")
                    continue
                sample["feature_tensor"] = tensor
                loaded_samples.append(sample)
        self.samples = loaded_samples

    def _feature(self, sample):
        if self.load_features_in_ram:
            return sample["feature_tensor"]
        return torch.load(os.path.join(self.pt_folder_path, sample["feature_file"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._feature(sample), sample["label"], sample["sublabel"]


def face_attribute_labels(attribute: str, face_data: dict):
    """(label, sublabel) of an FFHQ image for `attribute`, or None to exclude the image."""
    if attribute == "glasses":
        glasses_type = face_data.get("glasses", "NoGlasses")
        if glasses_type == "SwimmingGoggles":
            return None
        if glasses_type == "NoGlasses":
            return 0, 0
        return 1, GLASSES_SUBLABELS.get(glasses_type, 1)

    if attribute == "headPose":
        yaw = face_data.get("headPose", {}).get("yaw")
        if yaw is None:
            return 0, 0.0
        yaw_value = float(yaw)
        if YAW_BACKGROUND <= abs(yaw_value) <= YAW_EXCLUDED_MAX:
            return None
        return (0 if -YAW_BACKGROUND <= yaw_value <= YAW_BACKGROUND else 1), yaw_value

    if attribute == "smile":
        smile_value = face_data.get("smile", 0)
        if isinstance(smile_value, dict):
            smile_value = smile_value.get("value", 0)
        try:
            smile_value = float(smile_value)
        except (TypeError, ValueError):
            return None
        if smile_value > 1.0:
            smile_value = smile_value / 100.0
        label = 1 if smile_value >= 0.5 else 0
        return label, label

    if attribute == "gender":  # used by the gender classifier only
        gender_value = face_data.get("gender")
        if isinstance(gender_value, dict):
            gender_value = gender_value.get("value")
        if gender_value is None:
            return None
        gender_value = str(gender_value).lower()
        if gender_value == "male":
            return 1, 1
        if gender_value == "female":
            return 0, 0
        return None

    return 0, 0


class FeatureAttributeDataset(_FeatureDataset):
    """FFHQ features, labelled from the FFHQ-Features-Dataset JSON files (`<image>.json`).

    `attribute`: glasses, smile, headPose (sublabel = yaw in degrees, float), or gender.
    An image whose JSON file is missing or unreadable keeps label 0 and sublabel 0.
    """

    def __init__(self, pt_folder_path: str, json_folder_path: str, attribute: str,
                 load_features_in_ram: bool = False, preload_workers: int = 20):
        super().__init__(pt_folder_path, load_features_in_ram, preload_workers)
        self.json_folder_path = json_folder_path
        self.attribute = attribute
        sublabel_dtype = torch.float32 if attribute == "headPose" else torch.long

        print(f"Preprocessing dataset labels for attribute: {self.attribute}...")
        for feature_file in tqdm(self._feature_files()):
            base_filename = os.path.splitext(feature_file)[0]
            json_path = os.path.join(self.json_folder_path, f"{base_filename}.json")
            label, sublabel = 0, 0
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if data:
                    labels = face_attribute_labels(attribute, data[0].get("faceAttributes", {}))
                    if labels is None:
                        continue
                    label, sublabel = labels
            except (FileNotFoundError, json.JSONDecodeError, IndexError):
                print(f"Warning: Could not load attribute file {json_path}. Using label 0.")

            self.samples.append({
                "feature_file": feature_file,
                "label": torch.tensor(label, dtype=torch.long),
                "sublabel": torch.tensor(sublabel, dtype=sublabel_dtype),
            })
        self._preload_features()


class BratsFeatureDataset(_FeatureDataset):
    """BraTS slice features with binary labels read from a text file of `<filename> <label>` lines.
    The sublabel equals the label."""

    def __init__(self, pt_folder_path: str, labels_path: str, load_features_in_ram: bool = False,
                 preload_workers: int = 20):
        super().__init__(pt_folder_path, load_features_in_ram, preload_workers)
        self.labels_path = labels_path

        label_map = {}
        with open(labels_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    label_val = int(parts[1])
                except ValueError:
                    continue
                label_map[parts[0]] = label_val
                label_map[os.path.splitext(parts[0])[0]] = label_val

        for feature_file in tqdm(self._feature_files()):
            base_filename = os.path.splitext(feature_file)[0]
            slice_number = int(base_filename.split("_")[-1])
            if not BRATS_SLICES[0] < slice_number < BRATS_SLICES[1]:
                continue
            label = label_map.get(feature_file)
            if label is None:
                label = label_map.get(base_filename)
            if label is None:
                label = label_map.get(f"{base_filename}.png")
            if label is None:
                print(f"Warning: Label not found for {feature_file}. Skipping sample.")
                continue
            label_tensor = torch.tensor(int(label), dtype=torch.long)
            self.samples.append({"feature_file": feature_file, "label": label_tensor, "sublabel": label_tensor})
        self._preload_features()


def _read_afhq_breed_scores(sublabels_csv_path: Optional[str]):
    """CSV lines `filename, <class name> <score>, ...` -> {filename: {1: tabby, 2: Egyptian, 3: other cat scores}}."""
    sublabel_map = {}
    if not sublabels_csv_path:
        return sublabel_map
    with open(sublabels_csv_path, "r") as f:
        for line in f:
            parts = [part.strip() for part in line.strip().split(",") if part.strip()]
            if len(parts) < 2:
                continue
            class_scores = {1: 0.0, 2: 0.0, 3: 0.0}
            for entry in parts[1:]:
                pieces = entry.split()
                if len(pieces) < 2:
                    continue
                try:
                    score = float(pieces[-1])
                except ValueError:
                    continue
                name_lower = " ".join(pieces[:-1]).lower()
                if "tabby" in name_lower:
                    class_scores[1] += score
                elif "egyptian" in name_lower:
                    class_scores[2] += score
                else:
                    class_scores[3] += score
            sublabel_map[parts[0]] = class_scores
            sublabel_map[os.path.splitext(parts[0])[0]] = class_scores
    return sublabel_map


class AfhqFeatureDataset(_FeatureDataset):
    """AFHQ dog (label 0) / cat (label 1) features; "wild" images are excluded.

    Cat sublabels (1 tabby, 2 Egyptian cat, 3 other) are the argmax of the breed scores of `sublabels_csv_path`;
    with `return_soft_targets`, the normalized scores are returned as a 4th item.
    """

    def __init__(self, pt_folder_path: str, sublabels_csv_path: Optional[str] = None, load_features_in_ram: bool = False,
                 preload_workers: int = 20, return_soft_targets: bool = False):
        super().__init__(pt_folder_path, load_features_in_ram, preload_workers)
        self.sublabels_csv_path = sublabels_csv_path
        self.return_soft_targets = return_soft_targets
        self.sublabel_id_to_name = dict(AFHQ_SUBLABELS)
        sublabel_map = _read_afhq_breed_scores(sublabels_csv_path)

        for feature_file in tqdm(self._feature_files()):
            base_filename = os.path.splitext(feature_file)[0]
            lower_name = base_filename.lower()
            if "wild" in lower_name:
                continue
            if "dog" in lower_name:
                label = 0
            elif "cat" in lower_name:
                label = 1
            else:
                print(f"Warning: Could not infer label for {feature_file}. Skipping sample.")
                continue

            if label == 0:
                sublabel = 0
                soft_targets = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                class_scores = sublabel_map.get(feature_file)
                if class_scores is None:
                    class_scores = sublabel_map.get(base_filename)
                if class_scores is None:
                    class_scores = sublabel_map.get(f"{base_filename}.jpg")
                if class_scores is None:
                    print(f"Warning: Sublabel not found for {feature_file}. Skipping sample.")
                    continue
                scores = torch.tensor([0.0, class_scores.get(1, 0.0), class_scores.get(2, 0.0), class_scores.get(3, 0.0)],
                                      dtype=torch.float32)
                total = float(scores.sum().item())
                if total <= 0.0:
                    print(f"Warning: Sublabel scores not found for {feature_file}. Skipping sample.")
                    continue
                soft_targets = scores / total
                sublabel = int(torch.argmax(soft_targets[1:]).item()) + 1

            self.samples.append({
                "feature_file": feature_file,
                "label": torch.tensor(int(label), dtype=torch.long),
                "sublabel": torch.tensor(int(sublabel), dtype=torch.long),
                "soft_targets": soft_targets,
            })
        self._preload_features()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.return_soft_targets:
            return self._feature(sample), sample["label"], sample["sublabel"], sample["soft_targets"]
        return self._feature(sample), sample["label"], sample["sublabel"]


# ---------------------------------------------------------------------------
# Evaluation loaders
# ---------------------------------------------------------------------------

class PairedDataset(Dataset):
    """Item i pairs indices1[i] with indices2[i] (both cycled), for `max_images` items
    (default: the length of the shorter list)."""

    def __init__(self, dataset, indices1, indices2, max_images=None):
        self.dataset = dataset
        self.indices1 = indices1
        self.i1_sizes = len(indices1)
        self.indices2 = indices2
        self.i2_sizes = len(indices2)
        if self.i1_sizes == 0 or self.i2_sizes == 0:
            self.length = 0
        elif max_images is None:
            self.length = min(self.i1_sizes, self.i2_sizes)
        else:
            self.length = max_images

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[self.indices1[idx % self.i1_sizes]], self.dataset[self.indices2[idx % self.i2_sizes]]


def yaw_to_eval_class(yaw: torch.Tensor) -> torch.Tensor:
    """Evaluation classes of head pose: 0 straight (|yaw| <= 8), 1 left (< -12), 2 right (> 12), -1 excluded."""
    classes = torch.full_like(yaw, -1, dtype=torch.long)
    classes[(yaw >= -8.0) & (yaw <= 8.0)] = 0
    classes[yaw < -12.0] = 1
    classes[yaw > 12.0] = 2
    return classes


def create_evaluation_dataloaders(val_dataset, full_dataset, batch_size, attribute=None, max_pair_images=None):
    """Loaders of the evaluation protocol.

    Returns {"full": all evaluation samples, "pairs": {"i_j": pairs of classes i and j}, "class_ids": [...]},
    plus "zero_one" (pairs 0-1 then 0-2) and "style" (pairs 1-2) when the classes are exactly [0, 1, 2].
    Pair loaders hold `max_pair_images` pairs (default 1000).
    """
    attribute = attribute or getattr(full_dataset, "attribute", None)
    if isinstance(attribute, str) and attribute.lower() == "headpose":
        attribute = "headPose"

    def _extract_sublabel(dataset, idx):
        if hasattr(dataset, "samples"):
            return dataset.samples[idx]["sublabel"].item()
        _, _, sublabel = dataset[idx]
        return sublabel.item()

    val_indices = np.array(val_dataset.indices)
    val_sublabels = torch.tensor([_extract_sublabel(full_dataset, i) for i in val_indices])

    if attribute == "headPose":
        val_classes = yaw_to_eval_class(val_sublabels)
        keep = val_classes >= 0
        eval_indices, eval_sublabels = val_indices[keep], val_classes[keep]
    elif attribute == "glasses":
        keep = val_sublabels != 3
        eval_indices, eval_sublabels = val_indices[keep], val_sublabels[keep]
    else:
        eval_indices, eval_sublabels = val_indices, val_sublabels

    eval_dataset = Subset(full_dataset, eval_indices)
    full_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    if max_pair_images is None:
        max_pair_images = 1000

    class_ids = sorted(set(int(v) for v in eval_sublabels.tolist()))
    if attribute == "headPose" and class_ids != [0, 1, 2]:
        print(f"Warning: headPose evaluation classes found {class_ids}; swap loaders may be incomplete.")
    indices_by_class = {class_id: eval_indices[eval_sublabels == class_id] for class_id in class_ids}

    def pair_loader(class_i, class_j):
        pairs = PairedDataset(full_dataset, indices_by_class[class_i], indices_by_class[class_j], max_images=max_pair_images)
        return DataLoader(pairs, batch_size=batch_size, shuffle=False)

    pair_loaders = {}
    for i, class_i in enumerate(class_ids):
        for class_j in class_ids[i + 1:]:
            pair_loaders[f"{class_i}_{class_j}"] = pair_loader(class_i, class_j)

    loaders = {"full": full_loader, "pairs": pair_loaders, "class_ids": class_ids}
    if class_ids == [0, 1, 2]:
        zero_one = ConcatDataset([
            PairedDataset(full_dataset, indices_by_class[0], indices_by_class[1], max_images=max_pair_images),
            PairedDataset(full_dataset, indices_by_class[0], indices_by_class[2], max_images=max_pair_images),
        ])
        loaders["zero_one"] = DataLoader(zero_one, batch_size=batch_size, shuffle=False)
        loaders["style"] = pair_loader(1, 2)
    return loaders
