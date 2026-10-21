"""Train the referee classifier of one attribute on the extracted tokens (needed before separator training).

    python -m diffca.stage2.train_classifier --attribute glasses \
        --pt_folder_train <features>/train --pt_folder_test <features>/test --json_folder <ffhq-features>/json \
        --load_features_in_ram --save_name color_aug_OT_EMA

  glasses, smile: linear probe on the sublabels, class-weighted cross-entropy, best epoch by worst-class F1.
  headPose:       yaw regressor (MSE), best epoch by validation MAE.
  afhq:           MLP on the soft breed targets (class-weighted KL; needs --sublabels_csv), best by worst-class F1.
  brats:          MLP on the binary labels (needs --labels_train/--labels_test), best by worst-class F1.

The best model is saved to <output_dir>/<attribute>_<save_name>.pt; the separator configs point to it in
`referee_classifier_path`.
"""

import argparse
import os
import pprint
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffca.stage2.classifiers import build_referee
from diffca.stage2.datasets import AfhqFeatureDataset, BratsFeatureDataset, FeatureAttributeDataset
from diffca.utils import seed_everything

TARGET_NAMES = {
    "glasses": ["NoGlasses", "ReadingGlasses", "Sunglasses"],
    "smile": ["no_smile", "smile"],
    "afhq": ["dog", "tabby", "Egyptian_cat", "other_cat"],
    "brats": ["class_0", "class_1"],
}


def report_metrics(y_true, y_pred, epoch, target_names):
    """Print the confusion matrix and classification report; return the worst per-class F1."""
    print("\n" + "=" * 50)
    print(f"Validation Metrics for Epoch {epoch + 1}")
    print("Confusion Matrix:")
    pprint.pprint(confusion_matrix(y_true, y_pred))
    labels = list(range(len(target_names)))
    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=target_names,
                                        output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    f1_scores = [report_dict[name]["f1-score"] for name in target_names if name in report_dict]
    print("=" * 50 + "\n")
    return min(f1_scores) if f1_scores else 0.0


def yaw_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """MAE, and accuracy of the yaw bins (|yaw| <= 10 straight, < -10 left, > 10 right)."""
    def to_class(values):
        classes = np.full(values.shape, -1, dtype=np.int64)
        classes[(values >= -10) & (values <= 10)] = 0
        classes[values < -10] = 1
        classes[values > 10] = 2
        return classes

    pred_classes, target_classes = to_class(preds), to_class(targets)
    valid = (pred_classes >= 0) & (target_classes >= 0)
    accuracy = float(np.mean(pred_classes[valid] == target_classes[valid])) if np.any(valid) else float("nan")
    return {"mae": float(np.mean(np.abs(preds - targets))), "accuracy": accuracy}


def evaluate_regressor(model, loader, device) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for features, _, sublabels in loader:
            preds.append(model(features.to(device)).cpu().numpy())
            targets.append(sublabels.to(device).cpu().numpy())
    if not preds:
        return {"mae": float("nan"), "accuracy": float("nan")}
    return yaw_metrics(np.concatenate(preds), np.concatenate(targets))


def build_datasets(args, attribute):
    if attribute == "afhq":
        kwargs = dict(sublabels_csv_path=args.sublabels_csv, load_features_in_ram=args.load_features_in_ram,
                      return_soft_targets=True)
        return AfhqFeatureDataset(args.pt_folder_train, **kwargs), AfhqFeatureDataset(args.pt_folder_test, **kwargs)
    if attribute == "brats":
        return (BratsFeatureDataset(args.pt_folder_train, args.labels_train, load_features_in_ram=args.load_features_in_ram),
                BratsFeatureDataset(args.pt_folder_test, args.labels_test, load_features_in_ram=args.load_features_in_ram))
    kwargs = dict(json_folder_path=args.json_folder, attribute=attribute, load_features_in_ram=args.load_features_in_ram)
    return FeatureAttributeDataset(args.pt_folder_train, **kwargs), FeatureAttributeDataset(args.pt_folder_test, **kwargs)


def print_afhq_counts(dataset, split):
    counts = {0: Counter(), 1: Counter()}
    for sample in dataset.samples:
        counts[sample["label"].item()][sample["sublabel"].item()] += 1

    def fmt(counter):
        parts = [f"{s}:{dataset.sublabel_id_to_name.get(s, f'class_{s}')}={counter[s]}" for s in sorted(counter)]
        return ", ".join(parts) if parts else "none"

    print(f"{split} sublabel counts:")
    print(f"  label 0 (dog): {fmt(counts[0])}")
    print(f"  label 1 (cat): {fmt(counts[1])}")


def build_criterion(attribute, train_dataset, device):
    """The loss function of `attribute`, as a function (outputs, batch) -> loss."""
    if attribute == "headPose":
        mse = nn.MSELoss()
        return lambda outputs, batch: mse(outputs, batch[2].to(device))

    if attribute == "afhq":
        sublabels = torch.tensor([s["sublabel"].item() for s in train_dataset.samples])
        class_counts = torch.bincount(sublabels, minlength=4).float()
        class_weights = (1.0 / torch.sqrt(torch.clamp(class_counts, min=1.0))).to(device)
        kl_div = nn.KLDivLoss(reduction="none")

        def afhq_loss(outputs, batch):
            soft_targets = batch[3].to(device)
            per_sample_loss = kl_div(torch.log_softmax(outputs, dim=1), soft_targets).sum(dim=1)
            sample_weights = (soft_targets * class_weights).sum(dim=1)
            return (per_sample_loss * sample_weights).mean()
        return afhq_loss

    if attribute == "brats":
        labels = torch.tensor([s["label"].item() for s in train_dataset.samples])
        class_counts = torch.bincount(labels, minlength=2).float()
        ce = nn.CrossEntropyLoss(weight=(1.0 / torch.clamp(class_counts, min=1.0)).to(device))
        return lambda outputs, batch: ce(outputs, batch[1].to(device))

    sublabels = torch.tensor([s["sublabel"].item() for s in train_dataset.samples])
    class_counts = torch.bincount(sublabels)
    ce = nn.CrossEntropyLoss(weight=(1.0 / torch.sqrt(class_counts).float()).to(device))
    return lambda outputs, batch: ce(outputs, batch[2].to(device))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a referee classifier on the extracted tokens.")
    parser.add_argument("--attribute", type=str, required=True, choices=["glasses", "smile", "headPose", "headpose", "afhq", "brats"])
    parser.add_argument("--pt_folder_train", type=str, required=True, help="Folder of training .pt features.")
    parser.add_argument("--pt_folder_test", type=str, required=True, help="Folder of validation .pt features.")
    parser.add_argument("--json_folder", type=str, default=None, help="FFHQ attribute .json files (glasses, smile, headPose).")
    parser.add_argument("--sublabels_csv", type=str, default=None, help="AFHQ breed scores CSV.")
    parser.add_argument("--labels_train", type=str, default=None, help="BraTS training labels .txt.")
    parser.add_argument("--labels_test", type=str, default=None, help="BraTS validation labels .txt.")
    parser.add_argument("--load_features_in_ram", action="store_true", help="Load all features into RAM.")
    parser.add_argument("--output_dir", type=str, default="classifier_checkpoints", help="Directory of the best model.")
    parser.add_argument("--save_name", type=str, default="classifier", help="Saved as <attribute>_<save_name>.pt.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    attribute = "headPose" if args.attribute.lower() == "headpose" else args.attribute
    required = {"afhq": ["sublabels_csv"], "brats": ["labels_train", "labels_test"]}.get(attribute, ["json_folder"])
    missing = [f"--{key}" for key in required if not getattr(args, key)]
    if missing:
        parser.error(f"--attribute {attribute} requires {' '.join(missing)}")

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_datasets(args, attribute)
    print(f"Total samples in training dataset: {len(train_dataset)}")
    print(f"Total samples in validation dataset: {len(val_dataset)}")
    class_distribution = Counter(s["sublabel"].item() for s in train_dataset.samples)
    if attribute == "afhq":
        print_afhq_counts(train_dataset, "Train")
        print_afhq_counts(val_dataset, "Validation")
    elif attribute != "headPose":
        print("Class distribution:", class_distribution)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_tokens = next(iter(train_loader))[0].shape[1]
    num_classes = len(class_distribution) if attribute == "glasses" else None
    model = build_referee(attribute, num_tokens=num_tokens, num_classes=num_classes).to(device)
    criterion = build_criterion(attribute, train_dataset, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    save_path = os.path.join(args.output_dir, f"{attribute}_{args.save_name}.pt")
    target_names = TARGET_NAMES.get(attribute)
    best_score = float("inf") if attribute == "headPose" else 0.0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for batch in (pbar := tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]")):
            optimizer.zero_grad()
            outputs = model(batch[0].to(device))
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        print(f"Epoch {epoch + 1} - Average Training Loss: {total_train_loss / max(len(train_loader), 1):.4f}")

        if attribute == "headPose":
            train_metrics = evaluate_regressor(model, train_loader, device)
            val_metrics = evaluate_regressor(model, val_loader, device)
            print(f"Epoch {epoch + 1} - Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f} | "
                  f"Train Acc (yaw bins): {train_metrics['accuracy']:.4f} | Val Acc (yaw bins): {val_metrics['accuracy']:.4f}")
            improved, score = val_metrics["mae"] < best_score, val_metrics["mae"]
        else:
            model.eval()
            all_preds, all_targets = [], []
            target_index = 1 if attribute == "brats" else 2  # labels for BraTS, sublabels otherwise
            with torch.no_grad():
                for batch in val_loader:
                    _, predicted = torch.max(model(batch[0].to(device)).data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(batch[target_index].to(device).cpu().numpy())
            score = report_metrics(np.array(all_targets), np.array(all_preds), epoch, target_names)
            print(f"Epoch {epoch + 1} - Minimum F1-Score: {score:.4f};   Best overall F1: {best_score:.4f}")
            improved = score > best_score

        if improved:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} ({'Val MAE' if attribute == 'headPose' else 'Min F1'}: {best_score:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
