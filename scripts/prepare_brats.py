"""BraTS 2023 (GLI training data) -> 2D axial T1 (t1n) slices as PNG, with binary tumor labels.

    python scripts/prepare_brats.py --input <ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData> --output data/brats_cropped/images

Cases are split 90/10 into train/ and val/ (seed 42). Each volume is cropped to a cube around the brain (10%
margin) and z-scored on the brain voxels; slices are clipped to [-3, 3] and rescaled to [0, 255]. Slices that are
99.5% black or more are dropped. Each split folder gets a labels.txt with lines "<filename>\t<label>", label 1 when
the slice has at least 700 tumor voxels.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

TUMOR_VOXELS = 700
BLACK_PIXEL_THRESHOLD = 0.995


def zscore_volume(volume, mask=None):
    vox = volume[mask > 0] if mask is not None else volume[volume > 0]
    return (volume - vox.mean()) / (vox.std() + 1e-8)


def cube_bounds(mask, margin_ratio=0.1):
    """Slices of a cube around the non-zero voxels of `mask` (side: largest extent + margin), kept inside the volume."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    side = int(np.ceil((maxs - mins + 1).max() * (1.0 + margin_ratio)))
    center = (mins + maxs) / 2.0
    slices = []
    for axis in range(3):
        s = int(np.floor(center[axis] - side / 2.0))
        e = s + side
        if s < 0:
            e -= s
            s = 0
        if e > mask.shape[axis]:
            s -= e - mask.shape[axis]
            e = mask.shape[axis]
        slices.append(slice(max(0, s), e))
    return tuple(slices)


def process_case(case_dir, output_dir, include_empty=True, missing_label=-1):
    t1n_path = glob.glob(os.path.join(case_dir, "*-t1n.nii.gz"))
    seg_path = glob.glob(os.path.join(case_dir, "*-seg.nii.gz"))
    if not t1n_path:
        return []
    t1n = nib.load(t1n_path[0]).get_fdata()
    seg = nib.load(seg_path[0]).get_fdata() if seg_path else None

    brain_mask = np.abs(t1n) > 0
    bounds = cube_bounds(brain_mask)
    if bounds is not None:
        t1n, brain_mask = t1n[bounds], brain_mask[bounds]
        if seg is not None:
            seg = seg[bounds]
    t1n = zscore_volume(t1n, brain_mask)

    labels = []
    case_name = os.path.basename(case_dir)
    for i in range(t1n.shape[2]):
        has_tumor = seg is not None and seg[:, :, i].sum() >= TUMOR_VOXELS
        if seg is not None and not include_empty and not has_tumor:
            continue
        img = np.clip(t1n[:, :, i], -3, 3)
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        if np.mean(img == 0) >= BLACK_PIXEL_THRESHOLD:
            continue
        filename = f"{case_name}_slice_{i:03d}.png"
        Image.fromarray(np.stack([img] * 3, axis=-1)).save(os.path.join(output_dir, filename))
        label = missing_label if seg is None else int(has_tumor)
        labels.append(f"{filename}\t{label}\n")
    return labels


def process_cases(case_dirs, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(num_workers, len(case_dirs)))) as executor:
        futures = [executor.submit(process_case, case_dir, output_dir) for case_dir in case_dirs]
        with open(os.path.join(output_dir, "labels.txt"), "w") as labels_file:
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Writing {output_dir}"):
                labels_file.writelines(future.result())


def split_cases(case_dirs, val_fraction=0.1, seed=42):
    indices = np.arange(len(case_dirs))
    np.random.RandomState(seed).shuffle(indices)
    val_idx = set(indices[:max(1, int(round(len(case_dirs) * val_fraction)))])
    return ([c for i, c in enumerate(case_dirs) if i not in val_idx], [c for i, c in enumerate(case_dirs) if i in val_idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="BraTS 2023 GLI training data (BraTS-GLI-* folders).")
    parser.add_argument("--output", required=True, help="Where to create train/ and val/.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    case_dirs = sorted(glob.glob(os.path.join(args.input, "BraTS*")))
    if not case_dirs:
        raise SystemExit(f"No BraTS-* case folders in {args.input}")
    train_cases, val_cases = split_cases(case_dirs)
    process_cases(train_cases, os.path.join(args.output, "train"), args.num_workers)
    process_cases(val_cases, os.path.join(args.output, "val"), args.num_workers)


if __name__ == "__main__":
    main()
