"""Data preparation scripts vs the original ones."""

import importlib
import importlib.util
import os
import sys

import numpy as np
import pytest

from conftest import ORIG_SRC, requires_orig

pytestmark = requires_orig
SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_brats_labels_identical(tmp_path):
    nib = pytest.importorskip("nibabel")
    orig_path = os.path.join(ORIG_SRC, "preprocess_brats.py")
    if not os.path.exists(orig_path):
        pytest.skip("original preprocess_brats.py not in DIFFCA_ORIG_SRC")
    rng = np.random.default_rng(0)
    cases = []
    for c in range(3):
        case = tmp_path / "raw" / f"BraTS-GLI-{c:05d}-000"
        case.mkdir(parents=True)
        vol = np.zeros((40, 44, 30))
        vol[5:35, 6:40, 3:27] = rng.normal(100, 20, (30, 34, 24)) + c
        seg = np.zeros_like(vol)
        seg[10:30, 10:35, 8 + c:20] = 1  # 500 voxels per slice: no slice reaches the 700-voxel threshold ...
        seg[8:32, 8:38, 12:14] = 1       # ... except these
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(case / f"{case.name}-t1n.nii.gz"))
        if c != 2:  # a case without segmentation: label -1
            nib.save(nib.Nifti1Image(seg, np.eye(4)), str(case / f"{case.name}-seg.nii.gz"))
        cases.append(str(case))

    # Importable by name: the scripts run cases in worker processes.
    for path in (ORIG_SRC, SCRIPTS):
        if path not in sys.path:
            sys.path.insert(0, path)
    orig = importlib.import_module("preprocess_brats")
    new = importlib.import_module("prepare_brats")
    assert orig.split_cases(cases, val_fraction=0.1, seed=42) == new.split_cases(cases)
    orig.process_and_save_brats_slices(str(tmp_path / "raw"), str(tmp_path / "orig"), num_workers=1, case_dirs=cases)
    new.process_cases(cases, str(tmp_path / "new"), num_workers=1)
    orig_labels = open(tmp_path / "orig" / "labels.txt").read()
    assert orig_labels == open(tmp_path / "new" / "labels.txt").read()
    assert "\t1\n" in orig_labels and "\t0\n" in orig_labels and "\t-1\n" in orig_labels
    names = [line.split("\t")[0] for line in orig_labels.splitlines()]
    assert sorted(names) == sorted(f for f in os.listdir(tmp_path / "new") if f.endswith(".png"))


def test_prepare_ffhq_split(tmp_path):
    src = tmp_path / "ffhq"
    src.mkdir()
    for idx in (0, 59999, 60000, 69999):
        (src / f"{idx:05d}.png").write_bytes(b"x")
    (src / "readme.png").write_bytes(b"x")
    new = _load(os.path.join(SCRIPTS, "prepare_ffhq.py"), "new_prepare_ffhq")
    sys.argv = ["prepare_ffhq", "--input", str(src), "--output", str(tmp_path / "new")]
    new.main()
    orig = _load(os.path.join(ORIG_SRC, "split_ffhq.py"), "orig_split_ffhq") if os.path.exists(os.path.join(ORIG_SRC, "split_ffhq.py")) else None
    if orig is not None:
        orig.split_ffhq(str(src), str(tmp_path / "orig"))
        for split in ("train", "test"):
            assert sorted(os.listdir(tmp_path / "orig" / split)) == sorted(os.listdir(tmp_path / "new" / split))
    assert sorted(os.listdir(tmp_path / "new" / "test")) == ["60000.png", "69999.png"]
