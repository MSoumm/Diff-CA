"""Equivalence tests: the cleaned code must compute exactly what the original code computed.

Requirements:
  DIFFCA_ORIG_SRC   folder containing the original `cqae` package (the tests are skipped without it)
Optional:
  DIFFCA_STAGE1_CHECKPOINT   a real Stage 1 checkpoint folder (checkpoint-<step>) with its config_saved.yaml one
                             level up, for the checkpoint-loading tests
  DIFFCA_CLASSIFIER_DIR      folder with the trained referee classifier checkpoints
  DIFFCA_FIXTURES            where to build the tiny models and synthetic data (default: a temp folder)
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TESTS_DIR, "..", "src")
ORIG_SRC = os.environ.get("DIFFCA_ORIG_SRC")

sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, SRC_DIR)
if ORIG_SRC:
    sys.path.insert(1, ORIG_SRC)

requires_orig = pytest.mark.skipif(not ORIG_SRC, reason="DIFFCA_ORIG_SRC (original code) not set")


@pytest.fixture(scope="session")
def fixtures_root():
    from fixtures import build_all
    root = os.environ.get("DIFFCA_FIXTURES") or os.path.join(tempfile.gettempdir(), "diffca_test_fixtures")
    os.environ["DIFFCA_FIXTURES"] = root
    return build_all(root)


def run_impl(task, impl, out_dir, *args, timeout=3600):
    """Run `runners.py` in a fresh interpreter; return its combined output."""
    env = dict(os.environ, PYTHONWARNINGS="ignore", DIFFCA_ORIG_SRC=ORIG_SRC or "",
               DIFFCA_TINY_BACKBONE=os.path.join(os.environ.get("DIFFCA_FIXTURES", ""), "dino"))
    proc = subprocess.run([sys.executable, os.path.join(TESTS_DIR, "runners.py"), task, impl, out_dir, *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f"{task}/{impl} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-6000:]}")
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Deep comparison of outputs
# ---------------------------------------------------------------------------

def assert_same(a, b, path="root"):
    """Exact equality of nested containers of tensors, arrays and scalars (NaN == NaN)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, f"{path}: tensor type/shape"
        assert torch.equal(a, b) or (a.is_floating_point() and torch.equal(torch.nan_to_num(a, nan=7.7), torch.nan_to_num(b, nan=7.7))), f"{path}: tensor values differ"
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape, f"{path}: array shape"
        if a.dtype == object:
            assert_same(a.tolist(), b.tolist(), path)
        else:
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), f"{path}: array values differ"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(map(str, set(a) ^ set(b)))}"
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        pass
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def load_any(path):
    ext = os.path.splitext(path)[1]
    if ext == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    if ext in (".bin", ".pt", ".pkl"):
        return torch.load(path, map_location="cpu", weights_only=False)
    if ext == ".png":
        from PIL import Image
        return np.asarray(Image.open(path))
    if ext == ".npy":
        return np.load(path)
    if ext == ".h5":
        import h5py
        with h5py.File(path, "r") as f:
            return {k: f[k][:] for k in f}
    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    with open(path, "rb") as f:
        return f.read()


def assert_same_tree(dir_a, dir_b, ignore=lambda rel: False):
    """Same files, same content (loaded by type) in two output folders."""
    def files(d):
        out = set()
        for root, _, names in os.walk(d):
            for n in names:
                rel = os.path.relpath(os.path.join(root, n), d)
                if not ignore(rel):
                    out.add(rel)
        return out

    fa, fb = files(dir_a), files(dir_b)
    assert fa == fb, f"different files: only in {dir_a}: {sorted(fa - fb)[:10]}, only in {dir_b}: {sorted(fb - fa)[:10]}"
    assert fa, f"no files in {dir_a}"
    for rel in sorted(fa):
        assert_same(load_any(os.path.join(dir_a, rel)), load_any(os.path.join(dir_b, rel)), rel)
    return sorted(fa)
