"""Stage 1 end to end, original vs cleaned code: training, feature extraction, checkpoint evaluation, preprocessing.

Each step runs in a fresh process (see runners.py) and the outputs are compared exactly: checkpoints
(weights, EMA, optimizer, RNG states), saved EMA weights, metrics, extracted token files.
"""

import os
import shutil

import pytest
import yaml

from conftest import assert_same, assert_same_tree, load_any, requires_orig, run_impl
from fixtures import stage1_config, write_yaml

pytestmark = requires_orig

CASES = {
    # name: (encoder, online)
    "cqc_cached": ("CrossQueryColorEncoder", False),   # FFHQ paper setting (cached VQ latents + DINOv3 tokens)
    "cqc_online": ("CrossQueryColorEncoder", True),    # AFHQ/BraTS setting (KL VAE, features online, augmentations)
    "cq_cached": ("CrossQueryEncoder", False),
    "dino_cached": ("DinoEncoder", False),
    "encdiff_cached": ("EncDiffEncoder", False),
}


def _train(fixtures_root, tmp_root, name, impl):
    encoder, online = CASES[name]
    config = stage1_config(fixtures_root, name, encoder=encoder, online=online)
    config["output_dir"] = os.path.join(tmp_root, impl, name)
    path = write_yaml(config, os.path.join(tmp_root, f"{impl}_{name}.yaml"))
    run_impl("stage1_train", impl, os.path.join(tmp_root, impl, f"{name}_logs"), "--config", path)
    return config


@pytest.fixture(scope="module")
def stage1_runs(fixtures_root, tmp_path_factory):
    root = str(tmp_path_factory.mktemp("stage1"))
    configs = {}
    for name in CASES:
        for impl in ("orig", "new"):
            configs[(name, impl)] = _train(fixtures_root, root, name, impl)
    return root, configs


@pytest.mark.parametrize("name", list(CASES))
def test_training_outputs_identical(stage1_runs, name):
    root, _ = stage1_runs
    files = assert_same_tree(os.path.join(root, "orig", name), os.path.join(root, "new", name),
                             ignore=lambda rel: rel.startswith("config_saved"))
    # checkpoints at steps 3 and 6, final state, EMA weights, metrics of the quick (4) and FID (6) evaluations
    for expected in ("checkpoint-3/model.safetensors", "checkpoint-6/custom_checkpoint_0.pkl", "checkpoint-final/optimizer.bin",
                     "checkpoint-6/random_states_0.pkl", "pytorch_model_ema.bin", "encoder_ema.bin", "metrics.txt",
                     "quick-eval-step-4/images/recon_absdiff_grid.png"):
        assert expected in files, expected
    assert_same(load_any(os.path.join(root, "orig", f"{name}_logs", "eval_results.json")),
                load_any(os.path.join(root, "new", f"{name}_logs", "eval_results.json")))


def test_checkpoint_keys_keep_compile_prefix(stage1_runs):
    root, _ = stage1_runs
    keys = list(load_any(os.path.join(root, "new", "cqc_cached", "checkpoint-6", "model.safetensors")))
    assert keys and all(k.startswith(("diffusion_model._orig_mod.", "encoder._orig_mod.")) for k in keys)


@pytest.mark.parametrize("name", ["cqc_cached", "cqc_online"])
def test_extracted_features_identical(stage1_runs, name):
    root, configs = stage1_runs
    outs = {}
    for impl in ("orig", "new"):
        config = configs[(name, impl)]
        out = os.path.join(root, impl, f"{name}_features")
        run_impl("stage1_extract", impl, os.path.join(root, impl, f"{name}_extract_logs"),
                 "--config", os.path.join(root, f"{impl}_{name}.yaml"),
                 "--checkpoint", os.path.join(config["output_dir"], "checkpoint-6"), "--output_dir", out)
        outs[impl] = out
    files = assert_same_tree(outs["orig"], outs["new"])
    assert any(f.startswith("train/") for f in files) and any(f.startswith("test/") for f in files)
    assert tuple(load_any(os.path.join(outs["new"], files[0])).shape) == (32, 128)


def test_extract_overrides(stage1_runs):
    root, configs = stage1_runs
    outs = {}
    for impl in ("orig", "new"):
        out = os.path.join(root, impl, "override_features")
        run_impl("stage1_extract", impl, os.path.join(root, impl, "override_logs"),
                 "--config", os.path.join(root, f"{impl}_cqc_cached.yaml"),
                 "--checkpoint", os.path.join(configs[("cqc_cached", impl)]["output_dir"], "checkpoint-3"),
                 "--output_dir", out, "--data.batch_size", "5", "--data.num_workers=0")
        outs[impl] = out
    assert_same_tree(outs["orig"], outs["new"])


@pytest.mark.parametrize("name", ["cqc_cached", "dino_cached"])
def test_checkpoint_evaluation_identical(stage1_runs, name):
    root, configs = stage1_runs
    results = {}
    for impl in ("orig", "new"):
        logs = os.path.join(root, impl, f"{name}_evalckpt")
        # The original uses the EMA weights only when the path contains "EMA" (and fails otherwise).
        ckpt = os.path.join(root, impl, f"{name}_EMA", "checkpoint-6")
        if not os.path.exists(ckpt):
            shutil.copytree(os.path.join(configs[(name, impl)]["output_dir"], "checkpoint-6"), ckpt)
        args = ["--config", os.path.join(root, f"{impl}_{name}.yaml"), "--checkpoint", ckpt, "--eval_samples", "6"]
        if impl == "new":  # the original always loads a DINOv3 backbone (from a hard-coded path)
            args += ["--backbone_path", os.path.join(os.environ["DIFFCA_FIXTURES"], "dino")]
        run_impl("stage1_evaluate_checkpoint", impl, logs, *args)
        results[impl] = load_any(os.path.join(logs, "eval_results.json"))
    assert_same(results["orig"], results["new"])
    assert "FD-Dino" in results["new"][0]


@pytest.mark.parametrize("use_h5", [False, True])
def test_preprocess_identical_to_committed_version(fixtures_root, tmp_path, use_h5):
    """The working-tree preprocess.py had its save call commented out; the cleaned one matches the committed one."""
    pytest.importorskip("cqae.preprocess_committed")
    outs = {}
    for impl in ("orig", "new"):
        config = {"train_path": f"{fixtures_root}/ffhq/train", "test_path": f"{fixtures_root}/ffhq/test",
                  "output_path": str(tmp_path / impl / "features"), "batch_size": 7, "save_ae": True,
                  "ae_type": "VQModel", "ae_path": f"{fixtures_root}/vq", "display_metrics": True, "save_dino": True,
                  "dino_type": "DINOv3", "dino_path": f"{fixtures_root}/dino", "use_h5": use_h5}
        path = tmp_path / f"{impl}.yaml"
        path.write_text(yaml.safe_dump(config))
        run_impl("preprocess", impl, str(tmp_path / impl / "logs"), "--config", str(path))
        outs[impl] = str(tmp_path / impl / "features")
    files = assert_same_tree(outs["orig"], outs["new"])
    assert "ae_config.json" in files and any(("dino_features" in f) or f.endswith(".h5") for f in files)
