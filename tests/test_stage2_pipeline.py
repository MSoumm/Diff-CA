"""Stage 2 end to end, original vs cleaned code, in fresh processes: referee classifier training, separator
training with the paper configs (scaled down), and swap / interpolation generation with a trained separator.

Compared exactly: saved classifiers, separator checkpoints (weights, EMA copies, adversary and GRL buffers,
LR schedulers, best metrics, RNG states), every logged value (per step and per epoch), the evaluation
pair lists, and the generated images.
"""

import os

import pytest
import yaml

from conftest import ORIG_SRC, assert_same, assert_same_tree, load_any, requires_orig, run_impl
from fixtures import stage1_config, write_yaml

pytestmark = requires_orig

ATTRIBUTES = ["glasses", "smile", "headPose", "afhq", "brats"]
PAPER_YAML = {"glasses": "glasses", "smile": "smile", "headPose": "yaw", "afhq": "afhq", "brats": "brats"}
# Log entries of removed options (only the original logs them).
REMOVED_LOG_KEYS = {"push_loss", "loss_kl", "ema_push_loss", "adv_integral_error", "positive_sampled_norms",
                    "negative_sampled_norms"}


def data_args(root, attribute):
    if attribute == "afhq":
        return {"train_pt_folder": f"{root}/tokens_afhq/train", "test_pt_folder": f"{root}/tokens_afhq/test",
                "sublabels_csv": f"{root}/afhq_breeds.csv"}
    if attribute == "brats":
        return {"train_pt_folder": f"{root}/tokens_brats/train", "test_pt_folder": f"{root}/tokens_brats/test",
                "labels_train": f"{root}/brats_train_labels.txt", "labels_test": f"{root}/brats_test_labels.txt"}
    return {"train_pt_folder": f"{root}/tokens_ffhq/train", "test_pt_folder": f"{root}/tokens_ffhq/test",
            "json_folder": f"{root}/ffhq_json"}


def classifier_args(root, attribute):
    d = data_args(root, attribute)
    args = ["--attribute", attribute, "--pt_folder_train", d["train_pt_folder"], "--pt_folder_test", d["test_pt_folder"],
            "--epochs", "6", "--batch_size", "16", "--lr", "1e-3", "--load_features_in_ram", "--save_name", "test"]
    for key, flag in (("json_folder", "--json_folder"), ("sublabels_csv", "--sublabels_csv"),
                      ("labels_train", "--labels_train"), ("labels_test", "--labels_test")):
        if key in d:
            args += [flag, d[key]]
    return args


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return str(tmp_path_factory.mktemp("stage2"))


@pytest.fixture(scope="module")
def classifiers(fixtures_root, work):
    for attribute in ATTRIBUTES:
        for impl in ("orig", "new"):
            out = os.path.join(work, impl, "classifiers")
            run_impl("classifier", impl, os.path.join(work, impl, f"clf_{attribute}_logs"),
                     *classifier_args(fixtures_root, attribute), "--output_dir", out)
    return {impl: os.path.join(work, impl, "classifiers") for impl in ("orig", "new")}


def test_referee_classifier_training_identical(classifiers):
    files = assert_same_tree(classifiers["orig"], classifiers["new"])
    assert files == sorted(f"{a}_test.pt" for a in ATTRIBUTES)


def separator_yaml(fixtures_root, work, impl, attribute, referee):
    config = yaml.safe_load(open(os.path.join(ORIG_SRC, "cqae", "separator_configs", f"{PAPER_YAML[attribute]}.yaml")))
    config.update(data_args(fixtures_root, attribute))
    # batch_size: no split of the synthetic data leaves a batch of 1 (the cycle pass needs pairs, in both versions)
    config.update(output_dir=os.path.join(work, impl, f"sep_{attribute}"), referee_classifier_path=referee,
                  epochs=4, batch_size=12, mixed_precision="no", wandb_mode="disabled", config_file=None, save_dir=None)
    return write_yaml(config, os.path.join(work, f"{impl}_sep_{attribute}.yaml"))


@pytest.fixture(scope="module")
def separators(fixtures_root, work, classifiers):
    for attribute in ATTRIBUTES:
        referee = os.path.join(classifiers["orig"], f"{attribute}_test.pt")  # the same referee for both runs
        for impl in ("orig", "new"):
            run_impl("stage2_train", impl, os.path.join(work, impl, f"sep_{attribute}_logs"),
                     "--config_yaml", separator_yaml(fixtures_root, work, impl, attribute, referee))
    return work


def _compare_logs(orig_log, new_log):
    assert len(orig_log) == len(new_log) > 0
    for o, n in zip(orig_log, new_log):
        assert o["step"] == n["step"]
        removed = {k for k in o["data"] if k.split("/")[-1] in REMOVED_LOG_KEYS}
        assert set(n["data"]) == set(o["data"]) - removed, set(n["data"]) ^ (set(o["data"]) - removed)
        assert_same({k: o["data"][k] for k in n["data"]}, n["data"], f"step {n['step']}")


@pytest.mark.parametrize("attribute", ATTRIBUTES)
def test_separator_training_identical(separators, attribute):
    work = separators
    files = assert_same_tree(os.path.join(work, "orig", f"sep_{attribute}"), os.path.join(work, "new", f"sep_{attribute}"),
                             ignore=lambda rel: rel == "training_args.json")
    for expected in ("disentangler_checkpoints/epoch_3/model.safetensors",     # separator
                     "disentangler_checkpoints/epoch_3/model_1.safetensors",   # adversary (+ GRL state)
                     "disentangler_checkpoints/epoch_3/custom_checkpoint_0.pkl",
                     "disentangler_checkpoints/epoch_3/scheduler_1.bin"):
        assert expected in files
    logs = {impl: os.path.join(work, impl, f"sep_{attribute}_logs") for impl in ("orig", "new")}
    assert_same(load_any(os.path.join(logs["orig"], "best_metrics.json")), load_any(os.path.join(logs["new"], "best_metrics.json")))
    _compare_logs(load_any(os.path.join(logs["orig"], "wandb_log.json")), load_any(os.path.join(logs["new"], "wandb_log.json")))
    swap_keys = [k for k in load_any(os.path.join(logs["new"], "best_metrics.json")) if k.startswith("Acc_")]
    assert swap_keys


def test_swap_generation_identical(fixtures_root, separators, work):
    """--test_epoch --generate_all_images --interpolate from a trained separator and a (tiny) Stage 1 model."""
    stage1 = stage1_config(fixtures_root, "stage1_for_swaps")
    stage1["output_dir"] = os.path.join(work, "stage1_for_swaps")
    stage1_yaml = write_yaml(stage1, os.path.join(work, "stage1_for_swaps.yaml"))
    run_impl("stage1_train", "new", os.path.join(work, "stage1_for_swaps_logs"), "--config", stage1_yaml)

    checkpoint = os.path.join(work, "orig", "sep_glasses", "disentangler_checkpoints", "epoch_3")
    for impl in ("orig", "new"):
        os.environ["DIFFCA_MAX_PAIRS"] = "12"
        try:
            run_impl("stage2_train", impl, os.path.join(work, impl, "swaps_logs"),
                     "--config_yaml", os.path.join(work, f"{impl}_sep_glasses.yaml"),
                     "--output_dir", os.path.join(work, impl, "swaps"), "--resume_from_checkpoint", checkpoint,
                     "--test_epoch", "--generate_all_images", "--interpolate", "--epochs", "10",
                     "--config_file", os.path.join(stage1["output_dir"], "config_saved.yaml"),
                     "--save_dir", stage1["output_dir"], "--global_step", "6")
        finally:
            del os.environ["DIFFCA_MAX_PAIRS"]
    files = assert_same_tree(os.path.join(work, "orig", "swaps"), os.path.join(work, "new", "swaps"),
                             ignore=lambda rel: rel == "training_args.json")
    for part in ("final_evaluation/0_1.txt", "final_evaluation/swapped_images/1_2/swap_y2x/",
                 "final_evaluation/swapped_images/0_2/recon_cs_x/", "4/interpolations/interpolation_grid_0.png"):
        assert any(f.startswith(part) for f in files), part
    _compare_logs(load_any(os.path.join(work, "orig", "swaps_logs", "wandb_log.json")),
                  load_any(os.path.join(work, "new", "swaps_logs", "wandb_log.json")))
