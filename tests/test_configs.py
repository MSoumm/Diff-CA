"""The released configs hold the values of the configs that produced the paper results: every key the code
reads is unchanged, except paths (data, outputs, pretrained weights)."""

import os

import pytest
import yaml

from conftest import ORIG_SRC, requires_orig

pytestmark = requires_orig
CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")

PATH_KEYS = {"output_dir", "dataset_path", "cache_paths", "config_path", "pretrained_path", "backbone_path",
             "train_pt_folder", "test_pt_folder", "json_folder", "sublabels_csv", "labels_train", "labels_test",
             "referee_classifier_path", "config_file", "save_dir", "train_path", "test_path", "output_path",
             "ae_path", "dino_path"}
# Read by no code (dropped from the released files).
UNUSED_STAGE1 = {"model.type", "model.image_encoder.backbone_type", "model.image_encoder.frozen_backbone",
                 "training.warmup_steps", "sampling.method"}
UNUSED_STAGE2 = {"lambda_independence", "beta", "lambda_swap_adv", "lambda_regressor", "contrastive", "tau",
                 "lambda_diversity", "lambda_push", "push_target", "swap_low_target_acc", "swap_high_target_acc",
                 "job_start_time", "job_deadline_hours"}

STAGE1 = {"ffhq_cqc": "FFHQ_color", "ffhq_cq": "FFHQ", "ffhq_dinov3": "FFHQ_DINO", "ffhq_clip": "FFHQ_CLIP",
          "ffhq_encdiff": "FFHQ_EncDiff", "afhq_cqc": "AFHQ", "brats_cqc": "BraTS2023_cropped"}
STAGE2 = {"glasses": "glasses", "smile": "smile", "headpose": "yaw", "afhq": "afhq", "brats": "brats"}
PREPROCESS = {"ffhq": "FFHQ", "celebahq": "CelebA-HQ", "ffhq_clip": "FFHQ_CLIP", "celebahq_clip": "CelebA-HQ_CLIP",
              "afhq": "AFHQ", "brats": "BraTS2023_cropped"}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if k in PATH_KEYS:
            continue
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("new,orig", STAGE1.items())
def test_stage1_configs(new, orig):
    n = flatten(load(f"{CONFIGS}/stage1/{new}.yaml"))
    o = flatten(load(f"{ORIG_SRC}/cqae/diffusion_configs/{orig}.yaml"))
    o.setdefault("augmentations.drop_dino", False)  # absent = False
    o = {k: v for k, v in o.items() if k not in UNUSED_STAGE1}
    assert ("brats" in n.pop("experiment_id")) == ("brats" in o.pop("experiment_id"))  # the only use of the id
    assert n == o


@pytest.mark.parametrize("new,orig", STAGE2.items())
def test_stage2_configs(new, orig):
    n = flatten(load(f"{CONFIGS}/stage2/{new}.yaml"))
    o = {k: v for k, v in flatten(load(f"{ORIG_SRC}/cqae/separator_configs/{orig}.yaml")).items() if k not in UNUSED_STAGE2}
    if o["attribute"] == "headpose":
        o["attribute"] = "headPose"  # normalized by both versions
    assert n == o


@pytest.mark.parametrize("new,orig", PREPROCESS.items())
def test_preprocess_configs(new, orig):
    n = flatten(load(f"{CONFIGS}/preprocess/{new}.yaml"))
    o = flatten(load(f"{ORIG_SRC}/cqae/preprocess_configs/{orig}.yaml"))
    if new == "celebahq_clip":
        assert o.pop("save_dino") is False and n.pop("save_dino") is True  # needed by ffhq_clip.yaml (CLIP tokens)
    assert n == o
