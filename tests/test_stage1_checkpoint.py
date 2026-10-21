"""A real Stage 1 checkpoint (DIFFCA_STAGE1_CHECKPOINT=<run>/checkpoint-<step>, config_saved.yaml in <run>) loads
into the cleaned model exactly as into the original one: strict state dict, the positional EMA weights land on
the same parameters, and the encoder and the U-Net give identical outputs."""

import copy
import os

import pytest
import torch
import yaml
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file

from conftest import assert_same, requires_orig

CHECKPOINT = os.environ.get("DIFFCA_STAGE1_CHECKPOINT")
pytestmark = [requires_orig, pytest.mark.skipif(not CHECKPOINT, reason="DIFFCA_STAGE1_CHECKPOINT not set")]


def _build(impl, config):
    torch.manual_seed(0)
    unet_config = config["model"]["unet_config"]
    encoder_conf = copy.deepcopy(config["model"]["image_encoder"])
    encoder_type = encoder_conf.pop("type")
    if impl == "orig":
        from cqae.ae.models import DiffusionGenerator, TrainableWrapper, str_to_cls as encoders
    else:
        from diffca.stage1.models import ENCODERS as encoders, DiffusionGenerator, TrainableWrapper
    diffusion = DiffusionGenerator(unet_config=unet_config)
    encoder = encoders[encoder_type](num_input_channels=unet_config["in_channels"], **encoder_conf)
    # As in training: compiled parts inside the wrapper (compilation itself only happens at the first call).
    return TrainableWrapper(torch.compile(diffusion), torch.compile(encoder))


@pytest.fixture(scope="module")
def loaded():
    with open(os.path.join(os.path.dirname(CHECKPOINT.rstrip("/")), "config_saved.yaml")) as f:
        config = yaml.safe_load(f)
    state = load_file(os.path.join(CHECKPOINT, "model.safetensors"))
    ema_state = torch.load(os.path.join(CHECKPOINT, "custom_checkpoint_0.pkl"), map_location="cpu", weights_only=False)
    models = {}
    for impl in ("orig", "new"):
        wrapper = _build(impl, config)
        assert [k for k, _ in wrapper.named_parameters()] == [k for k, _ in _build("orig", config).named_parameters()]
        wrapper.load_state_dict(state, strict=True)
        ema = EMAModel(wrapper.parameters(), decay=0.9999)
        ema.load_state_dict(copy.deepcopy(ema_state))
        ema.copy_to(wrapper.parameters())
        models[impl] = wrapper.eval()
    return config, models


def test_state_dict_and_ema_identical(loaded):
    _, models = loaded
    o, n = models["orig"].state_dict(), models["new"].state_dict()
    assert list(o) == list(n)
    assert_same(o, n)


def test_outputs_identical(loaded):
    config, models = loaded
    g = torch.Generator().manual_seed(0)
    unet = config["model"]["unet_config"]
    dino = torch.randn(2, 261, 768, generator=g)
    latents = torch.randn(2, unet["in_channels"], unet["sample_size"], unet["sample_size"], generator=g)
    t = torch.tensor([0.1, 0.7])
    outputs = {}
    with torch.no_grad():
        for impl, wrapper in models.items():
            encoder = wrapper.encoder._orig_mod
            tokens = encoder.encode(backbone_embeds=dino, latents=latents, images=None)
            velocity = wrapper.diffusion_model._orig_mod(latents, t, tokens)
            outputs[impl] = (tokens, velocity)
    assert outputs["new"][0].shape == (2, config["model"]["image_encoder"]["n_tokens"], 128)
    assert_same(outputs["orig"], outputs["new"])
