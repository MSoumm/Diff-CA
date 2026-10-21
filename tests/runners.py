"""Run one pipeline step with the original (`orig`) or the cleaned (`new`) code, in a fresh process.

    python runners.py <task> <orig|new> <out_dir> [args...]

Patches applied identically to both implementations, so that the comparison runs on a CPU in reasonable time:
  - everything runs on the CPU, and VAE outputs are made contiguous (layout only);
  - torch.compile uses the "eager" backend (same parameter names, `_orig_mod.` included; eager numerics);
  - DataLoaders run in the main process (num_workers=0, no pinned memory);
  - the image metrics of Stage 1 (LPIPS, SSIM, DISTS, Inception FID/KID) are replaced by cheap deterministic
    stand-ins with the same interface (they would need large pretrained downloads); the code around them
    (sampling, aggregation, FID/KID formulas) runs unchanged;
  - wandb runs disabled and every `wandb.log` call is recorded to <out_dir>/wandb_log.json;
  - (original only) hard-coded cluster paths of DINOv3 are redirected to the tiny test backbone.
"""

import json
import os
import sys

import numpy as np
import torch

TASK, IMPL, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
ARGS = sys.argv[4:]
ORIG_SRC = os.environ["DIFFCA_ORIG_SRC"]
TINY_BACKBONE = os.environ.get("DIFFCA_TINY_BACKBONE")
sys.path.insert(0, ORIG_SRC if IMPL == "orig" else os.path.join(os.path.dirname(__file__), "..", "src"))
os.makedirs(OUT_DIR, exist_ok=True)


# --- CPU only (Accelerate would pick the Apple GPU) ---
os.environ["ACCELERATE_USE_CPU"] = "true"
torch.backends.mps.is_available = lambda: False

# --- Contiguous VAE outputs: on this CPU backend the VAE convolutions return channels-last tensors, on which the
#     original `.view(...)` calls fail (they do not on CUDA). Same values, standard memory layout. ---
from diffusers import VQModel  # noqa: E402
from diffusers.models.autoencoders import vae as _vae_module  # noqa: E402

_real_dgd_init = _vae_module.DiagonalGaussianDistribution.__init__
_vae_module.DiagonalGaussianDistribution.__init__ = lambda self, parameters, *a, **k: _real_dgd_init(self, parameters.contiguous(), *a, **k)
_real_vq_encode = VQModel.encode


def _vq_encode(self, *a, **k):
    out = _real_vq_encode(self, *a, **k)
    out.latents = out.latents.contiguous()
    return out


VQModel.encode = _vq_encode

# --- torch.compile -> eager backend ---
_real_compile = torch.compile
torch.compile = lambda model, *a, **k: _real_compile(model, backend="eager")

# --- DataLoader in the main process ---
_real_dl_init = torch.utils.data.DataLoader.__init__


def _dl_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["pin_memory"] = False
    kwargs.pop("prefetch_factor", None)
    kwargs.pop("persistent_workers", None)
    if len(args) > 6:  # num_workers given positionally (not the case in this code base)
        raise RuntimeError("unexpected positional DataLoader arguments")
    _real_dl_init(self, *args, **kwargs)


torch.utils.data.DataLoader.__init__ = _dl_init

# --- wandb: disabled, calls recorded ---
import wandb  # noqa: E402

WANDB_LOG = []
_real_wandb_log = wandb.log


def _to_jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(type(v).__name__)


def _record_log(data, step=None, **kwargs):
    WANDB_LOG.append({"step": step, "data": {k: _to_jsonable(v) for k, v in data.items()}})


wandb.log = _record_log
_real_wandb_init = wandb.init


def _wandb_init(*a, **k):
    run = _real_wandb_init(*a, **k)
    wandb.log = _record_log  # wandb.init rebinds wandb.log
    return run


wandb.init = _wandb_init
os.environ["WANDB_MODE"] = "disabled"


def dump(name, obj):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_to_jsonable)


# --- Stand-in image metrics ---
class _FakeModuleMetric:
    def to(self, *a, **k):
        return self

    def float(self):
        return self


class FakeLPIPS(_FakeModuleMetric):
    def __init__(self, net_type="alex", reduction="mean", **kwargs):
        self.scale = 1.0 if net_type == "alex" else 2.0

    def __call__(self, a, b):
        return (a - b).abs().mean(dim=[1, 2, 3]) * self.scale


class FakeSSIM(_FakeModuleMetric):
    def __init__(self, data_range=2.0, reduction="none", **kwargs):
        pass

    def __call__(self, a, b):
        return 1 - ((a - b) ** 2).mean(dim=[1, 2, 3])


def fake_dists(a, b):
    return (a.mean(dim=[2, 3]) - b.mean(dim=[2, 3])).abs().mean(dim=1)


def _image_features(feature, imgs):
    if isinstance(feature, int):
        x = imgs.float()
        return torch.cat([x.mean(dim=[2, 3]), x.std(dim=[2, 3])], dim=1).double()
    return feature(imgs)[:, :6].double()


class FakeFID(_FakeModuleMetric):
    def __init__(self, feature=2048, normalize=False, **kwargs):
        self.feature = feature
        self.reset()

    def reset(self):
        self.real_features_sum = self.fake_features_sum = None
        self.real_features_cov_sum = self.fake_features_cov_sum = None
        self.real_features_num_samples = torch.tensor(0, dtype=torch.long)
        self.fake_features_num_samples = torch.tensor(0, dtype=torch.long)

    def update(self, imgs, real):
        feats = _image_features(self.feature, imgs)
        prefix = "real" if real else "fake"
        s, c = getattr(self, f"{prefix}_features_sum"), getattr(self, f"{prefix}_features_cov_sum")
        setattr(self, f"{prefix}_features_sum", feats.sum(0) if s is None else s + feats.sum(0))
        setattr(self, f"{prefix}_features_cov_sum", feats.t().mm(feats) if c is None else c + feats.t().mm(feats))
        setattr(self, f"{prefix}_features_num_samples", getattr(self, f"{prefix}_features_num_samples") + feats.shape[0])


class FakeKID(_FakeModuleMetric):
    def __init__(self, feature=2048, subsets=10, subset_size=1000, degree=3, gamma=None, coef=1.0, **kwargs):
        self.subsets, self.subset_size, self.degree, self.gamma, self.coef = subsets, subset_size, degree, gamma, coef
        self.reset()

    def reset(self):
        self.real_features, self.fake_features = [], []

    def update(self, imgs, real):
        (self.real_features if real else self.fake_features).append(_image_features(2048, imgs))


def patch_metrics(module):
    module.LearnedPerceptualImagePatchSimilarity = FakeLPIPS
    module.StructuralSimilarityIndexMeasure = FakeSSIM
    module.deep_image_structure_and_texture_similarity = fake_dists
    module.FrechetInceptionDistance = FakeFID
    module.KernelInceptionDistance = FakeKID


def redirect_backbone_paths():
    """The original Stage 1 checkpoint evaluation loads DINOv3 from a hard-coded cluster path."""
    import transformers
    real = transformers.AutoModel.from_pretrained.__func__

    def from_pretrained(cls, path, *a, **k):
        if isinstance(path, str) and path.startswith("/lustre"):
            path = TINY_BACKBONE
        return real(cls, path, *a, **k)

    transformers.AutoModel.from_pretrained = classmethod(from_pretrained)


def run_with_argv(main, argv):
    sys.argv = ["prog"] + list(argv)
    return main()


# ---------------------------------------------------------------------------

if TASK == "stage1_train":
    if IMPL == "orig":
        import cqae.ae.evaluate as ev
        import cqae.ae.train as mod
        # The original final save calls get_state_dict(unwrap=True), which raises in accelerate 1.8.1 for a
        # module holding compiled submodules. Without unwrapping it saves what it was meant to save.
        from accelerate import Accelerator
        _real_get_state_dict = Accelerator.get_state_dict
        Accelerator.get_state_dict = lambda self, model, unwrap=True: _real_get_state_dict(self, model, unwrap=False)
    else:
        import diffca.stage1.evaluate as ev
        import diffca.stage1.train as mod
    patch_metrics(ev)
    results = []
    real_evaluate = mod.evaluate

    def recording_evaluate(*a, **k):
        res = real_evaluate(*a, **k)
        results.append(res)
        return res

    mod.evaluate = recording_evaluate
    run_with_argv(mod.main, ARGS)
    dump("eval_results.json", results)

elif TASK == "stage1_extract":
    mod = __import__("cqae.features.extract_features" if IMPL == "orig" else "diffca.stage1.extract_features",
                     fromlist=["main"])
    run_with_argv(mod.main, ARGS)

elif TASK == "stage1_evaluate_checkpoint":
    if IMPL == "orig":
        redirect_backbone_paths()
        import cqae.ae.evaluate as ev
        import cqae.ae.evaluation.evaluate_model as mod
    else:
        import diffca.stage1.evaluate as ev
        import diffca.stage1.evaluate_checkpoint as mod
    patch_metrics(ev)
    results = []
    real_evaluate = mod.evaluate
    mod.evaluate = lambda *a, **k: results.append(real_evaluate(*a, **k))
    run_with_argv(mod.main, ARGS)
    dump("eval_results.json", results)

elif TASK == "stage2_train":
    if IMPL == "orig":
        import cqae.disentangle.dis_train_cycle_2 as mod
    else:
        import diffca.stage2.train as mod
    # Fewer evaluation pairs and interpolation grids than the hard-coded 1000 / 100, for both implementations.
    max_pairs = int(os.environ.get("DIFFCA_MAX_PAIRS", "0"))
    if max_pairs:
        real_create, real_interp = mod.create_evaluation_dataloaders, mod.generate_interpolation_grid

        def create(*a, max_pair_images=None, **k):
            return real_create(*a, max_pair_images=min(max_pair_images or 1000, max_pairs), **k)

        def interp(*a, max_grids=3, **k):
            return real_interp(*a, max_grids=min(max_grids, 4), **k)

        mod.create_evaluation_dataloaders, mod.generate_interpolation_grid = create, interp
    best = run_with_argv(mod.train, ARGS)
    dump("best_metrics.json", best)
    dump("wandb_log.json", WANDB_LOG)

elif TASK == "classifier":
    attribute = ARGS[ARGS.index("--attribute") + 1]
    if IMPL == "orig":
        script = {"glasses": "classifier_train", "smile": "classifier_smile", "headPose": "classifier_headpose",
                  "afhq": "classifier_afhq", "brats": "classifier_brats"}[attribute]
        mod = __import__(f"cqae.disentangle.{script}", fromlist=["main"])
        argv = list(ARGS)
        if attribute in ("afhq", "brats"):  # these scripts have no --attribute flag
            i = argv.index("--attribute")
            del argv[i:i + 2]
    else:
        import diffca.stage2.train_classifier as mod
        argv = ARGS
    run_with_argv(mod.main, argv)

elif TASK == "preprocess":
    mod = __import__("cqae.preprocess_committed" if IMPL == "orig" else "diffca.stage1.preprocess", fromlist=["main"])
    run_with_argv(mod.main, ARGS)

else:
    raise SystemExit(f"unknown task {TASK}")
