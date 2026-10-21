"""Stage 2 building blocks, original vs cleaned code, in-process: separator, losses and GRL schedule, datasets,
evaluation loaders and protocol, swap / interpolation decoding, projections, configs, referee classifiers."""

import copy
import os

import pytest
import torch

from conftest import assert_same, assert_same_tree, requires_orig

pytestmark = requires_orig


def _orig():
    from cqae.disentangle import dis_loss, dis_model
    return dis_model, dis_loss


def _new():
    from diffca.stage2 import losses, model
    return model, losses


def build_separator(mod, seed, layers=2, ema_decay=0.0):
    torch.manual_seed(seed)
    wrapper = mod.CycleConsistencyWrapper(mod.TokenDisentangler(
        d_model=128, num_encoder_layers=layers, num_decoder_layers=0, dropout=0.1, ema_decay=ema_decay,
        cycle_ema_decay=0.99, low_rank=128))
    adversary = mod.AdversaryMLP(num_tokens=32, token_dim=128, hidden_dim=512, dropout_rate=0.2)
    return wrapper, adversary


# ---------------------------------------------------------------------------
# Separator and losses
# ---------------------------------------------------------------------------

def test_separator_state_dict_layout():
    (om, _), (nm, _) = _orig(), _new()
    ow, oadv = build_separator(om, 0, layers=5)
    nw, nadv = build_separator(nm, 0, layers=5)
    for o, n in ((ow, nw), (oadv, nadv)):
        assert list(o.state_dict()) == list(n.state_dict())
        assert [k for k, _ in o.named_parameters()] == [k for k, _ in n.named_parameters()]
        assert [k for k, _ in o.named_buffers()] == [k for k, _ in n.named_buffers()]
        assert_same(o.state_dict(), n.state_dict())
    # The paper separator: 1.53M trainable parameters (5 layers, rank 128), 2.10M for the adversary.
    trainable = sum(p.numel() for p in nw.parameters() if p.requires_grad)
    assert round(trainable / 1e6, 2) == 1.53
    assert round(sum(p.numel() for p in nadv.parameters()) / 1e6, 2) == 2.10


LOSS_KW = dict(lambda_recon=10.0, lambda_norm_pos=0.1, lambda_norm_neg=1, lambda_cycle_context=1, lambda_cycle_cls=1,
               lambda_adv=1, low_target_acc=0.75, high_target_acc=0.75)
OLD_ONLY_KW = dict(beta=0.0, lambda_push=0, push_target=0.2, lambda_independence=0, tau=0, lambda_swap_adv=0)


@pytest.mark.parametrize("batch,ema_decay,overrides,autocast", [
    (17, 0.0, {}, False),                                   # paper settings, odd batch
    (16, 0.999, {"lambda_cycle_context": 0.06, "lambda_cycle_cls": 3.33, "lambda_adv": 0.05}, False),
    (12, 0.0, {"lambda_adv": 0, "lambda_cycle_context": 0}, False),
    (16, 0.0, {}, True),                                    # bf16 autocast, as in the paper runs
])
def test_training_steps_bitwise_identical(batch, ema_decay, overrides, autocast):
    (om, ol), (nm, nl) = _orig(), _new()
    runs = {}
    for tag, mod, loss_mod, extra in (("orig", om, ol, OLD_ONLY_KW), ("new", nm, nl, {})):
        wrapper, adversary = build_separator(mod, 0, ema_decay=ema_decay)
        opt = torch.optim.AdamW(list(wrapper.parameters()) + list(adversary.parameters()), lr=1e-3, betas=(0.9, 0.99))
        g = torch.Generator().manual_seed(1)
        steps_per_epoch, epochs = 5, 8
        trace = []
        for step in range(steps_per_epoch * epochs):
            x = torch.randn(batch, 32, 128, generator=g)
            y = (torch.rand(batch, generator=g) > 0.4).long()
            if step == 7:
                y[:] = 1  # a batch without background samples
            torch.manual_seed(1000 + step)
            wrapper.train()
            adversary.train()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out = wrapper(x, labels=y, cycle=True)
                loss, comp = loss_mod.cycle_loss_criterion(out, joint_adversary=adversary, total_step=step,
                                                           is_training=True, steps_per_epoch=steps_per_epoch,
                                                           total_epochs=epochs, **{**LOSS_KW, **overrides, **extra})
            opt.zero_grad()
            loss.backward()
            grads = [p.grad.clone() if p.grad is not None else None for p in list(wrapper.parameters()) + list(adversary.parameters())]
            torch.nn.utils.clip_grad_norm_(list(wrapper.parameters()) + list(adversary.parameters()), 1.0)
            opt.step()
            wrapper.update_ema()
            wrapper.update_cycle_ema()
            trace.append((loss.detach(), comp, grads, copy.deepcopy(adversary.state_dict())))
        wrapper.eval()
        with torch.no_grad():
            xe = torch.randn(9, 32, 128, generator=g)
            val = loss_mod.cycle_loss_criterion(wrapper(xe, labels=(torch.arange(9) % 2), cycle=True),
                                                joint_adversary=adversary, total_step=999, is_training=False,
                                                steps_per_epoch=5, total_epochs=8, **{**LOSS_KW, **overrides, **extra})
        runs[tag] = (trace, wrapper.state_dict(), val)

    (o_trace, o_sd, o_val), (n_trace, n_sd, n_val) = runs["orig"], runs["new"]
    for step, ((ol_, oc, og, oa), (nl_, nc, ng, na)) in enumerate(zip(o_trace, n_trace)):
        assert torch.equal(ol_, nl_), f"loss differs at step {step}"
        assert_same(og, ng, f"grads@{step}")
        assert_same(oa, na, f"adversary buffers@{step}")
        assert set(nc) <= set(oc)
        assert_same({k: oc[k] for k in nc}, nc, f"components@{step}")
    assert_same(o_sd, n_sd)
    assert torch.equal(o_val[0], n_val[0])
    assert_same({k: o_val[1][k] for k in n_val[1]}, n_val[1])
    assert_same(o_val[2], n_val[2])
    # removed components: logging of removed options only
    assert set(o_trace[0][1]) - set(n_trace[0][1]) == {
        "loss_kl", "push_loss", "positive Similarity", "lambda_push_dynamic", "ema_push_loss", "adv_integral_error",
        "positive_sampled_norms", "negative_sampled_norms"}


def test_eval_outputs_and_swap_decoding_identical():
    (om, _), (nm, _) = _orig(), _new()
    ow, _ = build_separator(om, 3, ema_decay=0.5)
    nw, _ = build_separator(nm, 3, ema_decay=0.5)
    g = torch.Generator().manual_seed(5)
    for w in (ow, nw):  # make the EMA copies differ from the online weights
        w.train()
        for p in w.parameters():
            if p.requires_grad:
                p.data.add_(0.01)
        w.update_ema()
    x = torch.randn(10, 32, 128, generator=g)
    for use_ema in (False, True):
        ow.eval(), nw.eval()
        with torch.no_grad():
            o = ow(x, use_ema=use_ema)["standard_outputs"]
            n = nw(x, use_ema=use_ema)["standard_outputs"]
            assert_same([o["reconstructed_tokens"], o["cls_mean"], o["sampled_cls"], o["content_tokens"], o["predicted_y"]],
                        [n["reconstructed_tokens"], n["cls"], n["cls"], n["content_tokens"], n["predicted_y"]])
            assert_same(ow.model.decoder(o["content_tokens"], o["cls_mean"].flip(0).unsqueeze(1), use_ema=use_ema),
                        nw.model.decoder(n["content_tokens"], n["cls"].flip(0).unsqueeze(1), use_ema=use_ema))


# ---------------------------------------------------------------------------
# Datasets and evaluation loaders
# ---------------------------------------------------------------------------

def _dataset_pairs(fixtures_root):
    from cqae.data import datasets as od
    from diffca.stage2 import datasets as nd
    r = fixtures_root
    pairs = {}
    for attribute in ("glasses", "smile", "headPose"):
        for split in ("train", "test"):
            for ram in (True, False):
                if split == "train" and not ram:
                    continue
                kw = dict(pt_folder_path=f"{r}/tokens_ffhq/{split}", json_folder_path=f"{r}/ffhq_json", attribute=attribute,
                          load_features_in_ram=ram)
                pairs[(attribute, split, ram)] = (od.FeatureAttributeDataset(**kw), nd.FeatureAttributeDataset(**kw))
    for split in ("train", "test"):
        kw = dict(pt_folder_path=f"{r}/tokens_brats/{split}", labels_path=f"{r}/brats_{split}_labels.txt", load_features_in_ram=True)
        pairs[("brats", split, True)] = (od.BratsFeatureDataset(**kw), nd.BratsFeatureDataset(**kw))
        for soft in (False, True):
            kw = dict(pt_folder_path=f"{r}/tokens_afhq/{split}", sublabels_csv_path=f"{r}/afhq_breeds.csv",
                      load_features_in_ram=True, return_soft_targets=soft)
            pairs[("afhq", split, soft)] = (od.AfhqFeatureDataset(**kw), nd.AfhqFeatureDataset(**kw))
    return pairs


@pytest.fixture(scope="module")
def dataset_pairs(fixtures_root):
    return _dataset_pairs(fixtures_root)


def test_datasets_identical(dataset_pairs):
    for key, (o, n) in dataset_pairs.items():
        assert len(o) == len(n) > 0, key
        assert [s["feature_file"] for s in o.samples] == [s["feature_file"] for s in n.samples], key
        for i in range(len(o)):
            assert_same(tuple(o[i]), tuple(n[i]), f"{key}[{i}]")
    # every label branch is exercised
    glasses = dataset_pairs[("glasses", "test", True)][1]
    assert {s["sublabel"].item() for s in glasses.samples} == {0, 1, 2}
    assert len(dataset_pairs[("glasses", "train", True)][1]) < 40  # swimming goggles and the unreadable file dropped


@pytest.mark.parametrize("attribute,split_key", [("glasses", ("glasses", "test", True)), ("smile", ("smile", "test", True)),
                                                 ("headPose", ("headPose", "test", True)), ("afhq", ("afhq", "test", False)),
                                                 ("brats", ("brats", "test", True))])
def test_evaluation_loaders_identical(dataset_pairs, attribute, split_key):
    from torch.utils.data import Subset
    from cqae.data.datasets import create_evaluation_dataloaders as o_create
    from diffca.stage2.datasets import create_evaluation_dataloaders as n_create
    o_ds, n_ds = dataset_pairs[split_key]
    ol = o_create(Subset(o_ds, range(len(o_ds))), o_ds, 8, attribute=attribute)
    nl = n_create(Subset(n_ds, range(len(n_ds))), n_ds, 8, attribute=attribute)
    assert set(nl) == set(ol) - {"intra_class"}
    assert ol["class_ids"] == nl["class_ids"] and list(ol["pairs"]) == list(nl["pairs"])
    loaders = [(ol["full"], nl["full"])] + [(ol["pairs"][k], nl["pairs"][k]) for k in ol["pairs"]]
    loaders += [(ol[k], nl[k]) for k in ("zero_one", "style") if k in ol]
    for o_loader, n_loader in loaders:
        assert list(o_loader.batch_sampler) == list(n_loader.batch_sampler)
        for ob, nb in zip(o_loader, n_loader):
            assert_same(ob, nb)


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------

def _referees(attribute):
    from cqae.disentangle.classifier_afhq import AfhqMLP
    from cqae.disentangle.classifier_brats import BratsMLP
    from cqae.disentangle.classifier_headpose import HeadPoseMLP
    from cqae.disentangle.classifier_train import SimpleMLP
    from diffca.stage2.classifiers import build_referee
    torch.manual_seed(11)
    orig = {"glasses": lambda: SimpleMLP(num_tokens=32, num_classes=3), "smile": lambda: SimpleMLP(num_tokens=32, num_classes=2),
            "headPose": lambda: HeadPoseMLP(num_tokens=32), "afhq": lambda: AfhqMLP(num_tokens=32, num_classes=4),
            "brats": lambda: BratsMLP(num_tokens=32, num_classes=2)}[attribute]()
    torch.manual_seed(11)
    new = build_referee(attribute, num_tokens=32)
    assert_same(orig.state_dict(), new.state_dict())
    return orig.eval(), new.eval()


def _best_metrics_states(attribute):
    from cqae.disentangle.dis_train_cycle_2 import BestMetricsManager as OBM
    from diffca.stage2.train import BestMetricsManager as NBM
    o, n = OBM(attribute).state, NBM(attribute).state
    assert_same(o, n)
    return o, n


@pytest.mark.parametrize("attribute,split_key", [("glasses", ("glasses", "test", True)), ("smile", ("smile", "test", True)),
                                                 ("headPose", ("headPose", "test", True)), ("afhq", ("afhq", "test", False)),
                                                 ("brats", ("brats", "test", True))])
@pytest.mark.parametrize("with_referee", [True, False])
def test_evaluation_protocol_identical(dataset_pairs, tmp_path, attribute, split_key, with_referee):
    from torch.utils.data import Subset
    from cqae.data.datasets import create_evaluation_dataloaders as o_create
    from cqae.disentangle.dis_evaluate import run_evaluation_protocol as o_run
    from diffca.stage2.datasets import create_evaluation_dataloaders as n_create
    from diffca.stage2.evaluate import run_evaluation_protocol as n_run
    (om, _), (nm, _) = _orig(), _new()
    ow, _ = build_separator(om, 4)
    nw, _ = build_separator(nm, 4)
    o_ref, n_ref = _referees(attribute) if with_referee else (None, None)
    o_ds, n_ds = dataset_pairs[split_key]
    o_best, n_best = _best_metrics_states(attribute)
    results = {}
    for tag, run, create, w, ref, ds, best in (("orig", o_run, o_create, ow, o_ref, o_ds, o_best),
                                               ("new", n_run, n_create, nw, n_ref, n_ds, n_best)):
        loaders = create(Subset(ds, range(len(ds))), ds, 8, attribute=attribute)
        for epoch in range(2):
            torch.manual_seed(epoch)
            log, best = run(w, loaders, ref, "cpu", best, epoch, use_ema=True, attribute=attribute,
                            pair_names_dir=str(tmp_path / tag / f"names{epoch}"))
            with torch.no_grad():  # change the model between the two epochs
                for p in w.parameters():
                    p.add_(0.003)
        results[tag] = (log, best, torch.rand(3))
    assert_same(results["orig"], results["new"])  # logs, best metrics, and the RNG state after the protocol
    if with_referee:
        assert_same_tree(str(tmp_path / "orig"), str(tmp_path / "new"))


# ---------------------------------------------------------------------------
# Image generation: swaps and interpolations
# ---------------------------------------------------------------------------

def _tiny_generator(fixtures_root):
    from diffusers import VQModel
    from fixtures import TINY_UNET
    from diffca.stage1.models import DiffusionGenerator
    torch.manual_seed(21)
    return DiffusionGenerator(TINY_UNET).eval(), VQModel.from_pretrained(f"{fixtures_root}/vq").eval()


def test_swap_and_interpolation_images_identical(dataset_pairs, fixtures_root, tmp_path):
    from torch.utils.data import Subset
    from cqae.data.datasets import create_evaluation_dataloaders as o_create
    from cqae.disentangle import dis_evaluate as oe
    from diffca.stage2 import generate as ng
    from diffca.stage2.datasets import create_evaluation_dataloaders as n_create
    from fixtures import TINY_UNET
    (om, _), (nm, _) = _orig(), _new()
    ow, _ = build_separator(om, 6)
    nw, _ = build_separator(nm, 6)
    diffusion, vae = _tiny_generator(fixtures_root)
    config = {"model": {"unet_config": TINY_UNET}, "sampling": {"num_inference_steps": 2}}
    o_ds, n_ds = dataset_pairs[("glasses", "test", True)]
    for tag, create, w, ds, swap_fn, interp_fn in (
            ("orig", o_create, ow, o_ds, oe.generate_all_swapped_images, oe.generate_interpolation_grid),
            ("new", n_create, nw, n_ds, ng.generate_all_swapped_images, ng.generate_interpolation_grid)):
        loaders = create(Subset(ds, range(len(ds))), ds, 4, attribute="glasses", max_pair_images=6)
        torch.manual_seed(0)
        interp_fn(w, loaders, diffusion, vae, config, "cpu", 1.37, str(tmp_path / tag / "interp"), use_ema=True,
                  max_grids=3, num_negatives=2, negative_class=1, positive_class=2, num_steps=3)
        swap_fn(w, loaders, diffusion, vae, config, "cpu", 1.37, str(tmp_path / tag / "swaps"), use_ema=True)
    files = assert_same_tree(str(tmp_path / "orig"), str(tmp_path / "new"))
    assert any("swap_y2x" in f for f in files) and any("interpolation_grid" in f for f in files)


# ---------------------------------------------------------------------------
# Projections (paper App. G figure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["pca", "umap"])
@pytest.mark.parametrize("yaw", [False, True])
def test_projection_figures_identical(tmp_path, method, yaw):
    from cqae.disentangle import dis_evaluate as oe
    from diffca.stage2.projections import generate_projections
    g = torch.Generator().manual_seed(8)
    labels = [(torch.arange(40) % 2)]
    sublabels = [torch.randn(40, generator=g) * 20] if yaw else [torch.arange(40) % 3]
    cls = [torch.randn(40, 128, generator=g)]
    context = [torch.randn(40, 128, generator=g)]
    import wandb
    real_log = wandb.log
    wandb.log = lambda *a, **k: None  # the original logs PCA figures to wandb
    try:
        fn = oe.generate_pca_visualizations if method == "pca" else oe.generate_umap_visualizations
        fn(labels, sublabels, cls, cls, context, 3, str(tmp_path / "orig"), 42)
    finally:
        wandb.log = real_log
    generate_projections(method, labels, sublabels, cls, context, 3, str(tmp_path / "new"), 42)
    files = assert_same_tree(str(tmp_path / "orig"), str(tmp_path / "new"), ignore=lambda rel: rel.endswith(".pdf"))
    assert len(files) == (3 if yaw else 4)


# ---------------------------------------------------------------------------
# Configs and referee checkpoints
# ---------------------------------------------------------------------------

def _paper_yaml(tmp_path, name):
    import yaml
    path = os.path.join(os.environ["DIFFCA_ORIG_SRC"], "cqae", "separator_configs", f"{name}.yaml")
    config = yaml.safe_load(open(path))
    for key in ("train_pt_folder", "test_pt_folder", "json_folder", "sublabels_csv", "labels_train", "labels_test"):
        if key in config:
            config[key] = f"/data/{key}"
    out = tmp_path / f"{name}.yaml"
    out.write_text(yaml.safe_dump(config))
    return str(out)


@pytest.mark.parametrize("name", ["glasses", "smile", "yaw", "afhq", "brats"])
def test_paper_separator_configs_parse_identically(tmp_path, name):
    from cqae.config import get_config as o_get
    from cqae.disentangle.dis_train_cycle_2 import _apply_yaml_config, _normalize_attribute
    from diffca.stage2.config import get_config as n_get
    path = _paper_yaml(tmp_path, name)
    argv = ["--config_yaml", path, "--epochs", "7"]
    o = _apply_yaml_config(o_get(argv), argv)
    o.attribute = _normalize_attribute(o.attribute)
    n = n_get(argv)
    assert n.epochs == 7
    for key, value in vars(n).items():
        assert getattr(o, key) == value, key


def test_removed_option_is_rejected(tmp_path):
    from diffca.stage2.config import get_config
    path = _paper_yaml(tmp_path, "glasses_smile")  # uses lambda_push: 0.1 (removed) and glasses+smile
    with pytest.raises((ValueError, SystemExit)):
        get_config(["--config_yaml", path])


@pytest.mark.skipif(not os.environ.get("DIFFCA_CLASSIFIER_DIR"), reason="DIFFCA_CLASSIFIER_DIR not set")
@pytest.mark.parametrize("attribute,filename", [("glasses", "glasses_color_aug_OT_EMA.pt"), ("smile", "smile_color_aug_OT_EMA.pt"),
                                                ("headPose", "headPose_color_aug_OT_EMA.pt"), ("afhq", "afhq_color_aug_OT_EMA.pt"),
                                                ("brats", "brats_color_aug_OT_EMA.pt")])
def test_trained_referee_checkpoints_load(attribute, filename):
    o, n = _referees(attribute)
    sd = torch.load(os.path.join(os.environ["DIFFCA_CLASSIFIER_DIR"], filename), map_location="cpu")
    o.load_state_dict(sd)
    n.load_state_dict(sd)  # strict
    x = torch.randn(16, 32, 128, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert_same(o(x), n(x))
