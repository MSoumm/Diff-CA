# Equivalence tests

This code is a cleanup of the research code that produced the paper. These tests run the original code and this code
side by side, on a CPU, with tiny models and synthetic data. They check that both versions produce exactly the same
outputs, bit for bit: weights, EMA copies, optimizer and scheduler states, RNG states, losses, gradients, logged
metrics, extracted tokens and generated images. They need the original code:

```bash
pip install pytest
DIFFCA_ORIG_SRC=/path/to/original/src pytest tests
```

`DIFFCA_ORIG_SRC` is the folder containing the original `cqae` package. Without it, the tests are skipped. The tiny
models and data are built on first use in `$DIFFCA_FIXTURES` (default: a temporary folder). The full run takes about
10 minutes.

Optional, to also test real checkpoints:
- `DIFFCA_STAGE1_CHECKPOINT=<run>/checkpoint-<step>` (with `config_saved.yaml` in `<run>`): the checkpoint loads
  strictly, the EMA weights land on the same parameters, and the encoder and U-Net outputs are identical.
- `DIFFCA_CLASSIFIER_DIR=<folder of referee checkpoints>`: they load strictly and give identical outputs.

| File | What it compares |
|---|---|
| `test_stage1_pipeline.py` | Stage 1 training (5 encoder / data settings, incl. online features with augmentations), feature extraction, checkpoint evaluation, preprocessing |
| `test_stage1_checkpoint.py` | a real Stage 1 checkpoint |
| `test_stage2_units.py` | separator + losses over 40 training steps (bf16 included), datasets, evaluation loaders and protocol, swap / interpolation images, PCA / UMAP figures, config parsing, real referee checkpoints |
| `test_stage2_pipeline.py` | referee classifier training (5 attributes), separator training with the 5 paper configs, `--test_epoch` swap generation |
| `test_configs.py` | released configs vs the original ones (all values except paths) |
| `test_scripts.py` | data preparation scripts |

Pipeline steps run in fresh processes through `runners.py`, which applies the same adjustments to both versions:
- CPU only, and `torch.compile` with the eager backend;
- data loaders in the main process, and contiguous VAE outputs;
- stand-ins for the pretrained image metrics (LPIPS, DISTS, Inception), to avoid large downloads. The aggregation
  and FID / KID formulas around them still run.
