# Diff-CA: Separating Common and Salient Factors with Diffusion Models

Code for the NeurIPS 2026 paper ([project page](https://msoumm.github.io/Diff-CA/), [arXiv](https://arxiv.org/abs/2606.06120)).

Diff-CA works in two stages:

1. **Conditioning (Stage 1).** A prompt-free latent diffusion model (flow matching, U-Net) is trained to reconstruct
   images from conditioning tokens Z (32 tokens of dimension 128). Z comes from a Cross-Query module over frozen
   DINOv3 features, plus a color token computed from the VAE latents (CQC).
2. **Separation (Stage 2).** A small Transformer splits Z into a salient code (the CLS output, pinned to zero on the
   background dataset) and common tokens, with a decoder that adds the two. It is trained with reconstruction,
   pinning, an adversary behind a self-tuning gradient reversal layer, and cycle consistency. Decoding the common
   tokens of one image with the salient code of another swaps the salient attribute.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The paper used Python 3.12 and PyTorch 2.7.1 with CUDA. Stage 2 runs as a single process (one GPU).

**Pretrained weights.**
- DINOv3 ViT-B/16, `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated on the Hugging Face Hub: accept the license, then
  `huggingface-cli login`).
- FFHQ: the VQ autoencoder of `CompVis/ldm-celebahq-256`, expected in `pretrained/ldm-celebahq-256/vqvae`:
  `huggingface-cli download CompVis/ldm-celebahq-256 --include "vqvae/*" --local-dir pretrained/ldm-celebahq-256`
- AFHQ and BraTS: `stabilityai/sd-vae-ft-mse` (downloaded automatically).
- CLIP baseline: `openai/clip-vit-base-patch16`.

## Data

The configs expect this layout (paths are relative to the repository root; edit the configs to change them):

```
data/
  ffhq256/{train,test}/               FFHQ at 256x256: scripts/prepare_ffhq.py (first 60k train, last 10k test)
  ffhq-features-dataset/json/         FFHQ-Features-Dataset attributes (glasses, smile, head pose), one JSON per image
  celebahq/                           CelebA-HQ images (test set of the reconstruction metrics)
  afhq/{train,val}/                   AFHQ (file names contain cat / dog / wild)
  afhq/labels.csv                     cat-breed scores, one line per image: "<file>, tabby cat 0.71, Egyptian cat 0.12, ..."
  brats_cropped/images/{train,val}/   BraTS 2023 slices + labels.txt: scripts/prepare_brats.py
```

## Pipeline

All entry points take a config file; `configs/` holds the configs of the paper experiments.

**1. Pre-compute VAE latents and backbone tokens.** This also writes `ae_config.json`, which holds the latent
scaling factor that every Stage 1 config reads.

```bash
python -m diffca.stage1.preprocess --config configs/preprocess/ffhq.yaml
python -m diffca.stage1.preprocess --config configs/preprocess/celebahq.yaml   # for the reconstruction metrics
```

**2. Train Stage 1.** Checkpoints go to `runs/stage1/ffhq_cqc/checkpoint-<step>/` (with the EMA weights).

```bash
python -m diffca.stage1.train --config configs/stage1/ffhq_cqc.yaml
python -m diffca.stage1.evaluate_checkpoint --config configs/stage1/ffhq_cqc.yaml \
    --checkpoint runs/stage1/ffhq_cqc/checkpoint-300000 --eval_samples 1000     # PSNR, SSIM, LPIPS, DISTS, FID, ...
```

**3. Extract the tokens Z of every image** (one `.pt` file per image), the input of Stage 2.

```bash
python -m diffca.stage1.extract_features --config configs/stage1/ffhq_cqc.yaml \
    --checkpoint runs/stage1/ffhq_cqc/checkpoint-300000 \
    --output_dir runs/stage1/ffhq_cqc/extracted_features/step-300000
```

**4. Train the referee classifier** that scores swaps in token space during separator training.

```bash
F=runs/stage1/ffhq_cqc/extracted_features/step-300000
python -m diffca.stage2.train_classifier --attribute glasses --pt_folder_train $F/train --pt_folder_test $F/test \
    --json_folder data/ffhq-features-dataset/json --load_features_in_ram --output_dir runs/classifiers
```

AFHQ needs `--sublabels_csv data/afhq/labels.csv`; BraTS needs `--labels_train` and `--labels_test`.

**5. Train the separator.** Checkpoints go to `runs/stage2/glasses/disentangler_checkpoints/epoch_<n>/`. Each
epoch prints and logs (wandb, offline by default) the reconstruction and swap accuracies.

```bash
python -m diffca.stage2.train --config_yaml configs/stage2/glasses.yaml
```

**6. Swaps and interpolations** with a trained separator, decoded by the Stage 1 model named in the config.

```bash
python -m diffca.stage2.train --config_yaml configs/stage2/glasses.yaml \
    --resume_from_checkpoint runs/stage2/glasses/disentangler_checkpoints/epoch_999 \
    --test_epoch --generate_all_images --interpolate
```

This writes `runs/stage2/glasses/final_evaluation/swapped_images/<i>_<j>/` for every pair of classes i, j:
`swap_x2y/x_y.png` holds the common factor of x with the salient factor of y. The other folders are `swap_y2x/`,
and `recon_cs_x/` and `recon_cs_y/` for the reconstructions. The image pairs are listed in
`final_evaluation/<i>_<j>.txt`, and the interpolation grids are in `runs/stage2/glasses/<epoch>/interpolations/`.
Add `--pca --umap` for the projections of the salient codes.

## Configs

| Experiment | Stage 1 | Stage 2 |
|---|---|---|
| FFHQ glasses / smile / head pose | `stage1/ffhq_cqc.yaml` | `stage2/glasses.yaml`, `smile.yaml`, `headpose.yaml` |
| AFHQ cats vs dogs | `stage1/afhq_cqc.yaml` | `stage2/afhq.yaml` |
| BraTS tumor vs healthy slices | `stage1/brats_cqc.yaml` | `stage2/brats.yaml` |
| Conditioning baselines (FFHQ) | `stage1/ffhq_cq.yaml` (no color token), `ffhq_dinov3.yaml`, `ffhq_clip.yaml`, `ffhq_encdiff.yaml` | |

## Tests

`tests/` checks that this code computes exactly what the original research code computed (see `tests/README.md`).

## License

The code is released under the MIT License (see `LICENSE`). The datasets (FFHQ, CelebA-HQ, AFHQ, BraTS) and the
pretrained weights it uses (DINOv3, the VQ and KL autoencoders, CLIP) keep their own licenses and terms of use.

## Citation

```bibtex
@inproceedings{soumm2026diffca,
  title={Diff-CA: Separating Common and Salient Factors with Diffusion Models},
  author={Soumm, Micha{\"e}l and Fournier-Montgieux, Alexandre and He, Yunlong and Gori, Pietro and Newson, Alasdair},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
