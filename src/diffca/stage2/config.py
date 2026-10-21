"""Arguments of the separator training script. A YAML file (`--config_yaml`) sets defaults; command-line flags win.

YAML keys that are not arguments are ignored, as before. This lets old configs load: their keys for removed
options (beta, tau, push_target, lambda_independence, ...) never had an effect on training. The two removed
options that did (REMOVED_OPTIONS) must be left off.
"""

import argparse
import sys

from diffca.utils import load_config

ATTRIBUTES = ["glasses", "smile", "headPose", "afhq", "brats"]
FFHQ_ATTRIBUTES = {"glasses", "smile", "headPose"}

# Keys of removed options -> value that keeps them off. Any other value is rejected.
REMOVED_OPTIONS = {
    "lambda_push": 0,
    "contrastive": False,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Stage 2: train the salient/common separator.")

    # --- Data ---
    parser.add_argument("--config_yaml", type=str, default=None, help="YAML file with default values for all arguments.")
    parser.add_argument("--train_pt_folder", type=str, default=None, help="Folder of training .pt features.")
    parser.add_argument("--test_pt_folder", type=str, default=None, help="Folder of validation .pt features.")
    parser.add_argument("--attribute", type=str, default="glasses", help=f"One of {ATTRIBUTES}.")
    parser.add_argument("--json_folder", type=str, default=None, help="FFHQ attribute .json files (glasses, smile, headPose).")
    parser.add_argument("--sublabels_csv", type=str, default=None, help="AFHQ breed scores CSV (train and test).")
    parser.add_argument("--labels_train", type=str, default=None, help="BraTS training labels .txt.")
    parser.add_argument("--labels_test", type=str, default=None, help="BraTS validation labels .txt.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Checkpoints and generated images.")
    parser.add_argument("--referee_classifier_path", type=str, default="classifier_checkpoints/glasses_classifier_color_aug.pt",
                        help="Referee classifier (see stage2.train_classifier). Empty: no swap accuracy.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Separator checkpoint folder (.../epoch_<n>).")
    parser.add_argument("--test_epoch", action="store_true", help="Evaluate the checkpoint of --resume_from_checkpoint only.")

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Separator learning rate.")
    parser.add_argument("--adversary_lr", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- Model ---
    parser.add_argument("--model_type", type=str, default="token_disentangler", choices=["token_disentangler"])
    parser.add_argument("--num_encoder_layers", type=int, default=5)
    parser.add_argument("--num_decoder_layers", type=int, default=0, choices=[0], help="Must be 0 (additive decoder).")
    parser.add_argument("--low_rank", type=int, default=64, help="Rank of the salient-code projection.")

    # --- Losses ---
    parser.add_argument("--lambda_recon", type=float, default=10.0, help="Reconstruction loss weight.")
    parser.add_argument("--lambda_norm_pos", type=float, default=0.1, help="exp(-||h||^2) weight on Y=1.")
    parser.add_argument("--lambda_norm_neg", type=float, default=0.1, help="Pinning ||h||^2 weight on Y=0.")
    parser.add_argument("--lambda_cycle_context", type=float, default=0, help="Cycle loss weight, common tokens.")
    parser.add_argument("--lambda_cycle_cls", type=float, default=0, help="Cycle loss weight, salient code.")
    parser.add_argument("--lambda_adv", type=float, default=0, help="Adversarial loss weight.")
    parser.add_argument("--low_target_acc", type=float, default=0.85, help="Adversary target accuracy (lower bound).")
    parser.add_argument("--high_target_acc", type=float, default=0.85, help="Adversary target accuracy (upper bound).")

    # --- Stage 1 model, for image generation ---
    parser.add_argument("--config_file", type=str, help="Stage 1 config file.")
    parser.add_argument("--save_dir", type=str, help="Stage 1 output folder (with checkpoint-<step>/ folders).")
    parser.add_argument("--global_step", type=int, help="Stage 1 checkpoint step (default: the latest).")

    # --- Evaluation and visualization ---
    parser.add_argument("--umap", action="store_true", help="Save UMAP projections of the codes every epoch.")
    parser.add_argument("--pca", action="store_true", help="Save PCA projections of the codes every epoch.")
    parser.add_argument("--no_generate_images", action="store_true", help="Do not load the Stage 1 model while training.")
    parser.add_argument("--generate_all_images", action="store_true", help="With --test_epoch: decode all evaluation swaps.")
    parser.add_argument("--interpolate", action="store_true", help="With --test_epoch: decode salient-code interpolations.")

    # --- Logging ---
    parser.add_argument("--wandb_project", type=str, default="attribute-disentanglement")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_id", type=str, default=None)

    # --- EMA ---
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay of the evaluation weights.")
    parser.add_argument("--cycle_ema_decay", type=float, default=0.99, help="EMA decay of the cycle encoder.")
    return parser


def _provided_flags(raw_args):
    return {arg[2:].split("=", 1)[0] for arg in raw_args if arg.startswith("--") and len(arg) > 2}


def apply_yaml_config(args, raw_args, yaml_path):
    """Fill `args` from the YAML file, except the arguments given on the command line."""
    config_data = load_config(yaml_path)
    if config_data is None:
        raise ValueError(f"Could not load config YAML: {yaml_path}")
    provided = _provided_flags(raw_args)
    for key, value in config_data.items():
        if key in REMOVED_OPTIONS:
            if value != REMOVED_OPTIONS[key]:
                raise ValueError(f"'{key}: {value}' configures an option that was removed from this code.")
            continue
        if not hasattr(args, key) or key in provided:
            continue
        setattr(args, key, value)
    return args


def validate(args, parser):
    if args.attribute and args.attribute.lower() == "headpose":
        args.attribute = "headPose"
    if args.attribute not in ATTRIBUTES:
        parser.error(f"--attribute must be one of {ATTRIBUTES}.")
    if args.model_type != "token_disentangler" or args.num_decoder_layers != 0:
        parser.error("Only model_type=token_disentangler with num_decoder_layers=0 is supported.")
    if not args.train_pt_folder or not args.test_pt_folder:
        parser.error("--train_pt_folder and --test_pt_folder are required.")
    if args.attribute in FFHQ_ATTRIBUTES and not args.json_folder:
        parser.error("--json_folder is required for FFHQ attributes.")
    if args.attribute == "afhq" and not args.sublabels_csv:
        parser.error("--sublabels_csv is required for AFHQ.")
    if args.attribute == "brats" and not (args.labels_train and args.labels_test):
        parser.error("--labels_train and --labels_test are required for BraTS.")
    if args.test_epoch and not args.resume_from_checkpoint:
        parser.error("--test_epoch requires --resume_from_checkpoint.")
    if args.generate_all_images and not args.test_epoch:
        parser.error("--generate_all_images requires --test_epoch.")
    if args.interpolate and not args.test_epoch:
        parser.error("--interpolate requires --test_epoch.")


def get_config(args_list=None):
    parser = build_parser()
    args = parser.parse_args(args_list)
    if args.config_yaml:
        raw_args = sys.argv[1:] if args_list is None else list(args_list)
        apply_yaml_config(args, raw_args, args.config_yaml)
    validate(args, parser)
    return args
