"""Command-line settings for this model family."""
import argparse
from frpn.training.arguments import (
    add_ablation_arguments, add_optimizer_arguments, add_system_arguments, add_encoder_arguments,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DDP Trainer")

    # ---- Ablation Studys -------------------------
    add_ablation_arguments(parser)
    parser.add_argument("--equalize_active_params", action="store_true")

    # ---- Dataset / Path -----------------------------------
    parser.add_argument("--dataset_name",help="Name of the dataset")
    parser.add_argument("--non_kfold", default=False, action="store_true", help="Use separate train/test pkl instead of CV folds")
    parser.add_argument("--pkl_path", help="Path to the *.pkl file")
    parser.add_argument("--test_pkl_path", help="Path to the *_test.pkl file")
    parser.add_argument("--weight_path", default=None, help="Optional initialization checkpoint; omit for training from scratch.")
    parser.add_argument("--pretrain_train_path", default="datasets/pretraining/pretrain_train.lmdb")
    parser.add_argument("--pretrain_val_path",default="datasets/pretraining/pretrain_val.lmdb")
    parser.add_argument("--fold", type=int, help="Cross-validation fold index")
    parser.add_argument("--results_root", default="reproduce/outputs/training", help="Root directory to save results")
    parser.add_argument(
        "--save_finetune_checkpoint",
        action="store_true",
        help="If set, save best checkpoint.pt for finetune runs as well.",
    )

    # ---- Training Hyperparameters -------------------------
    add_optimizer_arguments(parser)
    parser.add_argument("--val_every_steps", type=int, default=50)
    parser.add_argument("--min_delta_abs", type=float, default=1e-3)
    parser.add_argument("--min_delta_rel", type=float, default=1e-2)
    parser.add_argument(
        "--label_zscore",
        action="store_true",
        help=(
            "Regression only: apply per-fold target transform + z-score using train split stats, "
            "train in normalized space, and report both z/raw metrics."
        ),
    )
    parser.add_argument(
        "--log10_labels",
        default="",
        help="Comma-separated label names to apply log10 transform before z-score (e.g. 'D,self-diffusion').",
    )
    parser.add_argument(
        "--log10_clamp_min",
        type=float,
        default=1e-16,
        help="Clamp minimum for log10 labels (values <=0 will be clamped).",
    )
    parser.add_argument(
        "--pretrain_use_split",
        action="store_true",
        help="Use fold split (train/val) for pretrain data instead of full/full loaders.",
    )
    parser.add_argument(
        "--pretrain_split_fold",
        type=int,
        default=0,
        help="Fold index used when --pretrain_use_split is enabled.",
    )
    parser.add_argument(
        "--mask_temperature_feature",
        action="store_true",
        help="Mask temperature feature (glob_feat0 / chain_glob_feat[0]) to avoid non-chemical/topology noise.",
    )

    # ---- Chain-level model args ----
    parser.add_argument("--num_chain_glob_feat", type=int, default=3)   # T, Mn, coexistence
    parser.add_argument("--num_chain_block_feat", type=int, default=1)  # volume fraction

    parser.add_argument("--chain_pair_dim", type=int, default=32)
    parser.add_argument("--max_chain_dist", type=int, default=10000)

    parser.add_argument("--chain_encoder_layers", type=int, default=4)
    parser.add_argument("--chain_attention_heads", type=int, default=12)
    parser.add_argument("--chain_pair_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_ffn_embed_dim", type=int, default=3072)
    parser.add_argument(
        "--max_chain_tokens", type=int, default=1536,
        help="Repeat-token budget; longer chains are proportionally rescaled before expansion (<=0 disables).",
    )
    parser.add_argument("--max_segments", type=int, default=10)
    parser.add_argument("--num_block_types", type=int, default=8)
    parser.add_argument("--num_seg_smiles_types", type=int, default=20000)
    parser.add_argument(
        "--stage2_only_repr",
        choices=["full", "smiles_only", "fp_morgan2048"],
        default="full",
        help=(
            "Stage2-only segment initialization: "
            "full=dop+block+smiles+pos, "
            "smiles_only=SMILES embedding only, "
            "fp_morgan2048=fixed Morgan/ECFP4(2048) fingerprints + small projection."
        ),
    )

    # ---- Stage2-only fingerprint (Morgan/ECFP) settings ----
    parser.add_argument(
        "--fp_csv",
        default="datasets/raw/bcdb/bcdb.csv",
        help="CSV used to rebuild dataset-level SMILES vocab and Morgan fingerprint table (Stage2-only fp_morgan2048).",
    )
    parser.add_argument(
        "--fp_smiles_prefix",
        default="SMILES",
        help="SMILES column prefix in fp_csv (expects columns like SMILES0..SMILES9).",
    )
    parser.add_argument(
        "--fp_radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (radius=2 corresponds to ECFP4).",
    )
    parser.add_argument(
        "--fp_nbits",
        type=int,
        default=2048,
        help="Morgan fingerprint bit-size.",
    )
    parser.add_argument(
        "--fp_replace_star_fallback",
        dest="fp_replace_star_fallback",
        action="store_true",
        default=True,
        help="If RDKit fails to parse SMILES, retry with '*' replaced by 'C' (enabled by default).",
    )
    parser.add_argument(
        "--no_fp_replace_star_fallback",
        dest="fp_replace_star_fallback",
        action="store_false",
        help="Disable '*'->'C' fallback when building fingerprints.",
    )
    parser.add_argument(
        "--disable_anchor_symmetry_break",
        action="store_true",
        help="Disable anchor-based symmetry breaking field in chain token construction.",
    )

    # ---- Target RMSE Early Stop (for finetune/reg) -------
    parser.add_argument(
        "--stop_on_target_rmse",
        action="store_true",
        help="Stop early if val RMSE reaches target within a given epoch range (finetune/reg)."
    )
    parser.add_argument("--target_rmse", type=float, default=0.4)
    parser.add_argument("--target_rmse_max_epoch", type=int, default=100)

    # ---- System Settings ----------------------------------
    add_system_arguments(parser)

    # ---- Model / Task -------------------------------------
    parser.add_argument("--main_task", choices=["pretrain", "finetune", "chain", "stage2_only"], default="finetune")
    parser.add_argument("--task_type", choices=["reg", "cls"], default="reg")
    parser.add_argument("--num_tasks", type=int, default=1)
    add_encoder_arguments(parser)

    return parser.parse_args(argv)
