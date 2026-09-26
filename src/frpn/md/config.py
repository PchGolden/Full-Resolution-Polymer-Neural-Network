"""Command-line settings for this model family."""
import argparse
from frpn.training.arguments import (
    add_ablation_arguments, add_optimizer_arguments, add_system_arguments, add_encoder_arguments,
)
from .targets import MD_LOG10_LABELS_DEFAULT, MD_D_CLAMP_MIN


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DDP Trainer")

    # ---- Ablation Studys -------------------------
    add_ablation_arguments(parser)
    parser.add_argument(
        "--finetune_optimizer_stage1_only",
        action="store_true",
        help="(finetune only) Optimize only stage-1 modules + reg_head, excluding chain-level modules.",
    )
    parser.add_argument(
        "--stage1_capacity_mlp_hidden",
        type=int,
        default=0,
        help="(finetune only) Add a stage-1-only capacity MLP with hidden dim H (uses CLS rep).",
    )

    # ---- Dataset / Path -----------------------------------
    parser.add_argument("--dataset_name",help="Name of the dataset")
    parser.add_argument("--non_kfold", default=False, action="store_true", help="Use separate train/test pkl instead of CV folds")
    parser.add_argument("--pkl_path", help="Path to the *.pkl file")
    parser.add_argument("--test_pkl_path", help="Path to the *_test.pkl file")
    parser.add_argument(
        "--locked_test_pkl_path",
        default=None,
        help="Optional locked outer-test pkl for reporting test metrics based on best val checkpoint.",
    )
    parser.add_argument("--weight_path", default="checkpoints/no_pretrain.pt")
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Skip loading monomer-level pretrained weights even if --weight_path is set.",
    )
    parser.add_argument(
        "--pretrained_stage1_only",
        action="store_true",
        help="When loading --weight_path, restrict loading to stage-1 encoder modules only.",
    )
    parser.add_argument("--pretrain_train_path", default="datasets/pretraining/pretrain_train.lmdb")
    parser.add_argument("--pretrain_val_path",default="datasets/pretraining/pretrain_val.lmdb")
    parser.add_argument("--fold", type=int, help="Cross-validation fold index")
    parser.add_argument("--results_root", default="reproduce/outputs/training", help="Root directory to save results")

    # ---- Training Hyperparameters -------------------------
    add_optimizer_arguments(parser)
    parser.add_argument(
        "--early_stop_start_epoch",
        type=int,
        default=1,
        help="Start counting finetune/chain early-stop patience only from this epoch.",
    )
    parser.add_argument("--val_every_steps", type=int, default=50)
    parser.add_argument("--min_delta_abs", type=float, default=1e-3)
    parser.add_argument("--min_delta_rel", type=float, default=1e-2)

    # ---- Chain-level model args ----
    parser.add_argument("--num_chain_glob_feat", type=int, default=3)   # T, Mn, coexistence
    parser.add_argument("--num_chain_block_feat", type=int, default=1)  # volume fraction
    parser.add_argument("--num_seg_smiles_types", type=int, default=20000)
    parser.add_argument(
        "--chain_only_repr",
        choices=["smiles_embed", "fp_morgan2048"],
        default="smiles_embed",
        help="(chain_only) segment representation source",
    )
    parser.add_argument("--fp_radius", type=int, default=2, help="(chain_only fp) Morgan radius (radius=2 -> ECFP4)")
    parser.add_argument("--fp_nbits", type=int, default=2048, help="(chain_only fp) fingerprint bits")
    parser.add_argument(
        "--fp_replace_star_fallback",
        action="store_true",
        default=True,
        help="(chain_only fp) if RDKit fails, try smiles.replace('*','C') once",
    )

    parser.add_argument("--chain_pair_dim", type=int, default=32)
    parser.add_argument("--max_chain_dist", type=int, default=10000)

    parser.add_argument("--chain_encoder_layers", type=int, default=4)
    parser.add_argument("--chain_attention_heads", type=int, default=12)
    parser.add_argument("--chain_pair_hidden_dim", type=int, default=64)
    parser.add_argument("--chain_ffn_embed_dim", type=int, default=3072)
    parser.add_argument(
        "--disable_chain_symmetry_break",
        action="store_true",
        help="Disable stage-2 topology-anchor symmetry-breaking positional rewrite on chain node tokens.",
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
    parser.add_argument("--main_task", choices=["pretrain", "finetune", "chain", "chain_only"], default="finetune")
    parser.add_argument("--task_type", choices=["reg", "cls"], default="reg")
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument(
        "--label_index",
        type=int,
        default=-1,
        help="For regression with multi-label pkl, select one label index (0-based). -1 keeps all labels.",
    )
    parser.add_argument(
        "--log10_labels",
        type=str,
        default=",".join(sorted(MD_LOG10_LABELS_DEFAULT)),
        help="(reg) Comma-separated label names to apply log10 transform (before per-fold z-score).",
    )
    parser.add_argument(
        "--d_clamp_min",
        type=float,
        default=MD_D_CLAMP_MIN,
        help="(reg) Clamp minimum applied before log10.",
    )
    add_encoder_arguments(parser)

    return parser.parse_args(argv)
