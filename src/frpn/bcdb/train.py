"""Training command: configure the run, select a task, and record its results."""
import os
import time
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from .config import parse_args
from .models.model import MultiMolModel
from .data import build_dataloader
from .engine import train_one_epoch, evaluate, evaluate_pretrain
from frpn.training.runtime import configure_runtime, set_seed, init_distributed, is_main_process
from frpn.training.checkpoints import save_json, save_checkpoint_finetune, load_monomer_weights
from frpn.training.pretraining import run_pretraining
from .targets import (
    _parse_comma_list, _label_transform_torch, _label_inverse_transform_torch, _compute_label_norm_from_train_dataset,
)
from .engine import maybe_mask_temperature_feature

METRIC_PRINT_DIGITS = 8


def load_monomer_pretrained_weights(model, weight_path, args):
    return load_monomer_weights(model, weight_path, args, include_pretrain_heads=True)


def freeze_for_chain_stage1(model):
    """
    Stage-1 chain training:
    - Freeze monomer-level encoder
    - Train chain-level encoder + reg_head
    """
    m = model.module if hasattr(model, "module") else model

    # ---- freeze everything first ----
    for p in m.parameters():
        p.requires_grad = False

    # ---- unfreeze chain-level modules ----
    for p in m.chain_token_feature.parameters():
        p.requires_grad = True
    for p in m.chain_edge_feature.parameters():
        p.requires_grad = True
    for p in m.chain_encoder.parameters():
        p.requires_grad = True
    for p in m.reg_head.parameters():
        p.requires_grad = True

    # ---- eval frozen parts ----
    m.embed_tokens.eval()
    m.atom_feature.eval()
    m.edge_feature.eval()
    m.encoder.eval()
    m.se3_invariant_kernel.eval()

    # ---- train chain parts ----
    m.chain_token_feature.train()
    m.chain_edge_feature.train()
    m.chain_encoder.train()
    m.reg_head.train()


def apply_pkl_metadata_to_args(args):
    """Best-effort: align model hyperparameters to dataset metadata.

    This is especially important for Stage-2-only SMILES-type embeddings:
    if num_seg_smiles_types is left at a large default, it will distort the
    Stage2big active-parameter equalization baseline.
    """
    if getattr(args, "pkl_path", None) is None:
        return

    try:
        with open(args.pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return

    smiles_vocab_size = data.get("smiles_vocab_size", None)
    if smiles_vocab_size is not None:
        smiles_vocab_size = int(smiles_vocab_size)
        # If user kept default, override; otherwise ensure it's large enough.
        if getattr(args, "num_seg_smiles_types", 20000) == 20000:
            args.num_seg_smiles_types = smiles_vocab_size
        else:
            args.num_seg_smiles_types = max(int(args.num_seg_smiles_types), smiles_vocab_size)

    max_segments = data.get("max_segments", None)
    if max_segments is not None:
        max_segments = int(max_segments)
        if getattr(args, "max_segments", 10) == 10:
            args.max_segments = max_segments
        else:
            args.max_segments = max(int(args.max_segments), max_segments)

    label_names = data.get("label_names", None)
    if label_names is not None:
        args.label_names = list(label_names)

    if getattr(args, "rank", 0) == 0:
        msg = (
            f"[META] pkl={args.pkl_path} smiles_vocab_size={smiles_vocab_size} "
            f"-> num_seg_smiles_types={getattr(args, 'num_seg_smiles_types', None)}; "
            f"max_segments(meta)={max_segments} -> max_segments={getattr(args, 'max_segments', None)}"
        )
        print(msg)


def build_optimizer(model, args):
    """Select parameters in the original model-specific order."""
    m = model.module if isinstance(model, DDP) else model

    if args.main_task == "chain" and args.freeze_encoder:
        optimizer = torch.optim.AdamW(
            [
                *m.chain_token_feature.parameters(),
                *m.chain_edge_feature.parameters(),
                *m.chain_encoder.parameters(),
                *m.reg_head.parameters(),
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    return optimizer


def run_supervised(model, args):
    """Train and select checkpoints using this workflow's evaluation protocol."""
    best_epoch = 0
    if args.weight_path is not None:
        try:
            model = load_monomer_pretrained_weights(model, args.weight_path, args)
        except FileNotFoundError:
            print("Checkpoint file not found, training from scratch.")
    if args.freeze_encoder:
        freeze_for_chain_stage1(model)
    model = model.cuda()
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,)
    pkl_tag = Path(args.pkl_path).stem if args.pkl_path else "nopkl"
    wt_tag = Path(args.weight_path).stem if args.weight_path else "nowt"
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(args.results_root)
        / args.dataset_name
        / f"task_{args.main_task}_eq_{int(args.equalize_active_params)}_freeze_{int(args.freeze_encoder)}"
        / f"data_{pkl_tag}"
        / f"wt_{wt_tag}"
    )

    if args.main_task in {"chain", "stage2_only"}:
        run_dir = run_dir / (
            f"repr_{args.stage2_only_repr}_noanchor_{int(args.disable_anchor_symmetry_break)}"
        )

    run_dir = run_dir / f"fold_{args.fold}" / (
        f"wd_{args.weight_decay}_lr_{args.lr}_do_{args.dropout}_wogroup_{args.wo_pair}"
    )
    run_dir = run_dir / f"run_{run_stamp}"
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    pred_dir = run_dir / "predictions"
    for d in (ckpt_dir, log_dir, pred_dir):
        if is_main_process(args.rank):
            d.mkdir(parents=True, exist_ok=True)

    # ============ Data Loaders ============
    if not args.non_kfold:
        train_loader, train_sampler = build_dataloader(args.pkl_path, args, mode="train")
        val_loader, _ = build_dataloader(args.pkl_path, args, mode="val")
    else:
        train_loader, train_sampler = build_dataloader(args.pkl_path, args, mode="train")
        val_loader, _ = build_dataloader(args.test_pkl_path, args, mode="full")

    # ============ Label normalization (regression only) ============
    label_norm = None
    label_names = list(getattr(args, "label_names", [])) if getattr(args, "label_zscore", False) else None
    if args.task_type == "reg" and getattr(args, "label_zscore", False):
        if not label_names:
            try:
                with open(args.pkl_path, "rb") as f:
                    header = pickle.load(f)
                label_names = list(header.get("label_names", []))
            except Exception:
                label_names = []

        if not label_names:
            raise ValueError("--label_zscore requires label_names in pkl header (missing label_names).")

        requested_log10 = _parse_comma_list(getattr(args, "log10_labels", ""))
        log10_labels = [k for k in requested_log10 if k in set(label_names)]
        unknown = [k for k in requested_log10 if k not in set(label_names)]
        if unknown and is_main_process(args.rank):
            print(f"[WARN] log10_labels not in label_names and will be ignored: {unknown}")

        label_norm = _compute_label_norm_from_train_dataset(
            train_loader.dataset,
            label_names=label_names,
            log10_labels=log10_labels,
            clamp_min=float(getattr(args, "log10_clamp_min", 1e-16)),
        )
        if is_main_process(args.rank):
            save_json(label_norm, run_dir / "label_norm.json")

    # ============ Optimizer & Scheduler ============
    optimizer = build_optimizer(model, args)
    scaler = GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=args.lr * 0.01
    )
    # ============ Training Loop ============
    metrics_history = {}
    best_score = float("inf") if args.task_type == "reg" else 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            args,
            train_sampler,
            label_norm=label_norm,
            label_names=label_names,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, args, label_norm=label_norm, label_names=label_names)
        epoch_time = time.time() - epoch_start
        peak_mem_gb = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        if is_main_process(args.rank):
            # ----- Logging & Printing -----
            if args.task_type == "reg":
                peak_mem_suffix = (
                    f" peak_mem_gb={peak_mem_gb:.3f}"
                    if peak_mem_gb is not None
                    else ""
                )
                print(
                    f"[E{epoch:03d}] "
                    f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                    f"val_mae={val_metrics['MAE']:.{METRIC_PRINT_DIGITS}f} "
                    f"val_rmse={val_metrics['RMSE']:.{METRIC_PRINT_DIGITS}f} "
                    f"val_r2={val_metrics['R2']:.{METRIC_PRINT_DIGITS}f} "
                    f"val_rmse_raw={val_metrics.get('RMSE_raw', val_metrics['RMSE']):.{METRIC_PRINT_DIGITS}f} "
                    f"val_r2_raw={val_metrics.get('R2_raw', val_metrics['R2']):.{METRIC_PRINT_DIGITS}f} "
                    f"time={epoch_time:.1f}s"
                    f"{peak_mem_suffix}"
                )
            else:
                peak_mem_suffix = (
                    f" peak_mem_gb={peak_mem_gb:.3f}"
                    if peak_mem_gb is not None
                    else ""
                )
                print(
                    f"[E{epoch:03d}] "
                    f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                    f"val_acc={val_metrics['ACC']:.{METRIC_PRINT_DIGITS}f} "
                    f"time={epoch_time:.1f}s"
                    f"{peak_mem_suffix}"
                )

            # ----- Save Metrics -----
            metrics_history[epoch] = {
                "train_loss": train_loss,
                **val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "peak_mem_gb": peak_mem_gb,
            }
            save_json(metrics_history, run_dir / "metrics.json")

            # ----- Save Residuals or Confusion Matrix -----
            if args.task_type == "reg":
                np.save(pred_dir / f"residuals_ep{epoch}.npy", val_metrics["residuals"])
            else:
                np.save(pred_dir / f"confusion_ep{epoch}.npy", val_metrics["confusion"])
                np.save(pred_dir / f"preds_ep{epoch}.npy", val_metrics["preds"])
                np.save(pred_dir / f"labels_ep{epoch}.npy", val_metrics["labels"])

            # ----- Save Best Checkpoint -----
            is_better =  (
                (args.task_type == "reg" and val_metrics["RMSE"] < best_score)
                or (args.task_type == "cls" and val_metrics["ACC"] > best_score)
            )

            if is_better:
                best_epoch = epoch
                best_score = (
                    val_metrics["RMSE"] if args.task_type == "reg" else val_metrics["ACC"]
                )
                if args.main_task != "finetune" or args.save_finetune_checkpoint:
                    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    save_checkpoint_finetune(
                        {
                            "epoch": epoch,
                            "model_state": model_state,
                            "optimizer_state": optimizer.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "args": vars(args),
                            "label_norm": label_norm,
                        },
                        ckpt_dir,
                    )
        if args.distributed:
            be_t = torch.tensor(best_epoch, device=args.local_rank)
            dist.broadcast(be_t, src=0)
            best_epoch = be_t.item()

        # ----- Target-RMSE early stopping -----
        target_stop = False
        if args.task_type == "reg" and args.stop_on_target_rmse:
            cur_rmse = float(val_metrics["RMSE"])
            if epoch <= args.target_rmse_max_epoch and cur_rmse < args.target_rmse:
                target_stop = True

        # Make sure all ranks stop together
        if args.distributed and dist.is_initialized():
            if is_main_process(args.rank):
                flag_tensor = torch.tensor(int(target_stop), device="cuda")
            else:
                flag_tensor = torch.zeros(1, device="cuda")
            dist.broadcast(flag_tensor, src=0)
            target_stop = bool(flag_tensor.item())

        if target_stop:
            if is_main_process(args.rank):
                print(
                    f"Target-RMSE early stop triggered at epoch {epoch}: "
                    f"val_rmse={val_metrics['RMSE']:.4f} < {args.target_rmse} "
                    f"(max_epoch={args.target_rmse_max_epoch})"
                )
            break

        if args.early_stop_patience and (epoch - best_epoch >= args.early_stop_patience):
            if is_main_process(args.rank):
                print(f"Early stopping triggered at epoch {epoch}, best was epoch {best_epoch}, best score was {best_score}")
            break


def prepare_pretraining(model, args):
    # -------- Dataloader & model --------
    model = model.cuda()
    train_path = args.pretrain_train_path
    val_path   = args.pretrain_val_path
    if args.pretrain_use_split:
        pre_loader, pre_sampler = build_dataloader(train_path, args, mode="train")
        val_loader, _ = build_dataloader(
            train_path if os.path.abspath(train_path) == os.path.abspath(val_path) else val_path,
            args,
            mode="val",
        )
    else:
        pre_loader, pre_sampler = build_dataloader(train_path, args, mode="full")
        val_loader, _ = build_dataloader(val_path, args, mode="full")

    # -------- Directory setup --------
    pretrain_data_tag = Path(train_path).stem if train_path else "nopkl"
    split_tag = (
        f"split_fold{int(args.pretrain_split_fold)}"
        if args.pretrain_use_split
        else "split_full"
    )
    run_dir  = (
        Path(args.results_root)
        / "random_pretrain"
        / f"data_{pretrain_data_tag}"
        / split_tag
        / f"lr_{args.lr}_wd_{args.weight_decay}_bs_{args.batch_size}"
    )
    ckpt_dir = run_dir / "checkpoints"
    log_dir  = run_dir / "logs"
    if is_main_process(args.rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True,  exist_ok=True)

    run_pretraining(model, args, pre_loader, pre_sampler, val_loader, run_dir, maybe_mask_temperature_feature)


def main(argv=None):
    args = parse_args(argv)
    configure_runtime()
    args.distributed = args.world_size > 1 or args.distributed
    set_seed(args.seed)
    init_distributed(args)
    apply_pkl_metadata_to_args(args)
    try:
        model = MultiMolModel(args)
        if args.main_task == "pretrain":
            prepare_pretraining(model, args)
        else:
            run_supervised(model, args)
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
