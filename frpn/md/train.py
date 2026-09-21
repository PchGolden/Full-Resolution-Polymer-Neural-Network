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
    _parse_log10_labels_arg, _resolve_label_names_from_dataset, _label_transform_torch, _label_inverse_transform_np, _compute_label_norm_from_train_dataset,
)
from .targets import MD_REG_LABEL_NAMES, MD_LOG10_LABELS_DEFAULT, MD_D_CLAMP_MIN

METRIC_PRINT_DIGITS = 8


def load_monomer_pretrained_weights(model, weight_path, args):
    return load_monomer_weights(model, weight_path, args, include_pretrain_heads=not getattr(args, "pretrained_stage1_only", False))


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
    if hasattr(m, "seg_only_projs"):
        for p in m.seg_only_projs.parameters():
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
    if hasattr(m, "seg_only_projs"):
        m.seg_only_projs.train()
    m.reg_head.train()


def freeze_pretrain_heads(model):
    """Freeze pretrain-only heads (unused in finetune/chain tasks)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "lm_head"):
        # NOTE: MaskLMHead stores a *tied* reference to embed_tokens.weight in `lm_head.weight`.
        # Freezing `lm_head.parameters()` would inadvertently freeze the embedding table.
        for p in m.lm_head.dense.parameters():
            p.requires_grad = False
        for p in m.lm_head.layer_norm.parameters():
            p.requires_grad = False
        if hasattr(m.lm_head, "bias") and isinstance(m.lm_head.bias, torch.nn.Parameter):
            m.lm_head.bias.requires_grad = False
        m.lm_head.eval()
    if hasattr(m, "movement_pred_head"):
        for p in m.movement_pred_head.parameters():
            p.requires_grad = False
        m.movement_pred_head.eval()


def build_optimizer(model, args):
    """Select parameters in the original model-specific order."""
    m = model.module if isinstance(model, DDP) else model

    if args.main_task == "finetune" and getattr(args, "finetune_optimizer_stage1_only", False):
        stage1_params = [
            *m.embed_tokens.parameters(),
            *m.atom_feature.parameters(),
            *m.edge_feature.parameters(),
            *m.encoder.parameters(),
            *m.se3_invariant_kernel.parameters(),
            *(m.stage1_capacity_mlp.parameters() if getattr(m, "stage1_capacity_mlp", None) is not None else []),
            *m.reg_head.parameters(),
        ]
        optimizer = torch.optim.AdamW(
            stage1_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.main_task in ("chain", "chain_only") and args.freeze_encoder:
        optimizer = torch.optim.AdamW(
            [
                *m.chain_token_feature.parameters(),
                *m.chain_edge_feature.parameters(),
                *m.chain_encoder.parameters(),
                *(m.seg_only_projs.parameters() if hasattr(m, "seg_only_projs") else []),
                *m.reg_head.parameters(),
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if is_main_process(args.rank):
        trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optim_numel = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        print(f"[ParamCount] trainable_numel={trainable_numel} optim_numel={optim_numel}")

    return optimizer


def run_supervised(model, args):
    """Train and select checkpoints using this workflow's evaluation protocol."""
    best_epoch = 0
    if (not getattr(args, "no_pretrained", False)) and args.weight_path is not None:
        try:
            model = load_monomer_pretrained_weights(model, args.weight_path, args)
        except FileNotFoundError:
            print("Checkpoint file not found, training from scratch.")

    # Pretrain-only heads are unused in finetune/chain tasks.
    freeze_pretrain_heads(model)

    # Finetune stage-1-only runs: freeze chain-level modules so trainable == optim params.
    if args.main_task == "finetune" and getattr(args, "finetune_optimizer_stage1_only", False):
        m = model.module if hasattr(model, "module") else model
        for module_name in ("chain_token_feature", "chain_edge_feature", "chain_encoder", "seg_only_projs"):
            if hasattr(m, module_name):
                mod = getattr(m, module_name)
                for p in mod.parameters():
                    p.requires_grad = False
                mod.eval()
    if args.freeze_encoder:
        freeze_for_chain_stage1(model)
    model = model.cuda()
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,)
    run_dir = (
        Path(args.results_root)
        / args.dataset_name
        / f"fold_{args.fold}"
        / f"wd_{args.weight_decay}_lr_{args.lr}_do_{args.dropout}_wogroup_{args.wo_pair}"
    )
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

    locked_test_loader = None
    if args.locked_test_pkl_path:
        locked_test_loader, _ = build_dataloader(args.locked_test_pkl_path, args, mode="full")

    label_names = None
    label_norm = None
    if args.task_type == "reg":
        label_names = _resolve_label_names_from_dataset(train_loader.dataset, args)
        args.label_names = label_names  # convenience for debugging
        log10_labels = _parse_log10_labels_arg(getattr(args, "log10_labels", ""))
        label_norm = _compute_label_norm_from_train_dataset(
            train_loader.dataset,
            label_names,
            log10_labels=log10_labels,
            d_clamp_min=float(getattr(args, "d_clamp_min", MD_D_CLAMP_MIN)),
        )
        if is_main_process(args.rank):
            save_json(label_norm, run_dir / "label_norm.json")

    # ============ Optimizer & Scheduler ============
    optimizer = build_optimizer(model, args)
    scaler = GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(10, args.epochs // 3),
        T_mult=1,
        eta_min=args.lr * 0.01,
    )
    # ============ Training Loop ============
    metrics_history = {}
    best_score = float("inf") if args.task_type == "reg" else 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_stats = train_one_epoch(
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
        train_loss = float(train_stats["loss"])
        peak_mem_gb = float(train_stats.get("peak_mem_gb", 0.0))
        scheduler.step()

        val_metrics = evaluate(model, val_loader, args, label_norm=label_norm, label_names=label_names)
        epoch_time = time.time() - epoch_start

        if is_main_process(args.rank):
            # ----- Logging & Printing -----
            if args.task_type == "reg":
                print(
                    f"[E{epoch:03d}] "
                    f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                    f"val_mae={val_metrics['MAE']:.{METRIC_PRINT_DIGITS}f} "
                    f"val_rmse={val_metrics['RMSE']:.{METRIC_PRINT_DIGITS}f} "
                    f"val_r2={val_metrics['R2']:.{METRIC_PRINT_DIGITS}f} "
                    f"peak_mem_gb={peak_mem_gb:.3f} "
                    f"time={epoch_time:.1f}s"
                )
            else:
                print(
                    f"[E{epoch:03d}] "
                    f"train_loss={train_loss:.{METRIC_PRINT_DIGITS}f} "
                    f"val_acc={val_metrics['ACC']:.{METRIC_PRINT_DIGITS}f} "
                    f"peak_mem_gb={peak_mem_gb:.3f} "
                    f"time={epoch_time:.1f}s"
                )

            # ----- Save Metrics -----
            metrics_history[epoch] = {
                "train_loss": train_loss,
                "peak_mem_gb": peak_mem_gb,
                **val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
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
                best_test_metrics = None
                if locked_test_loader is not None:
                    test_metrics = evaluate(
                        model,
                        locked_test_loader,
                        args,
                        label_norm=label_norm,
                        label_names=label_names,
                    )
                    best_test_metrics = {
                        "epoch": epoch,
                        **test_metrics,
                    }
                    save_json(best_test_metrics, run_dir / "best_test_metrics.json")

                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                save_checkpoint_finetune(
                    {
                        "epoch": epoch,
                        "model_state": model_state,
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "best_test_metrics": best_test_metrics,
                        "label_norm": label_norm,
                        "args": vars(args),
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
                    f"val_rmse={val_metrics['RMSE']:.{METRIC_PRINT_DIGITS}f} < {args.target_rmse:.{METRIC_PRINT_DIGITS}f} "
                    f"(max_epoch={args.target_rmse_max_epoch})"
                )
            break

        if args.early_stop_patience and epoch >= int(getattr(args, "early_stop_start_epoch", 1)):
            start_epoch = int(getattr(args, "early_stop_start_epoch", 1))
            effective_best_epoch = best_epoch if best_epoch >= start_epoch else (start_epoch - 1)
            if epoch - effective_best_epoch >= args.early_stop_patience:
                if is_main_process(args.rank):
                    print(
                        f"Early stopping triggered at epoch {epoch}, "
                        f"best was epoch {best_epoch}, "
                        f"effective_best_for_patience was {effective_best_epoch}, "
                        f"best score was {best_score:.{METRIC_PRINT_DIGITS}f}"
                    )
                break


def prepare_pretraining(model, args):
    # -------- Dataloader & model --------
    model = model.cuda()
    train_path = args.pretrain_train_path
    val_path   = args.pretrain_val_path
    pre_loader, pre_sampler = build_dataloader(train_path, args, mode="full")
    val_loader, _           = build_dataloader(val_path,   args, mode="full")

    # -------- Directory setup --------
    run_dir  = Path(args.results_root) / "random_pretrain"
    ckpt_dir = run_dir / "checkpoints"
    log_dir  = run_dir / "logs"
    if is_main_process(args.rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True,  exist_ok=True)

    run_pretraining(model, args, pre_loader, pre_sampler, val_loader, run_dir)


def main(argv=None):
    args = parse_args(argv)
    configure_runtime()
    args.distributed = args.world_size > 1 or args.distributed
    set_seed(args.seed)
    init_distributed(args)
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
