"""Supervised epoch updates and evaluation for this workflow."""
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm
from frpn.training.runtime import is_main_process, move_batch_to_cuda, optimizer_step
from frpn.training.pretraining import evaluate_pretrain as _evaluate_pretrain
from .targets import (
    _label_transform_torch, _label_inverse_transform_np,
)
from .targets import MD_D_CLAMP_MIN


def train_one_epoch(model, loader, optimizer, scaler, epoch, args, sampler, label_norm=None, label_names=None):
    """Train on the selected MD targets and report loss plus peak CUDA memory."""
    model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mean_torch = None
    std_torch = None
    d_clamp_min = None
    log10_labels = None
    if label_norm is not None and label_names is not None and args.task_type == "reg":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mean_torch = torch.tensor(label_norm["mean"], device=dev, dtype=torch.float32)
        std_torch = torch.tensor(label_norm["std"], device=dev, dtype=torch.float32)
        d_clamp_min = float(label_norm.get("d_clamp_min", MD_D_CLAMP_MIN))
        log10_labels = set(label_norm.get("log10_labels", []))

    if getattr(args, "freeze_encoder", False):
        m = model.module if hasattr(model, "module") else model
        m.embed_tokens.eval()
        m.atom_feature.eval()
        m.edge_feature.eval()
        m.encoder.eval()
        m.se3_invariant_kernel.eval()
        m.reg_head.train()
        if hasattr(m, "seg_only_projs"):
            m.seg_only_projs.train()

    if sampler is not None:
        sampler.set_epoch(epoch)

    running_loss = 0.0
    total_samples = 0

    for step, batch in tqdm(
        enumerate(loader),
        total=len(loader),
        disable=not is_main_process(args.rank),
        desc=f"Epoch {epoch}"
    ):
        move_batch_to_cuda(batch)

        with autocast(enabled=args.amp):
            # ---------- 1. Forward ----------
            pred = model(batch)

            # ---------- 2. Loss ----------
            if args.task_type == "reg":
                raw_label = batch["label"]
                label = raw_label.float().cuda()
                if getattr(args, "label_index", -1) >= 0:
                    li = int(args.label_index)
                    if label.dim() == 1:
                        label = label.unsqueeze(-1)
                    if li >= label.size(-1):
                        raise ValueError(f"label_index={li} out of range for label dim={label.size(-1)}")
                    label = label[:, li:li + 1]

                if mean_torch is not None and std_torch is not None:
                    label_t = _label_transform_torch(
                        label,
                        label_names,
                        log10_labels=log10_labels,
                        d_clamp_min=d_clamp_min,
                    )
                    label = (label_t.float() - mean_torch) / std_torch

                loss = F.mse_loss(pred, label) / args.grad_accum_steps
            else:
                raw_label = batch["label"].squeeze(-1)
                label = raw_label.long().cuda()
                loss = F.cross_entropy(pred, label) / args.grad_accum_steps

        # ---------- 3. Backward ----------
        scaler.scale(loss).backward()

        # Flush the final batch even when the accumulation group is incomplete.
        if (step + 1) % args.grad_accum_steps == 0 or (step + 1 == len(loader)):
            optimizer_step(model, optimizer, scaler, args.grad_clip)

        running_loss += loss.item() * label.size(0) * args.grad_accum_steps
        total_samples += label.size(0)

    if args.distributed:
        tensor_loss = torch.tensor(running_loss, device="cuda")
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        running_loss = tensor_loss.item()

        tensor_samples = torch.tensor(total_samples, device="cuda")
        dist.all_reduce(tensor_samples, op=dist.ReduceOp.SUM)
        total_samples = tensor_samples.item()

    epoch_loss = running_loss / total_samples
    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = float(torch.cuda.max_memory_allocated()) / (1024.0**3)
    return {"loss": epoch_loss, "peak_mem_gb": peak_mem_gb}


@torch.no_grad()
def evaluate(model, loader, args, label_norm=None, label_names=None):
    """Gather predictions and score normalized, transformed and raw MD targets."""
    model.eval()

    if args.task_type == "reg":
        all_pred_z = []
        all_label_raw = []
    else:
        conf_matrix = None
        C = args.num_tasks
        conf_matrix = np.zeros((C, C), dtype=np.int64)
        all_preds = []
        all_labels = []

    for batch in loader:
        # Move batch to CUDA
        move_batch_to_cuda(batch)

        # Forward pass
        pred = model(batch)
        label = batch["label"].float().cuda()
        if args.task_type == "reg" and getattr(args, "label_index", -1) >= 0:
            li = int(args.label_index)
            if label.dim() == 1:
                label = label.unsqueeze(-1)
            if li >= label.size(-1):
                raise ValueError(f"label_index={li} out of range for label dim={label.size(-1)}")
            label = label[:, li:li + 1]

        if args.task_type == "reg":
            all_pred_z.append(pred.detach().cpu().numpy())
            all_label_raw.append(label.detach().cpu().numpy())
        else:
            pred_cpu = pred.cpu().numpy()
            label_cpu = label.cpu().numpy().flatten()
            pred_cls = pred_cpu.argmax(axis=-1).flatten()

            all_preds.append(pred_cls)
            all_labels.append(label_cpu)

            for t, p in zip(label_cpu, pred_cls):
                conf_matrix[int(t), int(p)] += 1

    if args.task_type == "reg":
        if not all_pred_z:
            return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "residuals": None}

        pred_z = np.concatenate(all_pred_z, axis=0)
        label_raw = np.concatenate(all_label_raw, axis=0)

        if label_norm is None or label_names is None:
            label_names = list(getattr(args, "label_names", [])) or [f"y{i}" for i in range(pred_z.shape[1])]
            label_norm = {
                "label_names": list(label_names),
                "log10_labels": [],
                "d_clamp_min": float(MD_D_CLAMP_MIN),
                "mean": [0.0] * pred_z.shape[1],
                "std": [1.0] * pred_z.shape[1],
            }

        log10_labels = set(label_norm.get("log10_labels", []))
        mean = np.asarray(label_norm["mean"], dtype=np.float64).reshape(1, -1)
        std = np.asarray(label_norm["std"], dtype=np.float64).reshape(1, -1)

        # Ground truth in transformed space (D/tau log10)
        label_trans = label_raw.astype(np.float64, copy=True)
        for i, name in enumerate(label_names):
            if name in log10_labels:
                label_trans[:, i] = np.log10(
                    np.clip(label_trans[:, i], a_min=float(label_norm["d_clamp_min"]), a_max=None)
                )

        label_z = (label_trans - mean) / std
        residuals_z = pred_z.astype(np.float64) - label_z

        def _r2(y_t, y_p):
            ss_res = float(np.sum((y_p - y_t) ** 2))
            ss_tot = float(np.sum((y_t - float(np.mean(y_t))) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        z_per_label = {}
        raw_per_label = {}
        for j, name in enumerate(label_names):
            y_t_z = label_z[:, j]
            y_p_z = pred_z[:, j].astype(np.float64)
            z_per_label[name] = {
                "MAE": float(np.mean(np.abs(y_p_z - y_t_z))),
                "RMSE": float(np.sqrt(np.mean((y_p_z - y_t_z) ** 2))),
                "R2": float(_r2(y_t_z, y_p_z)),
            }

        # Raw-space metrics (after inverse transform). Note: D==0 becomes clamp_min by design.
        pred_trans = pred_z.astype(np.float64) * std + mean
        pred_raw_proc = _label_inverse_transform_np(pred_trans, label_names, log10_labels=log10_labels)
        label_raw_proc = _label_inverse_transform_np(label_trans, label_names, log10_labels=log10_labels)

        for j, name in enumerate(label_names):
            y_t = label_raw_proc[:, j]
            y_p = pred_raw_proc[:, j]
            raw_per_label[name] = {
                "MAE": float(np.mean(np.abs(y_p - y_t))),
                "RMSE": float(np.sqrt(np.mean((y_p - y_t) ** 2))),
                "R2": float(_r2(y_t, y_p)),
            }

        mae_z_macro = float(np.mean([m["MAE"] for m in z_per_label.values()]))
        rmse_z_macro = float(np.mean([m["RMSE"] for m in z_per_label.values()]))
        r2_z_macro = float(np.mean([m["R2"] for m in z_per_label.values()]))

        mae_raw_macro = float(np.mean([m["MAE"] for m in raw_per_label.values()]))
        rmse_raw_macro = float(np.mean([m["RMSE"] for m in raw_per_label.values()]))
        r2_raw_macro = float(np.mean([m["R2"] for m in raw_per_label.values()]))

        return {
            "MAE": mae_z_macro,
            "RMSE": rmse_z_macro,
            "R2": r2_z_macro,
            "MAE_raw": mae_raw_macro,
            "RMSE_raw": rmse_raw_macro,
            "R2_raw": r2_raw_macro,
            "z_per_label": z_per_label,
            "raw_per_label": raw_per_label,
            "residuals": residuals_z,
        }
    else:
        if args.distributed and conf_matrix is not None:
            tensor_conf = torch.tensor(conf_matrix, device="cuda", dtype=torch.int64)
            dist.all_reduce(tensor_conf, op=dist.ReduceOp.SUM)
            conf_matrix = tensor_conf.cpu().numpy()

        if conf_matrix is None:
            conf_matrix = np.zeros((1, 1))
            acc = 0.0
        else:
            correct = conf_matrix.trace()
            total = conf_matrix.sum()
            acc = correct / total if total > 0 else 0.0

        preds_concat = np.concatenate(all_preds) if is_main_process(args.rank) else None
        labels_concat = np.concatenate(all_labels) if is_main_process(args.rank) else None
        return {
            "ACC": acc,
            "confusion": conf_matrix,
            "preds": preds_concat,
            "labels": labels_concat
        }


def evaluate_pretrain(model, loader, args):
    return _evaluate_pretrain(model, loader, args)
