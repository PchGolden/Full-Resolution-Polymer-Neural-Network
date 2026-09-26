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
    _label_transform_torch, _label_inverse_transform_torch,
)


def maybe_mask_temperature_feature(batch, args):
    """Optionally remove temperature channels from batch features in-place."""
    if not getattr(args, "mask_temperature_feature", False):
        return

    if "glob_feat" in batch and "glob_mask" in batch:
        glob_feat = batch["glob_feat"]
        glob_mask = batch["glob_mask"]
        if torch.is_tensor(glob_feat) and torch.is_tensor(glob_mask) and glob_feat.dim() == 2 and glob_feat.size(1) > 0:
            glob_feat[:, 0] = 0.0
            glob_mask[:, 0] = 0.0
            if "glob_valid_mask" in batch and torch.is_tensor(batch["glob_valid_mask"]):
                batch["glob_valid_mask"] = (glob_mask.sum(dim=1, keepdim=True) > 0).float()

    if "chain_glob_feat" in batch and "chain_glob_mask" in batch:
        c_feat = batch["chain_glob_feat"]
        c_mask = batch["chain_glob_mask"]
        if torch.is_tensor(c_feat) and torch.is_tensor(c_mask) and c_feat.dim() == 2 and c_feat.size(1) > 0:
            c_feat[:, 0] = 0.0
            c_mask[:, 0] = 0.0


def train_one_epoch(model, loader, optimizer, scaler, epoch, args, sampler, label_norm=None, label_names=None):
    """Train one epoch, including an update for a final partial accumulation group."""
    model.train()

    mean_torch = None
    std_torch = None
    clamp_min = None
    log10_labels = None
    if label_norm is not None and label_names is not None and args.task_type == "reg":
        mean_torch = torch.tensor(label_norm["mean"], device="cuda", dtype=torch.float32)
        std_torch = torch.tensor(label_norm["std"], device="cuda", dtype=torch.float32)
        clamp_min = float(label_norm.get("log10_clamp_min", getattr(args, "log10_clamp_min", 1e-16)))
        log10_labels = set(label_norm.get("log10_labels", []))

    if getattr(args, "freeze_encoder", False):
        m = model.module if hasattr(model, "module") else model
        m.embed_tokens.eval()
        m.atom_feature.eval()
        m.edge_feature.eval()
        m.encoder.eval()
        m.se3_invariant_kernel.eval()
        m.reg_head.train()

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
        maybe_mask_temperature_feature(batch, args)

        with autocast(enabled=args.amp):
            # ---------- 1. Forward ----------
            pred = model(batch)

            # ---------- 2. Loss ----------
            if args.task_type == "reg":
                raw_label = batch["label"]
                label = raw_label.float().cuda()

                if mean_torch is not None and std_torch is not None:
                    label_t = _label_transform_torch(
                        label,
                        label_names,
                        log10_labels=log10_labels,
                        clamp_min=clamp_min,
                    )
                    label = (label_t - mean_torch) / std_torch

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
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, args, label_norm=None, label_names=None):
    """Report classification metrics or regression errors in model and raw units."""
    model.eval()

    if args.task_type == "reg":
        sum_abs_z = 0.0
        sum_sq_z = 0.0
        sum_label_z = 0.0
        sum_label_z_sq = 0.0
        sum_abs_raw = 0.0
        sum_sq_raw = 0.0
        sum_label_raw = 0.0
        sum_label_raw_sq = 0.0
        count = 0
        all_residuals = []  # store z residuals if normalized, else raw residuals
    else:
        conf_matrix = None
        C = args.num_tasks
        conf_matrix = np.zeros((C, C), dtype=np.int64)
        all_preds = []
        all_labels = []

    mean_t = None
    std_t = None
    clamp_min = None
    log10_labels = None
    use_norm = bool(
        args.task_type == "reg"
        and label_norm is not None
        and label_names is not None
        and getattr(args, "label_zscore", False)
    )
    if use_norm:
        mean_t = torch.tensor(label_norm["mean"], device="cuda", dtype=torch.float32)
        std_t = torch.tensor(label_norm["std"], device="cuda", dtype=torch.float32)
        clamp_min = float(label_norm.get("log10_clamp_min", getattr(args, "log10_clamp_min", 1e-16)))
        log10_labels = set(label_norm.get("log10_labels", []))

    for batch in loader:
        # Move batch to CUDA
        move_batch_to_cuda(batch)
        maybe_mask_temperature_feature(batch, args)

        # Forward pass
        pred = model(batch)
        label_raw = batch["label"].float().cuda()

        if args.task_type == "reg":
            if use_norm:
                label_trans = _label_transform_torch(
                    label_raw,
                    label_names,
                    log10_labels=log10_labels,
                    clamp_min=clamp_min,
                )
                label_z = (label_trans - mean_t) / std_t
                pred_z = pred
                residual_z = (pred_z - label_z).detach().cpu().numpy().flatten()

                pred_trans = pred_z * std_t + mean_t
                pred_raw_proc = _label_inverse_transform_torch(pred_trans, label_names, log10_labels=log10_labels)
                label_raw_proc = _label_inverse_transform_torch(label_trans, label_names, log10_labels=log10_labels)
                residual_raw = (pred_raw_proc - label_raw_proc).detach().cpu().numpy().flatten()

                if is_main_process(args.rank):
                    all_residuals.append(residual_z)

                sum_abs_z += np.abs(residual_z).sum()
                sum_sq_z += (residual_z ** 2).sum()
                label_z_cpu = label_z.detach().cpu().numpy().flatten()
                sum_label_z += label_z_cpu.sum()
                sum_label_z_sq += (label_z_cpu ** 2).sum()

                sum_abs_raw += np.abs(residual_raw).sum()
                sum_sq_raw += (residual_raw ** 2).sum()
                label_raw_cpu = label_raw_proc.detach().cpu().numpy().flatten()
                sum_label_raw += label_raw_cpu.sum()
                sum_label_raw_sq += (label_raw_cpu ** 2).sum()
                count += len(residual_z)

            else:
                pred_cpu = pred.detach().cpu().numpy().flatten()
                label_cpu = label_raw.detach().cpu().numpy().flatten()
                residuals = pred_cpu - label_cpu

                if is_main_process(args.rank):
                    all_residuals.append(residuals)

                sum_abs_raw += np.abs(residuals).sum()
                sum_sq_raw += (residuals ** 2).sum()
                sum_label_raw += label_cpu.sum()
                sum_label_raw_sq += (label_cpu ** 2).sum()
                count += len(residuals)
        else:
            pred_cpu = pred.cpu().numpy()
            label_cpu = label_raw.cpu().numpy().flatten()
            pred_cls = pred_cpu.argmax(axis=-1).flatten()

            all_preds.append(pred_cls)
            all_labels.append(label_cpu)

            for t, p in zip(label_cpu, pred_cls):
                conf_matrix[int(t), int(p)] += 1

    if args.task_type == "reg":
        if args.distributed:
            stats = torch.tensor(
                [
                    sum_abs_z, sum_sq_z, sum_label_z, sum_label_z_sq,
                    sum_abs_raw, sum_sq_raw, sum_label_raw, sum_label_raw_sq,
                    count,
                ],
                device="cuda",
                dtype=torch.float32,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            (
                sum_abs_z, sum_sq_z, sum_label_z, sum_label_z_sq,
                sum_abs_raw, sum_sq_raw, sum_label_raw, sum_label_raw_sq,
                count,
            ) = stats.cpu().numpy()

        if count == 0:
            return {"MAE": 0, "RMSE": 0, "R2": 0, "residuals": None}

        mae_raw = sum_abs_raw / count
        rmse_raw = np.sqrt(sum_sq_raw / count)
        total_var_raw = sum_label_raw_sq - (sum_label_raw ** 2) / count
        r2_raw = 1.0 - (sum_sq_raw / total_var_raw) if total_var_raw > 1e-7 else 0.0

        if label_norm is not None and label_names is not None and getattr(args, "label_zscore", False):
            mae_z = sum_abs_z / count
            rmse_z = np.sqrt(sum_sq_z / count)
            total_var_z = sum_label_z_sq - (sum_label_z ** 2) / count
            r2_z = 1.0 - (sum_sq_z / total_var_z) if total_var_z > 1e-7 else 0.0
        else:
            mae_z = mae_raw
            rmse_z = rmse_raw
            r2_z = r2_raw

        residuals_concat = np.concatenate(all_residuals) if is_main_process(args.rank) else None
        return {
            "MAE": mae_z,
            "RMSE": rmse_z,
            "R2": r2_z,
            "MAE_raw": mae_raw,
            "RMSE_raw": rmse_raw,
            "R2_raw": r2_raw,
            "residuals": residuals_concat
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
    return _evaluate_pretrain(model, loader, args, maybe_mask_temperature_feature)
