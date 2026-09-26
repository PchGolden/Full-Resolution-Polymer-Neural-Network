"""Masked-atom/geometry pretraining shared by the linear and MD workflows."""
import math
from pathlib import Path
from typing import NamedTuple
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from .runtime import is_main_process, optimizer_step, move_batch_to_device
from .checkpoints import save_checkpoint_pretrain


class PretrainingLosses(NamedTuple):
    atom: torch.Tensor
    coordinate: torch.Tensor
    distance: torch.Tensor


def pretraining_losses(logits, pred_pos, pred_dist, batch):
    """Masked CE and per-molecule coordinate/distance errors, with padding excluded."""
    target_token = batch["target_token"]
    token_mask = target_token.ne(0)
    if token_mask.any():
        atom_loss = F.cross_entropy(logits[token_mask], target_token[token_mask], reduction="mean")
    else:
        atom_loss = logits.new_tensor(0.0)
    atom_mask = batch["atom_mask"].bool()
    position_mask = atom_mask.unsqueeze(-1)
    target_pos = batch["target_pos"].float()
    position_error = (pred_pos - target_pos).abs() * position_mask
    coordinate_loss = (position_error.sum((-1, -2)) / (position_mask.sum((-1, -2)) + 1e-10)).mean()
    pair_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
    target_dist = torch.cdist(target_pos, target_pos)
    distance_error = (pred_dist - target_dist).abs() * pair_mask
    distance_loss = (distance_error.sum((-1, -2)) / (pair_mask.sum((-1, -2)) + 1e-10)).mean()
    return PretrainingLosses(atom_loss, coordinate_loss, distance_loss)


@torch.no_grad()
def evaluate_pretrain(model, loader, args, prepare_batch=None):
    model.eval()
    device = next(model.parameters()).device
    running = dict(
        Latom_sum=0.0, Lcoord_sum=0.0, Ldist_sum=0.0,
        n_masked=0, n_atoms=0, n_pairs=0
    )

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if prepare_batch is not None:
            prepare_batch(batch, args)
        logits, pred_pos, pred_dist = model(batch)

        losses = pretraining_losses(logits, pred_pos, pred_dist, batch)
        L_atom, L_coord, L_dist = losses.atom, losses.coordinate, losses.distance
        n_masked = batch["target_token"].ne(0).sum().item()
        atom_mask = batch["atom_mask"].bool()
        n_atoms = atom_mask.sum().item()
        n_pairs = (atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)).sum().item()

        running["Latom_sum"] += L_atom.item() * n_masked
        running["Lcoord_sum"] += L_coord.item() * n_atoms
        running["Ldist_sum"] += L_dist.item() * max(n_pairs, 1)
        running["n_masked"] += n_masked
        running["n_atoms"] += n_atoms
        running["n_pairs"] += n_pairs

    # ---- DDP sync (optional) ----
    if args.distributed:
        keys = ["Latom_sum", "Lcoord_sum", "Ldist_sum", "n_masked", "n_atoms", "n_pairs"]
        tensors = [torch.tensor(running[k], device=device) for k in keys]
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for k, t in zip(keys, tensors):
            running[k] = t.item()

    # ---- Compute average ----
    avg_Latom = running["Latom_sum"] / max(running["n_masked"], 1)
    avg_Lcoord = running["Lcoord_sum"] / max(running["n_atoms"], 1)
    avg_Ldist = running["Ldist_sum"] / max(running["n_pairs"], 1)

    return {
        "Latom": avg_Latom,
        "Lcoord": avg_Lcoord,
        "Ldist": avg_Ldist
    }


def warmup_cosine_multiplier(step: int, warmup: int, total_steps: int) -> float:
    """Learning-rate multiplier evaluated at optimizer-update steps."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def run_pretraining(model, args, pre_loader, pre_sampler, val_loader, run_dir, prepare_batch=None):
    """Run masked reconstruction after the workflow chooses data and output paths.

    Validation and scheduling count optimizer updates. Accumulation follows the
    existing pretraining protocol: incomplete groups carry gradients into the
    next epoch. Resume restores the saved epoch and optimizer/scheduler states.
    """
    device = next(model.parameters()).device
    ckpt_dir = Path(run_dir) / "checkpoints"
    # -------- Optimizer & Scaler --------
    eff_bs   = args.batch_size * args.world_size * args.grad_accum_steps
    base_lr  = args.lr
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=args.weight_decay
    )
    dataset_size   = len(pre_loader.dataset)
    steps_per_epoch = math.ceil(dataset_size / eff_bs)
    total_steps     = steps_per_epoch * args.epochs
    warmup = (
        args.warmup_steps if args.warmup_steps > 0
        else max(1000, int(total_steps * args.warmup_ratio))
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine_multiplier(step, warmup, total_steps)
    )
    scaler = GradScaler(enabled=args.amp)

    # -------- Resume from checkpoint or initialize --------
    ckpt_path = run_dir / "checkpoints" / "checkpoint_randomrandom.pt"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing or unexpected:
            print(f"[Resume][WARN] strict=False missing={len(missing)} unexpected={len(unexpected)}")
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch      = ckpt["epoch"]
        global_step      = ckpt["global_step"]
        best_loss        = ckpt["best_loss"]
        best_step        = ckpt["best_step"]
        intervals_no_improve = ckpt["intervals_no_improve"]
        print(f"[Resume] epoch={start_epoch}  step={global_step}  best_loss={best_loss:.5f}")
    else:
        start_epoch      = 1
        global_step      = 0
        best_loss        = None
        best_step        = 0
        intervals_no_improve = 0

    # -------- DDP wrapper --------
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # -------- High-frequency validation --------
    val_every_steps = getattr(args, "val_every_steps", 20)
    min_delta_abs   = getattr(args, "min_delta_abs", 1e-3)
    min_delta_rel   = getattr(args, "min_delta_rel", 1e-2)

    # -------- Training loop --------
    stop_training = False
    for epoch in range(start_epoch, args.epochs + 1):
        if pre_sampler is not None:
            pre_sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(
            pre_loader,
            disable=not is_main_process(args.rank),
            desc=f"Pre-E{epoch}",
        )

        for step_idx, batch in enumerate(pbar):
            # -------------- Forward & Loss --------------
            batch = move_batch_to_device(batch, device)
            if prepare_batch is not None:
                prepare_batch(batch, args)

            with autocast(enabled=args.amp):
                logits, pred_pos, pred_dist = model(batch)

                losses = pretraining_losses(logits, pred_pos, pred_dist, batch)
                L_atom, L_coord, L_dist = losses.atom, losses.coordinate, losses.distance

                loss = (L_atom + L_coord + L_dist) / args.grad_accum_steps

            # -------------- Backward --------------
            scaler.scale(loss).backward()

            if (step_idx + 1) % args.grad_accum_steps != 0:
                continue

            optimizer_step(model, optimizer, scaler, args.grad_clip, scheduler)

            global_step += 1

            # -------------- Validation & Checkpoint --------------
            if global_step % val_every_steps != 0:
                continue

            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_pretrain(model, val_loader, args, prepare_batch)

            torch.cuda.empty_cache()

            val_total = (val_metrics["Latom"]
                         + val_metrics["Lcoord"]
                         + val_metrics["Ldist"])

            # ---------- Rank-0: decide whether this is an improvement ----------
            if is_main_process(args.rank):
                print(
                    f"[Step {global_step}] quick-val "
                    f"Latom={val_metrics['Latom']:.4f} "
                    f"Lcoord={val_metrics['Lcoord']:.4f} "
                    f"Ldist={val_metrics['Ldist']:.4f} "
                    f"total={val_total:.4f}"
                )

                if best_loss is None:
                    improve = True                       # first ever evaluation
                else:
                    improve = (best_loss - val_total) > max(
                        min_delta_abs,
                        best_loss * min_delta_rel
                    )

                if improve:
                    best_loss = val_total
                    best_step = global_step
                    intervals_no_improve = 0

                    model_state = (model.module.state_dict()
                                   if isinstance(model, DDP)
                                   else model.state_dict())
                    save_checkpoint_pretrain(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state": model_state,
                            "optimizer_state": optimizer.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_loss": best_loss,
                            "best_step": best_step,
                            "intervals_no_improve": intervals_no_improve,
                            "args": vars(args),
                        },
                        ckpt_dir,
                        step=global_step,
                    )
                else:
                    intervals_no_improve += 1

                # rank-0 prepares values to broadcast
                loss_sync  = float(best_loss)
                step_sync  = float(best_step)
                noimp_sync = float(intervals_no_improve)
            else:
                # placeholder values on non-main ranks
                loss_sync  = float("inf")
                step_sync  = 0.0
                noimp_sync = 0.0

            # ---------- broadcast best_* & intervals_no_improve ----------
            if args.distributed and dist.is_initialized():
                sync_tensor = torch.tensor(
                    [loss_sync, step_sync, noimp_sync],
                    dtype=torch.float32,
                    device=device,
                )
                dist.broadcast(sync_tensor, src=0)
                best_loss, best_step, intervals_no_improve = sync_tensor.tolist()
                intervals_no_improve = int(intervals_no_improve)  # cast back to int

            model.train()

            if is_main_process(args.rank):
                patience = args.early_stop_patience
                early_stop_flag = intervals_no_improve >= patience
                flag_tensor = torch.tensor(int(early_stop_flag), device=device)
            else:
                flag_tensor = torch.zeros(1, device=device)

            if args.distributed and dist.is_initialized():
                dist.broadcast(flag_tensor, src=0)
            stop_training = bool(flag_tensor.item())

            if stop_training:
                if is_main_process(args.rank):
                    print(f"[Early-Stop] No improvement for {patience} evals, "
                          f"stop at global_step={global_step}.")
                break

        # Exit outer loop early
        if stop_training:
            break
