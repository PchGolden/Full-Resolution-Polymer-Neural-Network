"""Runtime setup and gradient updates used by both training workflows."""
import random
import warnings
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def configure_runtime():
    """Apply training-process settings only when a training command starts."""
    mp.set_sharing_strategy("file_system")
    warnings.filterwarnings("ignore", category=UserWarning)


def move_batch_to_cuda(batch):
    """Move tensor fields in place, preserving non-tensor batch metadata."""
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    return batch


def move_batch_to_device(batch, device):
    """Copy a batch mapping onto the model's device, preserving metadata."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def optimizer_step(model, optimizer, scaler, grad_clip, scheduler=None):
    """Unscale, clip, update and clear gradients in the original training order."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    """Initialize torch.distributed"""
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        assert dist.is_initialized()
    else:
        args.rank = 0
        args.world_size = 1


def is_main_process(rank: int) -> bool:
    return rank == 0
