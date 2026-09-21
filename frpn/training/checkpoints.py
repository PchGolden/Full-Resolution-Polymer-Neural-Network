"""Checkpoint serialization and shape-compatible monomer initialization."""
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def save_json(obj: Dict[str, Any], path: Path):
    def convert(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def save_checkpoint_pretrain(state: Dict[str, Any], ckpt_dir: Path, step: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{step}.pt"
    torch.save(state, ckpt_path)


def save_checkpoint_finetune(state: Dict[str, Any], ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint.pt"
    torch.save(state, ckpt_path)


def load_monomer_weights(model, weight_path, args, *, include_pretrain_heads=True):
    """Load matching monomer tensors without overwriting prediction/chain heads."""
    ckpt = torch.load(weight_path, map_location="cpu")

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model_dict = model.state_dict()

    allowed_prefixes = (
        "embed_tokens.",
        "atom_feature.",
        "edge_feature.",
        "encoder.",
        "se3_invariant_kernel.",
    )

    if include_pretrain_heads:
        allowed_prefixes = allowed_prefixes + (
            "lm_head.",
            "movement_pred_head.",
        )

    skip_prefixes = (
        "reg_head.",
        "chain_token_feature.",
        "chain_edge_feature.",
        "chain_encoder.",
    )

    load_dict = {}
    for k, v in state_dict.items():
        if k not in model_dict:
            continue
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if any(k.startswith(p) for p in allowed_prefixes):
            if model_dict[k].shape == v.shape:
                load_dict[k] = v

    missing, unexpected = model.load_state_dict(load_dict, strict=False)

    if args.rank == 0:
        print("====== Loaded monomer-level pretrained weights ======")
        print(f"Loaded keys: {len(load_dict)}")
        print(f"Missing keys (expected): {missing}")
        print(f"Unexpected keys (ignored): {unexpected}")
        print("====================================================")

    return model
