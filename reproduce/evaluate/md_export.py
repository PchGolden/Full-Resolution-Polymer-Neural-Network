#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export held-out predictions from the released MD model checkpoints.

Each checkpoint is evaluated on its validation fold, and the five folds are
merged into one out-of-fold table. From the repository root:

python -m reproduce.evaluate.md_export \
  --raw_csv datasets/raw/md_final1640_v2/final_1640dataset.csv \
  --pkl_path ../zenodo/data_cache/md_final1640_v2/main/split.pkl \
  --results_root ../zenodo/checkpoints/md_final1640_v2 \
  --labels density,Rg,D,S_q_peak,nematic_order,dielectric_constant,refractive_index \
  --models FRPN --out_dir reproduce/outputs/md_oof --cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from frpn.md.data import build_dataloader
from frpn.md.models.model import MultiMolModel


def _split_csv_list(s: str) -> List[str]:
    if s is None:
        return []
    out: List[str] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def _json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _inverse_pred_to_raw(
    pred_z: np.ndarray,
    label_norm: Dict,
) -> np.ndarray:
    """
    pred_z: [N, 1] (model output is in z-score space of transformed labels)
    label_norm: dict with keys: label_names, log10_labels, mean, std, d_clamp_min
    """
    mean = np.asarray(label_norm["mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(label_norm["std"], dtype=np.float64).reshape(1, -1)
    pred_trans = pred_z.astype(np.float64) * std + mean

    label_names = list(label_norm.get("label_names", []))
    log10_labels = set(label_norm.get("log10_labels", []))

    pred_raw = pred_trans.copy()
    for j, name in enumerate(label_names):
        if name in log10_labels:
            pred_raw[:, j] = np.power(10.0, pred_raw[:, j])
    return pred_raw


def _find_checkpoint_for_fold(
    results_root: Path,
    model_name: str,
    label_index: int,
    label_name: str,
    fold: int,
) -> Optional[Path]:
    """
    Expected path layout:
      results_root/<model_name>/label{idx}_{label}/fold_{k}/wd_.../checkpoints/checkpoint.pt
    """
    packaged = results_root / model_name / label_name / f"{model_name}__{label_name}__fold{fold}.pt"
    if packaged.is_file():
        return packaged
    base = results_root / model_name / f"label{label_index}_{label_name}" / f"fold_{fold}"
    if not base.exists():
        return None

    candidates = list(base.glob("wd_*/*/checkpoint.pt"))
    # More strict pattern: wd_*/checkpoints/checkpoint.pt
    candidates = list(base.glob("wd_*/checkpoints/checkpoint.pt"))
    if not candidates:
        # fallback: search deeper
        candidates = list(base.glob("**/checkpoints/checkpoint.pt"))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass
class FoldPred:
    row_idx: np.ndarray
    y_true_raw: np.ndarray
    y_pred_raw: np.ndarray


@torch.no_grad()
def _infer_fold(
    ckpt_path: Path,
    pkl_path: Path,
    fold: int,
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> FoldPred:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state"]
    label_norm = ckpt.get("label_norm", None)
    if label_norm is None:
        raise ValueError(f"checkpoint missing label_norm: {ckpt_path}")

    args_dict = dict(ckpt.get("args", {}))
    args_dict.update(
        dict(
            pkl_path=str(pkl_path),
            fold=int(fold),
            distributed=False,
            world_size=1,
            rank=0,
            local_rank=0,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
    )
    args = argparse.Namespace(**args_dict)

    model = MultiMolModel(args)
    # Some buffers (e.g., fp_table) can have dynamic shapes; load_state_dict will
    # hard-error on size mismatch even with strict=False. Handle them explicitly.
    fp_table = None
    if isinstance(state, dict) and "fp_table" in state:
        fp_table = state.pop("fp_table")
    model.load_state_dict(state, strict=False)
    if fp_table is not None:
        # Restore buffer and mark as built if non-empty so RDKit rebuild is not needed.
        model.fp_table = fp_table
        try:
            model._fp_table_built = bool(getattr(model, "fp_table").numel() > 0)
        except Exception:
            model._fp_table_built = True
    model.to(device)
    model.eval()

    val_loader, _ = build_dataloader(str(pkl_path), args, mode="val")

    row_idxs: List[int] = []
    y_true_raw: List[float] = []
    pred_z_all: List[float] = []

    li = int(getattr(args, "label_index", -1))
    if li < 0:
        raise ValueError("OOF export expects single-label runs with args.label_index >= 0")

    for batch in val_loader:
        # Move tensor fields to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        pred_z = model(batch)  # [B, 1]
        if pred_z.dim() == 1:
            pred_z = pred_z.unsqueeze(-1)

        label = batch["label"].float()
        if label.dim() == 1:
            label = label.unsqueeze(-1)
        if li >= label.size(-1):
            raise ValueError(f"label_index={li} out of range for label dim={label.size(-1)}")
        label_one = label[:, li : li + 1]

        ridx = batch.get("row_idx", None)
        if ridx is None:
            raise KeyError("batch missing row_idx; please preprocess with row_idx")
        if not isinstance(ridx, list):
            # collate_fn should keep non-tensors as list
            ridx = list(ridx)

        row_idxs.extend([int(x) for x in ridx])
        y_true_raw.extend([float(x) for x in label_one.detach().cpu().numpy().reshape(-1)])
        pred_z_all.extend([float(x) for x in pred_z.detach().cpu().numpy().reshape(-1)])

    pred_z_arr = np.asarray(pred_z_all, dtype=np.float64).reshape(-1, 1)
    pred_raw = _inverse_pred_to_raw(pred_z_arr, label_norm=label_norm).reshape(-1)

    return FoldPred(
        row_idx=np.asarray(row_idxs, dtype=np.int64),
        y_true_raw=np.asarray(y_true_raw, dtype=np.float64),
        y_pred_raw=np.asarray(pred_raw, dtype=np.float64),
    )


def _build_sample_meta(raw_csv: Path) -> pd.DataFrame:
    import json as _json

    df = pd.read_csv(raw_csv)
    df = df.copy()
    df["row_idx"] = np.arange(len(df), dtype=np.int64)

    # chain_len from JSON list stored as string
    def _len_from_json_list(s):
        try:
            x = _json.loads(s)
            return int(len(x))
        except Exception:
            return int(-1)

    if "chain_node_seg_id" in df.columns:
        df["chain_len"] = df["chain_node_seg_id"].map(_len_from_json_list).astype(np.int64)
    else:
        df["chain_len"] = -1

    # Keep only columns needed for downstream structure-controlled analysis.
    keep_cols = [
        "row_idx",
        "topology",
        "mix_mode",
        "glob_feat0",
        "rigid_ratio",
        "polar_ratio",
        "chem_profile_id",
        "chain_len",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--labels", required=True, help="Comma-separated label names")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv)
    pkl_path = Path(args.pkl_path)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _split_csv_list(args.labels)
    models = _split_csv_list(args.models)

    device = torch.device("cpu") if args.cpu or (not torch.cuda.is_available()) else torch.device("cuda")
    print(f"[OOF] device={device} torch_cuda_available={torch.cuda.is_available()}")

    # Load pkl header to resolve label indices (requires torch to be importable for unpickle)
    import pickle

    header = pickle.load(open(pkl_path, "rb"))
    pkl_label_names = list(header.get("label_names", []))
    name_to_index = {str(n): int(i) for i, n in enumerate(pkl_label_names)}

    unknown_labels = [l for l in labels if l not in name_to_index]
    if unknown_labels:
        raise ValueError(f"Labels not found in pkl label_names: {unknown_labels}; available={pkl_label_names}")

    # Save sample metadata for matching logic (idempotent under job arrays).
    meta_path = out_dir / "sample_meta.csv"
    if not meta_path.exists():
        meta = _build_sample_meta(raw_csv)
        meta.to_csv(meta_path, index=False)
        print(f"[OOF] wrote sample_meta: {meta_path} (rows={len(meta)})")
    else:
        print(f"[OOF] sample_meta exists: {meta_path}")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_csv": str(raw_csv),
        "pkl_path": str(pkl_path),
        "results_root": str(results_root),
        "labels": labels,
        "models": models,
        "device": str(device),
        "exported": {},
        "missing": [],
    }

    for model_name in models:
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for label_name in labels:
            label_index = name_to_index[label_name]
            fold_tables: List[pd.DataFrame] = []

            for fold in range(5):
                ckpt_path = _find_checkpoint_for_fold(
                    results_root=results_root,
                    model_name=model_name,
                    label_index=label_index,
                    label_name=label_name,
                    fold=fold,
                )
                if ckpt_path is None:
                    report["missing"].append(
                        {
                            "model": model_name,
                            "label": label_name,
                            "label_index": int(label_index),
                            "fold": int(fold),
                            "reason": "checkpoint_not_found",
                        }
                    )
                    continue

                print(f"[OOF] infer model={model_name} label={label_name} fold={fold} ckpt={ckpt_path}")
                pred = _infer_fold(
                    ckpt_path=ckpt_path,
                    pkl_path=pkl_path,
                    fold=fold,
                    device=device,
                    num_workers=int(args.num_workers),
                    pin_memory=bool(args.pin_memory),
                )
                df_fold = pd.DataFrame(
                    {
                        "row_idx": pred.row_idx,
                        "y_true_raw": pred.y_true_raw,
                        "y_pred_raw": pred.y_pred_raw,
                        "fold": int(fold),
                    }
                )
                fold_path = model_dir / f"fold_{fold}__{label_name}.csv"
                df_fold.to_csv(fold_path, index=False)
                fold_tables.append(df_fold)

            if not fold_tables:
                continue

            df_all = pd.concat(fold_tables, axis=0, ignore_index=True)
            # Keep one pred per row_idx (val split should be disjoint across folds).
            dup = df_all["row_idx"].duplicated().sum()
            if dup:
                print(f"[OOF][WARN] duplicated row_idx in merged folds: {dup} (keep first)")
                df_all = df_all.drop_duplicates(subset=["row_idx"], keep="first")

            df_all = df_all.sort_values("row_idx").reset_index(drop=True)
            merged_path = out_dir / f"oof_{model_name}__{label_name}.csv"
            df_all.to_csv(merged_path, index=False)

            report["exported"][f"{model_name}__{label_name}"] = {
                "rows": int(len(df_all)),
                "fold_files": int(len(fold_tables)),
                "merged_file": str(merged_path),
            }
            print(f"[OOF] wrote merged: {merged_path} (rows={len(df_all)})")

    # Per-task report to avoid clobbering under job arrays.
    task_tag = f"{os.getpid()}"
    report_path = out_dir / f"export_report__pid{task_tag}.json"
    _json_dump(report, report_path)
    print(f"[OOF] wrote report: {report_path}")


if __name__ == "__main__":
    main()
