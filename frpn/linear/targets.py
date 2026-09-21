"""Label selection and transforms for this workflow."""
import numpy as np
import torch
from frpn.training.targets import fit_label_normalization


def _parse_comma_list(v: str) -> list[str]:
    if v is None:
        return []
    v = str(v).strip()
    if not v:
        return []
    return [s.strip() for s in v.split(",") if s.strip()]


def _label_transform_torch(
    y_raw: torch.Tensor,
    label_names: list[str],
    *,
    log10_labels: set[str],
    clamp_min: float,
) -> torch.Tensor:
    if not log10_labels:
        return y_raw
    y = y_raw.clone()
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = torch.log10(torch.clamp(y[:, i], min=float(clamp_min)))
    return y


def _label_inverse_transform_torch(
    y_trans: torch.Tensor,
    label_names: list[str],
    *,
    log10_labels: set[str],
) -> torch.Tensor:
    if not log10_labels:
        return y_trans
    y = y_trans.clone()
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = torch.pow(10.0, y[:, i])
    return y


def _compute_label_norm_from_train_dataset(train_dataset, label_names, log10_labels, clamp_min):
    return fit_label_normalization(train_dataset, label_names, log10_labels, clamp_min, "log10_clamp_min")
