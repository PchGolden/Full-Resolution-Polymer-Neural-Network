"""Label selection and transforms for this workflow."""
import numpy as np
import torch
from frpn.training.targets import fit_label_normalization

MD_REG_LABEL_NAMES = [
    "density",
    "Rg",
    "D",
    "MSD_A2",
    "S_q_peak",
    "q_peak_invA",
    "eta0_Pa_s",
    "tau_maxwell_s",
    "omega_cross_rad_s",
]
MD_LOG10_LABELS_DEFAULT = {
    "D",
    "MSD_A2",
    "eta0_Pa_s",
    "tau_maxwell_s",
    "omega_cross_rad_s",
    "Cp",
    "K_T",
}
MD_D_CLAMP_MIN = 1e-16


def _parse_log10_labels_arg(s: str):
    if s is None:
        return set()
    items = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            items.append(x)
    return set(items)


def _resolve_label_names_from_dataset(dataset, args):
    raw_label_names = getattr(dataset, "label_names", None)
    if raw_label_names:
        raw_label_names = list(raw_label_names)
    else:
        raw_label_names = list(MD_REG_LABEL_NAMES[: int(getattr(args, "num_tasks", 1))])

    li = int(getattr(args, "label_index", -1))
    if li >= 0:
        # Single-label mode: pick one label by index.
        if int(getattr(args, "num_tasks", 1)) != 1:
            raise ValueError("--label_index requires --num_tasks=1 for single-label training")
        if raw_label_names and 0 <= li < len(raw_label_names):
            return [str(raw_label_names[li])]
        if 0 <= li < len(MD_REG_LABEL_NAMES):
            return [str(MD_REG_LABEL_NAMES[li])]
        raise ValueError(f"label_index={li} out of range")

    label_names = raw_label_names

    if int(getattr(args, "num_tasks", 1)) == len(MD_REG_LABEL_NAMES) and label_names != MD_REG_LABEL_NAMES:
        raise ValueError(
            f"Expected label_names={MD_REG_LABEL_NAMES}, got {label_names}. "
            "Please preprocess with explicit --labels in the expected order."
        )
    if len(label_names) != int(getattr(args, "num_tasks", 1)):
        raise ValueError(f"label_names length mismatch: len(label_names)={len(label_names)} num_tasks={args.num_tasks}")

    return label_names


def _label_transform_torch(
    y_raw: torch.Tensor,
    label_names,
    log10_labels,
    d_clamp_min: float = MD_D_CLAMP_MIN,
) -> torch.Tensor:
    y = y_raw
    if y.dim() == 1:
        y = y.unsqueeze(-1)

    y = y.clone()
    log10_labels = set(log10_labels or [])
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = torch.log10(torch.clamp(y[:, i], min=float(d_clamp_min)))
    return y


def _label_inverse_transform_np(y_trans: np.ndarray, label_names, log10_labels) -> np.ndarray:
    y = np.array(y_trans, dtype=np.float64, copy=True)
    log10_labels = set(log10_labels or [])
    for i, name in enumerate(label_names):
        if name in log10_labels:
            y[:, i] = np.power(10.0, y[:, i])
    return y


def _compute_label_norm_from_train_dataset(train_dataset, label_names, log10_labels, d_clamp_min=MD_D_CLAMP_MIN):
    return fit_label_normalization(train_dataset, label_names, log10_labels, d_clamp_min, "d_clamp_min")
