"""Training-fold statistics in the transformed label space."""
import numpy as np


def fit_label_normalization(
    train_dataset,
    label_names: list[str],
    log10_labels: list[str],
    clamp_min: float,
    clamp_key: str,
) -> dict[str, object]:
    ys = []
    for sid in getattr(train_dataset, "ids", list(range(len(train_dataset)))):
        sample = train_dataset._get_raw_sample(sid) if hasattr(train_dataset, "_get_raw_sample") else train_dataset[sid]
        lbl = sample.get("label", None)
        if isinstance(lbl, dict):
            row = [lbl.get(k, None) for k in label_names]
        else:
            row = list(lbl)
        ys.append(row)

    y_raw = np.array(ys, dtype=np.float64)
    if np.isnan(y_raw).any():
        raise ValueError("NaN found in training labels; please clean/preprocess dataset")

    log10_set = set(log10_labels or [])
    y_trans = y_raw.copy()
    for i, name in enumerate(label_names):
        if name in log10_set:
            y_trans[:, i] = np.log10(np.clip(y_trans[:, i], a_min=float(clamp_min), a_max=None))

    mean = y_trans.mean(axis=0)
    std = y_trans.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    return {
        "label_names": list(label_names),
        "log10_labels": sorted([k for k in label_names if k in log10_set]),
        clamp_key: float(clamp_min),
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
    }
