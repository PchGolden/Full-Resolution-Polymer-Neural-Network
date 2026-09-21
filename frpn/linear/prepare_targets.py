#!/usr/bin/env python3
"""Split the preprocessed homopolymer dataset into six target-specific PKLs.

The samples and stored fold assignments are retained from the input PKL.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


TARGET_FILES = {
    "density": "density",
    "Rg": "rg",
    "self-diffusion": "self_diffusion",
    "Cp": "cp",
    "dielectric_const_dc": "dielectric_const",
    "refractive_index": "refractive_index",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Trusted preprocessed homopolymer split.pkl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("rb") as stream:
        header = pickle.load(stream)
    labels = header["label_names"]
    samples = header["samples"]
    missing = set(TARGET_FILES) - set(labels)
    if missing:
        raise ValueError(f"Missing expected homopolymer targets: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, file_key in TARGET_FILES.items():
        out = dict(header)
        out["label_names"] = [label]
        out["samples"] = [dict(sample, label={label: sample["label"][label]}) for sample in samples]
        destination = args.output_dir / f"split_{file_key}.pkl"
        with destination.open("wb") as stream:
            pickle.dump(out, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {destination}: {len(samples)} samples, preserved fold assignments")


if __name__ == "__main__":
    main()
