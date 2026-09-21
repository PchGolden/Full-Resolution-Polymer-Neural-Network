# Uni-Macro-FRPN

Code and data for **Uni-Macro-FRPN: Full-Resolution and Cross-Scale Learning for Polymers** by Jintao Wu, Yiran Shan and Rui Zhang.

FRPN connects learned monomer representations with an explicit polymer graph. The repository contains BCDB classification, linear homopolymer regression, and the 1,640-record MD benchmark.

## Contents

| Path | Contents |
| --- | --- |
| `frpn/pipelines/bcdb/` | BCDB and homopolymer model, preprocessing, training and evaluation |
| `frpn/pipelines/md_final1640_v2/` | MD graph model, preprocessing, training and evaluation |
| `data/raw/`, `data/splits/` | Benchmark CSVs and the manuscript's fixed folds |
| `data/lammps/` | 1,640 final atomistic LAMMPS data files, coefficient/charge audit and row mapping |
| `scripts/` | Executable reproduction, export, normalization and figure scripts |
| `scripts/configs/` | Original run settings, including model variants and seed 42 |
| `results/oof_predictions/` | Retained out-of-fold predictions for result reproduction |
| `results/summaries/`, `results/diagnostics/` | Benchmark and predictive-behavior result tables |
| `docs/checkpoint_manifest.csv` | Model, target, fold, file size and SHA-256 mapping for checkpoints |

Checkpoints and larger processed/embedding caches are prepared separately for Zenodo. In the local publication bundle they are under `../zenodo/`. The Zenodo record has not yet been published; its download link will be added here after deposit. See [asset layout](docs/checkpoint_download.md).

## Setup and quick verification

Run commands from this repository root. The supplied environment specifies Python 3.10. CLI and small-input checks were run in the existing Python 3.9.20 / PyTorch 2.5.0 environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
python -m frpn.cli.train_bcdb --help
python -m frpn.cli.train_md --help
python -m frpn.cli.export_oof bcdb --help
python -m frpn.cli.export_oof md --help
```

`environment.yml` supplies an alternative CPU environment for preprocessing, diagnostics and CPU evaluation. Training uses CUDA; use a CUDA-enabled PyTorch installation and GPU host for the training commands below. Check it with `python -c "import torch; print(torch.cuda.is_available())"`.

With the complete packed publication bundle or the complete extracted Zenodo assets available:

```bash
python scripts/verify_release.py --zenodo ../zenodo --lammps-hashes
```

This checks all 1,640 LAMMPS checksums and the paths/sizes of either packed checkpoint archives or extracted checkpoints. Add `--archive-hashes` to hash packed checkpoint TARs, or `--checkpoint-hashes` to hash the extracted model files. See [download and extraction](docs/checkpoint_download.md) for selected-model downloads.

## Reproduction commands

The full commands for preprocessing all three datasets, training model variants and evaluating released checkpoints are in [reproduce_paper.md](docs/reproduce_paper.md). For example, this command applies the released MD target transformations using only fold 0's training rows for normalization:

```bash
mkdir -p reproduced/targets
python scripts/preprocess/normalize_targets.py \
  --input data/splits/md_final1640_v2/with_fold.csv --fold 0 \
  --output reproduced/targets/fold0.csv \
  --parameters reproduced/targets/fold0_normalization.json
```

To reproduce the revised 1,417 structure-control pairs and their metrics from the retained predictions:

```bash
python scripts/evaluate/reproduce_structure_pairs.py \
  --output reproduced/structure_pairs
```

The released BCDB/homopolymer linear-chain allocator rescales requested monomer counts proportionally when their sum exceeds 1,536; integer allocation preserves their composition as closely as the token budget permits. MD models consume the explicit supplied graph.

## Data and model provenance

`data/lammps/dataset_record_mapping.csv` links the one-based `dataset_record` and zero-based `dataset_row` to each final data file. MD polymer graph fields and target processing are described in [data/README.md](data/README.md).

See [Code and data availability](docs/code_data_availability.md) and [third-party baselines](docs/third_party_baselines.md).
