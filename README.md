# Uni-Macro-FRPN

**Note that, the codes of this version of FRPN published on Github has been sigfinicantly optimized for readability. We have verified that the behavior of the model remain unchanged. Contact me if you would like use original version for reference.**

Code and data for **Uni-Macro-FRPN: Full-Resolution and Cross-Scale Learning for Polymers** by Jintao Wu, Yiran Shan and Rui Zhang.

FRPN connects learned monomer representations with an explicit polymer graph. The repository contains BCDB classification, linear homopolymer regression, and the 1,640-record MD benchmark.

## Repository layout

```text
frpn/
  common/              shared attention and molecular graph features
  linear/              BCDB and homopolymer workflows
    models/            monomer encoder, chain model and proportional rescaling
    pretraining/       auxiliary BCDB unsupervised training and evaluation
  md/                  explicit-graph MD workflow and target preparation
    models/            MD monomer and polymer graph models
  analysis/            plots, summaries and predictive-behavior diagnostics
configs/slurm/         BCDB, homopolymer and MD job configurations
data/                 raw benchmarks, fixed folds and LAMMPS inputs
results/               retained OOF predictions, summaries and diagnostics
docs/                  reproduction guide, asset manifests and source map
scripts/               release verification and manifest maintenance
tests/                 CPU regression and package integration checks
```

BCDB and homopolymers share the `linear` implementation because both construct chains from monomer identities and repeat counts. MD reads the graph supplied in each record and has its own model and data loader. Identical attention and molecular graph utilities live in `common`. Each workflow exposes `preprocess`, `train` and `export` modules without import-path wrappers.

See the [source and entry-point map](docs/architecture.md) for dependencies and the [reproduction guide](docs/reproduce_paper.md) for complete commands. Existing data, results and extracted checkpoint paths are retained.

Checkpoints and larger processed/embedding caches are prepared separately for Zenodo. In the local publication bundle they are under `../zenodo/`. The Zenodo record has not yet been published; its download link will be added here after deposit. See [asset layout](docs/checkpoint_download.md).

## Setup and quick verification

Run commands from this repository root. The supplied environment specifies Python 3.10. CLI and small-input checks were run in the existing Python 3.9.20 / PyTorch 2.5.0 environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/verify_release.py
python scripts/file_manifest.py
python -m unittest discover -s tests -v
python -m frpn.linear.train --help
python -m frpn.md.train --help
python -m frpn.linear.export --help
python -m frpn.md.export --help
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
python -m frpn.md.normalize_targets \
  --input data/splits/md_final1640_v2/with_fold.csv --fold 0 \
  --output reproduced/targets/fold0.csv \
  --parameters reproduced/targets/fold0_normalization.json
```

To reproduce the revised 1,417 structure-control pairs and their metrics from the retained predictions:

```bash
python -m frpn.analysis.structure_pairs \
  --output reproduced/structure_pairs
```

The released BCDB/homopolymer linear-chain allocator rescales requested monomer counts proportionally when their sum exceeds 1,536; integer allocation preserves their composition as closely as the token budget permits. MD models consume the explicit supplied graph.

## Data and model provenance

`data/lammps/dataset_record_mapping.csv` links the one-based `dataset_record` and zero-based `dataset_row` to each final data file. MD polymer graph fields and target processing are described in [data/README.md](data/README.md).

See [Code and data availability](docs/code_data_availability.md) and [third-party baselines](docs/third_party_baselines.md).

The [reorganization validation record](docs/reorganization.md) documents the checks against the initial release.
