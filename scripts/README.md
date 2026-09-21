# Executable scripts

- `verify_release.py`: row mappings, expected OOF coverage, pair counts and optional file checksum verification; no training.
- `preprocess/`: wrappers for canonical BCDB/homopolymer and MD preprocessing, single-target homopolymer caches, and training-fold target normalization.
- `evaluate/`: real MD checkpoint inference and the final 1,417-pair diagnostic reproduction.
- `figures/`: BCDB/homopolymer plots from retained caches and MD predictive-behavior diagnostics.
- `configs/`: original manuscript model settings and seed 42 in scheduler-job form; adjust cluster-specific paths if using those jobs directly.
- `train/`: lightweight package-layout checks; model training is exposed through `python -m frpn.cli.train_bcdb`, `train_homopolymer` and `train_md`.

`python -m frpn.cli.export_oof bcdb --help` and `python -m frpn.cli.export_oof md --help` expose actual checkpoint exporters. Complete commands are in `docs/reproduce_paper.md`.
