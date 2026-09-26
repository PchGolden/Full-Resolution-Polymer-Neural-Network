# Repository reorganization and validation

Historical record of the September 2026 implementation changes. Module names, paths and validation results below refer to that revision. Use the [current source map](../architecture.md) and [reproduction commands](../../README.md) for this checkout.

Reference revision: `fc98348e118af061debb0dcd21557d82a6735681` (`Initial public release`).

## Source changes

- BCDB and homopolymer code now lives in `frpn.linear`, and explicit-graph MD code in `frpn.md`. Each has direct preprocessing, training and inference entry points.
- Identical attention and molecular graph utilities share `frpn.common`. Family-specific model, encoder, feature and batching implementations remain separate.
- `frpn.analysis` contains the canonical figure, summary and diagnostic programs. The public diagnostic implementation with the final 1417 structure-control pairs is retained. The obsolete alternative and the older homopolymer figure copy are removed.
- Import-path wrappers and the unused attention backup are removed. Package-qualified imports allow both model families in one process.
- The 17 scheduler configurations are under `configs/slurm/{bcdb,homopolymer,md}`. Their arguments, seeds and job-array settings are retained. Python/torchrun commands invoke the relocated modules.
- `pyproject.toml` supports package installation. README, reproduction commands and the source map explain the workflow. Release maintenance remains under `scripts/`.

Path defaults now use `data/pretraining/pretrain_train.lmdb`, `data/pretraining/pretrain_val.lmdb`, and `checkpoints/no_pretrain.pt` in place of machine-specific absolute paths. These optional inputs are selected through the existing arguments. The BCDB fingerprint CSV default points to the released `data/raw/bcdb/bcdb.csv`. The published MD commands retain `--no_pretrained`.

Data, result and extracted Zenodo asset locations are unchanged. Model parameter names, tensor shapes, task definitions, target transforms and numerical training settings are preserved.

## Completed validation

Checks ran on CPU with Python 3.9.20 and PyTorch 2.5.0, against an unchanged checkout of the reference revision.

| Check | Result |
| --- | --- |
| Existing regression tests before migration | 8 passed |
| Regression and package integration tests after migration | 10 passed |
| Forward/backward and strict checkpoint loading | 18 small-model configurations passed, with bitwise-identical predictions, gradients and state tensors |
| Small-input BCDB, homopolymer and MD preprocessing, target preparation, checkpoint export and retained-output reproduction | 76 artifacts matched exactly in values, or in CSV/JSON bytes |
| Canonical implementation comparison | 33 modules have equivalent computational syntax after accounting for imports, entry points and path relocation |
| Command-line entry points | All 23 module `--help` commands passed |
| Scheduler configurations | All 17 passed shell syntax checks and differ only in entry-point paths |
| Retained datasets and results | All 1894 non-Markdown files are byte-identical to the reference |
| Release verification | 1640 mapped LAMMPS files passed checksum checks, 28 MD model/target OOF tables retain 1640 rows each, and retained pair tables contain 1417 and 29027 pairs |
| Package build and installation | Wheel built and installed in an isolated directory, with both workflows imported outside the checkout |

The 18 model checks cover linear finetune, full chain and three Stage2-Only representations, and MD finetune, full graph and two chain-only representations, each with anchor symmetry breaking enabled and disabled. The linear fixtures exercise chains above the token limit. Strict loading uses state dictionaries produced by the reference implementation.

The artifact comparisons include five-record preprocessing fixtures for all three datasets, all six homopolymer target caches, BCDB and fingerprint-BCDB synthetic-checkpoint inference, MD density/Rg inference across five folds, the final structure-pair metrics and homopolymer figure tables from retained predictions. These are migration checks. No new training, simulations or GPU jobs were run.

Two BCDB checkpoint-comparison adapters were reviewed separately: they now launch the same exporter with `sys.executable -m frpn.linear.export`. The MD diagnostic's generated t-SNE job likewise invokes its relocated module. Numerical computation in these analysis workflows is retained.

The release file manifest can be checked with `python scripts/file_manifest.py`. Dataset and optional Zenodo checks remain available through `python scripts/verify_release.py`.
