# Source map and execution flow

## Workflow ownership

| Responsibility | BCDB / homopolymer | MD benchmark |
| --- | --- | --- |
| CSV to graph cache | `frpn.linear.preprocess` | `frpn.md.preprocess` |
| Dataset and batch collation | `frpn.linear.data` | `frpn.md.data` |
| Model implementation | `frpn.linear.models.model` | `frpn.md.models.model` |
| Training / task selection | `frpn.linear.train` | `frpn.md.train` |
| CLI arguments | `frpn.linear.config` | `frpn.md.config` |
| Epoch updates and evaluation | `frpn.linear.engine` | `frpn.md.engine` |
| Target transforms and selection | `frpn.linear.targets` | `frpn.md.targets` |
| Held-out checkpoint inference | `frpn.linear.export` (BCDB), `frpn.analysis.homopolymer_figures` (homopolymer) | `frpn.md.export` |
| Model-specific run settings | `configs/slurm/bcdb/`, `configs/slurm/homopolymer/` | `configs/slurm/md/` |

The linear workflow expands monomer repeat counts into chain tokens, with proportional rescaling above the token budget. The MD workflow consumes `chain_node_seg_id`, `chain_node_types` and `chain_edges`. Each family owns its model constructor, token features and collator. Both use the token/pair Transformer in `frpn.common.encoder`, attention and layer primitives in `common`, and molecular graph construction in `frpn.common.molecular_features`.

`train` and `export` import their own family's `data` and `models` modules. `common.modeling` implements the shared monomer forward and masked reconstruction using modules owned by each model, preserving parameter names and initialization order. Model constructors and chain prediction retain family-specific inputs, anchors and ablations. The family `models.encoder` modules re-export the shared encoder for existing imports.

`train` selects supervised or monomer pretraining, prepares loaders/output paths and selects checkpoints. `engine` implements the family's supervised update and evaluation protocol. `targets` retains its target selection and forward/inverse transforms. `frpn.training` supplies common argument groups, runtime setup, checkpoint serialization, train-only normalization fitting, and masked-atom/geometry pretraining. Runtime settings are applied when training starts, so importing model or inference code does not configure multiprocessing or suppress process-wide warnings.

The supervised schedulers remain distinct: linear training uses cosine annealing, while MD uses cosine annealing with warm restarts. Supervised training updates on a final partial accumulation group. Monomer pretraining keeps its existing optimizer-step validation cadence and carries incomplete accumulation groups across epochs. Imports are package-qualified, so the two families coexist without modifying `sys.path`.

## Reproduction flow

1. Read the CSVs and fixed folds under `data/`.
2. Preprocess with `python -m frpn.linear.preprocess` or `python -m frpn.md.preprocess`, or use the released Zenodo processed caches. Homopolymer caches are split by target with `frpn.linear.prepare_targets`.
3. Train a model with the corresponding `train` module and its recorded configuration, or use a released checkpoint.
4. Export validation-fold predictions with the corresponding inference module.
5. Produce figures and diagnostics with `frpn.analysis`. Retained OOF outputs under `results/` enter directly at this step.

All commands in the reproduction guide run from the repository root. `pip install --no-deps -e .` also makes the modules importable outside the checkout, while relative data/output arguments remain relative to the working directory. Data, result filenames, checkpoint keys and the Zenodo extraction layout are unchanged.

## Analysis and auxiliary tools

| Module (`python -m …`) | Purpose |
| --- | --- |
| `frpn.analysis.bcdb_figures` | BCDB figures from retained metadata and prediction/embedding caches |
| `frpn.analysis.homopolymer_figures` | Homopolymer figures from retained caches or checkpoint inference |
| `frpn.analysis.structure_pairs` | Reproduce the final 1417 structure-control pairs and their metrics |
| `frpn.analysis.md_diagnostics` | Matched pairs, additive-null comparisons and residual stacking |
| `frpn.analysis.md_tsne` | Recompute MD t-SNE coordinates and prediction-colored panels |
| `frpn.analysis.md_tsne_plot` | Plot existing t-SNE coordinates |
| `frpn.analysis.md_summary` | Summarize MD training logs |
| `frpn.analysis.homopolymer_summary` | Summarize homopolymer training logs |
| `frpn.linear.compare_checkpoints` | Export and compare BCDB model checkpoints across folds |
| `frpn.linear.compare_tiny` | Summarize the BCDB capacity-control checkpoints |
| `frpn.linear.pretraining.train`, `.evaluate`, `.analyze` | Auxiliary BCDB unsupervised workflow |
| `frpn.md.normalize_targets` | Training-fold target normalization and saved parameters |
| `frpn.md.make_folds`, `frpn.md.rebuild_dataset` | Dataset preparation utilities. Published reproduction uses the supplied fixed folds |

Auxiliary monomer pretraining accepts `--pretrain_train_path` and `--pretrain_val_path`; these LMDB inputs must be supplied separately. Initialization checkpoints are selected with `--weight_path`. Published MD commands explicitly select scratch initialization with `--no_pretrained`. BCDB fingerprint input defaults to `data/raw/bcdb/bcdb.csv`.

## Entry-point migration

| Earlier entry point | Current entry point |
| --- | --- |
| `frpn.cli.train_bcdb`, `frpn.cli.train_homopolymer` | `frpn.linear.train` |
| `frpn.cli.train_md` | `frpn.md.train` |
| `frpn.cli.export_oof bcdb` | `frpn.linear.export` |
| `frpn.cli.export_oof md` | `frpn.md.export` |
| `scripts/preprocess/preprocess_bcdb_polymer.py` | `frpn.linear.preprocess` |
| `scripts/preprocess/preprocess_md_polymer.py` | `frpn.md.preprocess` |
| `scripts/figures/*`, `scripts/evaluate/reproduce_structure_pairs.py` | The named `frpn.analysis` modules above |
| `scripts/configs/` | `configs/slurm/` |

Use `python -m MODULE` and `torchrun … -m MODULE`, followed by the existing arguments. The old `cli`/`pipelines` wrappers, duplicate analysis files and unused attention backup have been removed. The canonical MD diagnostic is the former public figure entry point, which already uses the final 1417-pair definition. The canonical homopolymer figure module retains support for the released prediction caches and packaged checkpoints.
