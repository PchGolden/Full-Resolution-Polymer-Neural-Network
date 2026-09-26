# Source map and execution flow

## Directory responsibilities

The repository has three top-level directories: `datasets/` holds inputs and fixed folds, `src/frpn/` holds model and training code, and `reproduce/` holds evaluation, run configurations, retained results and tests. New artifacts belong in `reproduce/outputs/`; preprocessed caches belong in `datasets/processed/`. Both locations are ignored by Git.

| Responsibility | BCDB / homopolymer | MD benchmark |
| --- | --- | --- |
| CSV to graph cache | `frpn.bcdb.preprocess` | `frpn.md.preprocess` |
| Dataset and collation | `frpn.bcdb.data` | `frpn.md.data` |
| Model implementation | `frpn.bcdb.models.model` | `frpn.md.models.model` |
| Training / task selection | `frpn.bcdb.train` | `frpn.md.train` |
| CLI arguments | `frpn.bcdb.config` | `frpn.md.config` |
| Supervised epoch updates and validation | `frpn.bcdb.engine` | `frpn.md.engine` |
| Target transforms and selection | `frpn.bcdb.targets` | `frpn.md.targets` |
| Checkpoint inference | `reproduce.evaluate.bcdb_export`, `reproduce.evaluate.homopolymer_figures` | `reproduce.evaluate.md_export` |
| Run settings | `reproduce/configs/slurm/bcdb/`, `reproduce/configs/slurm/homopolymer/` | `reproduce/configs/slurm/md/` |

BCDB and homopolymers share a linear-chain implementation. It expands monomer repeat counts into chain tokens, using proportional rescaling when the requested count exceeds 1536 repeat tokens. Four auxiliary tokens are outside that budget. Integer allocation preserves composition as closely as the budget permits. MD consumes `chain_node_seg_id`, `chain_node_types` and `chain_edges` from each record, without this repeat-token cap.

Each family owns its model constructor, token features and collator. Both use `frpn.common` for the token/pair Transformer, attention, layers, molecular graph construction and shared monomer forward operations. Constructors retain ownership and initialization of parameters, including family-specific inputs, anchors and ablations.

`train` prepares loaders, selects supervised or monomer pretraining, and manages checkpoints. `engine` implements supervised updates and validation. `targets` defines target selection and transformations. `frpn.training` provides shared arguments, runtime setup, checkpoint serialization, train-only normalization fitting and masked-atom/geometry pretraining. Runtime settings are applied when training starts.

The supervised schedulers differ: BCDB/homopolymer training uses cosine annealing, and MD uses cosine annealing with warm restarts. Supervised training updates on a final partial accumulation group. Monomer pretraining retains its optimizer-step validation cadence and accumulation across epoch boundaries.

## Reproduction flow

1. Read raw data and fixed folds under `datasets/`.
2. Preprocess with the corresponding `frpn.bcdb` or `frpn.md` module, or use Zenodo's processed caches. `frpn.bcdb.prepare_targets` splits homopolymer caches by target.
3. Train with the recorded model configuration, or select a released checkpoint.
4. Export validation-fold predictions with `reproduce.evaluate`.
5. Produce figures and diagnostics. Retained outputs under `reproduce/results/` enter directly at this step.

Install with `python -m pip install --no-deps -e .` to import both model families outside the checkout. Repository evaluation commands remain under `reproduce/` and run from the repository root, as shown in the [reproduction guide](../README.md). Explicit relative data/output arguments are relative to the current working directory. Zenodo archive names, extracted paths and checkpoint state keys retain their existing format.

## Evaluation and auxiliary commands

| Module (`python -m …`) | Purpose |
| --- | --- |
| `reproduce.evaluate.bcdb_export` | BCDB validation-fold checkpoint inference |
| `reproduce.evaluate.md_export` | MD checkpoint inference across models, targets and folds |
| `reproduce.evaluate.bcdb_figures` | BCDB figures from metadata and prediction/embedding caches |
| `reproduce.evaluate.homopolymer_figures` | Homopolymer figures from retained caches or checkpoint inference |
| `reproduce.evaluate.structure_pairs` | Final 1417 structure-control pairs and metrics |
| `reproduce.evaluate.md_diagnostics` | Matched pairs, additive-null comparisons and residual stacking |
| `reproduce.evaluate.md_tsne`, `reproduce.evaluate.md_tsne_plot` | Compute or plot MD t-SNE panels |
| `reproduce.evaluate.md_summary`, `reproduce.evaluate.homopolymer_summary` | Summarize training logs |
| `reproduce.evaluate.bcdb_compare_checkpoints`, `reproduce.evaluate.bcdb_compare_tiny` | Compare BCDB checkpoints and capacity controls |
| `frpn.bcdb.pretraining.train` | Auxiliary BCDB unsupervised training |
| `reproduce.evaluate.bcdb_pretraining_evaluate`, `reproduce.evaluate.bcdb_pretraining_analyze` | Auxiliary BCDB unsupervised evaluation and analysis |
| `frpn.md.normalize_targets` | Training-fold target normalization and saved parameters |
| `frpn.md.make_folds`, `frpn.md.rebuild_dataset` | Dataset preparation utilities; manuscript reproduction uses supplied folds |

Auxiliary monomer pretraining accepts separately supplied LMDB inputs through `--pretrain_train_path` and `--pretrain_val_path`. Initialization checkpoints use `--weight_path`. Published MD commands select scratch initialization with `--no_pretrained`. BCDB fingerprint input defaults to `datasets/raw/bcdb/bcdb.csv`.

## Migration from commit `2410d88`

| Previous location or module | Current location or module |
| --- | --- |
| `data/` | `datasets/` |
| `frpn/linear/`, `frpn.linear.*` | `src/frpn/bcdb/`, `frpn.bcdb.*` |
| `frpn/{md,common,training}/` | `src/frpn/{md,common,training}/`; import names unchanged |
| `frpn.linear.export`, `frpn.md.export` | `reproduce.evaluate.bcdb_export`, `reproduce.evaluate.md_export` |
| `frpn.analysis.*` | `reproduce.evaluate.*` |
| `frpn.linear.compare_checkpoints`, `frpn.linear.compare_tiny` | `reproduce.evaluate.bcdb_compare_checkpoints`, `reproduce.evaluate.bcdb_compare_tiny` |
| `frpn.linear.pretraining.evaluate`, `frpn.linear.pretraining.analyze` | `reproduce.evaluate.bcdb_pretraining_evaluate`, `reproduce.evaluate.bcdb_pretraining_analyze` |
| `configs/`, `results/`, `tests/`, `docs/` | Corresponding directories under `reproduce/` |
| `scripts/` | `reproduce/tools/` |
| Generated training, figures and logs at the repository root | `reproduce/outputs/` |

Use the current module names with `python -m` or `torchrun … -m`; old commands have no forwarding wrappers. CLI options and numerical settings retain their meanings. The [initial reorganization](history/reorganization.md) and [model/training refactor](history/code_quality_refactor.md) records describe earlier revisions and their validation.

The [layout verification record](layout_validation.md) documents checks for this migration.
