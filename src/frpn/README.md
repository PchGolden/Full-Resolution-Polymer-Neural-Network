# FRPN source

- `bcdb/`: BCDB classification and linear homopolymer regression. `preprocess`, `prepare_targets` and `train` are the main executable modules.
- `md/`: explicit-graph MD preprocessing, training, target transforms and fold preparation.
- `common/`: shared encoders, layers, molecular features and monomer forward operations.
- `training/`: shared arguments, runtime setup, normalization, checkpoints and monomer pretraining.

Each model family owns its model construction, batching, supervised epoch routines and target transformations. BCDB and homopolymers construct chains from monomer identities and repeat counts. Counts above 1536 repeat tokens are rescaled proportionally before integer allocation and expansion. MD uses the explicit graph supplied in each record.

Use `python -m frpn.bcdb.train --help` or `python -m frpn.md.train --help` for training. Standalone checkpoint inference and analysis live in `reproduce.evaluate`, separate from training-time validation.

The [source map](../../reproduce/docs/architecture.md) explains dependencies. The [reproduction guide](../../reproduce/README.md) provides end-to-end commands.
