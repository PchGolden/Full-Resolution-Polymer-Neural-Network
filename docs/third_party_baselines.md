# Third-party baselines

The manuscript compares against PerioGT, TransPolymer and SA-GCN/Kimmig-style baselines. Their retained predictions are included in the BCDB OOF cache, and their staged checkpoints are identified separately in `checkpoint_manifest.csv`. FRPN training commands do not train these external architectures.

Original baseline run settings are retained in `configs/slurm/bcdb/`. The upstream implementations and their environments remain separate dependencies for retraining those baselines. This repository does not assign new licenses to third-party software or checkpoints; the original terms continue to apply.
