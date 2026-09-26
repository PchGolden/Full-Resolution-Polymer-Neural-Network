# Third-party baselines

The manuscript compares against PerioGT, TransPolymer and SA-GCN/Kimmig-style baselines. Their retained predictions are included in the BCDB OOF cache, and their staged checkpoints are identified separately in `checkpoint_manifest.csv`. FRPN training commands do not train these external architectures.

Original baseline run settings are retained in `reproduce/configs/slurm/bcdb/`. The upstream implementations and their environments are separate dependencies for retraining those baselines. The PerioGT jobs use a sibling checkout at `../FRPN_PerioGT/` by default. Set `PERIOGT_ROOT` to use another checkout location. This repository does not assign new licenses to third-party software or checkpoints; the original terms continue to apply.
