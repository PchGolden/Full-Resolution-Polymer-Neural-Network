# FRPN source package

`pipelines/bcdb/` implements BCDB classification and homopolymer regression. `pipelines/md_final1640_v2/` implements the MD benchmark with explicit input graphs. CLI wrappers place the chosen pipeline on Python's import path.

- `python -m frpn.cli.train_bcdb --help`
- `python -m frpn.cli.train_homopolymer --help`
- `python -m frpn.cli.train_md --help`
- `python -m frpn.cli.export_oof bcdb --help`
- `python -m frpn.cli.export_oof md --help`
- `python -m frpn.cli.make_figures --help`

BCDB/homopolymer linear chains use composition-preserving proportional rescaling at the token budget. MD uses the graph fields supplied by the dataset. See the repository README for runnable commands and `docs/reproduce_paper.md` for model-specific settings.
