# Release maintenance

Run these commands from the repository root:

- `python reproduce/tools/verify_release.py`: verify dataset mappings, OOF coverage and matched-pair counts. Add `--lammps-hashes` to check all atomistic-input hashes. See the [asset guide](../docs/checkpoint_download.md) for Zenodo verification.
- `python reproduce/tools/file_manifest.py`: verify tracked release files against `reproduce/docs/github_file_manifest.csv`.
- `python reproduce/tools/file_manifest.py --write`: refresh that manifest after staging an intentional release change.

Training and preprocessing are modules in `frpn.bcdb` and `frpn.md`. Standalone inference and analysis are modules in `reproduce.evaluate`. See the [reproduction guide](../README.md).
