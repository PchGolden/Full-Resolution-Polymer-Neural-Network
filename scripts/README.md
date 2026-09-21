# Release maintenance

- `python scripts/verify_release.py`: verify dataset mappings, OOF coverage and matched-pair counts. Add `--lammps-hashes` for all atomistic-input checksums. Zenodo verification options are described in the main README.
- `python scripts/file_manifest.py`: verify every tracked release file against `docs/github_file_manifest.csv`.
- `python scripts/file_manifest.py --write`: refresh that manifest after staging an intentional release change.

Training, preprocessing, export and analysis are directly executable modules under `frpn/`. Scheduler configurations are under `configs/slurm/`. See [reproduction commands](../docs/reproduce_paper.md).
