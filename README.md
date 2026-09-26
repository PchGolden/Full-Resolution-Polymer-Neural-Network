# Uni-Macro-FRPN

Code and data for **Uni-Macro-FRPN: Full-Resolution and Cross-Scale Learning for Polymers**, covering BCDB classification, homopolymer regression and the 1640-record MD benchmark.

```text
datasets/       Raw benchmarks, fixed folds and LAMMPS inputs
src/frpn/       BCDB/homopolymer and MD models, preprocessing and training
reproduce/      Evaluation, configurations, retained results, tests and guides
```

## Install

Run commands from this repository root with Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduce/configs/requirements.txt
python -m pip install --no-deps -e .
```

For a Conda CPU environment, use `reproduce/configs/environment.yml`. Training requires CUDA-enabled PyTorch and a GPU host.

## Reproduce from included predictions

These commands use the repository's retained outputs and require no checkpoints:

```bash
python -m reproduce.evaluate.structure_pairs \
  --output reproduce/outputs/structure_pairs
python -m reproduce.evaluate.homopolymer_figures \
  --cache-dir reproduce/results/oof_predictions/homopolymer \
  --out_dir reproduce/outputs/homopolymer_figures
```

See the [reproduction guide](reproduce/README.md) for BCDB figures, MD diagnostics and target normalization. Generated files go under `reproduce/outputs/`.

## Run checkpoint inference

Checkpoints and larger caches are prepared for Zenodo. The record is not yet published. The [asset guide](reproduce/docs/checkpoint_download.md) lists the archive layout and extraction steps, using the sibling directory `../zenodo/`.

After extracting the BCDB FRPN checkpoint and processed cache:

```bash
python -m reproduce.evaluate.bcdb_export \
  --checkpoint ../zenodo/checkpoints/bcdb/FRPN/frpn_anchor_withvol_fold0_recover_ep58.pt \
  --pkl_path ../zenodo/data_cache/bcdb/BCDB_chain_withvolume.pkl \
  --fold 0 --output_npz reproduce/outputs/bcdb/fold_0.npz \
  --device cpu --batch_size 32 --num_workers 0
```

[MD inference and other models](reproduce/README.md#checkpoint-inference) use the same released fold definitions and checkpoint metadata.

## Train and verify

Use the [preprocessing and training commands](reproduce/README.md#training-from-data) for BCDB, homopolymers and MD. Model-specific settings and Slurm jobs are in [reproduce/configs](reproduce/configs/README.md).

```bash
python -m unittest discover -s reproduce/tests -t . -v
python reproduce/tools/verify_release.py --lammps-hashes
python reproduce/tools/file_manifest.py
```

[Source map](reproduce/docs/architecture.md) · [Data description](datasets/README.md) · [Code and data availability](reproduce/docs/code_data_availability.md)
