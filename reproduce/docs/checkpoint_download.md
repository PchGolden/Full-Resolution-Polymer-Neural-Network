# Zenodo assets

The Zenodo record is prepared locally and will be published by the author. No DOI or download URL is available yet. Its actual link should be added to the repository README after deposit.

The upload set is **`zenodo/uploads/`**, containing model-group checkpoint archives and two archives for processed/embedding caches. The manifests identify each archive and each extracted checkpoint. Checkpoints retain full-precision model weights, model settings and normalization metadata for inference and weight initialization. Optimizer and learning-rate-scheduler resume states are omitted from the upload copies; the original complete training files remain backed up locally.

Download the needed archive(s) into a sibling directory named `zenodo/uploads/`, preserving their filenames. From the GitHub repository root:

```bash
mkdir -p ../zenodo/uploads
# After downloading or locating the archive files:
tar -xf ../zenodo/uploads/processed_and_embedding_caches.tar -C ../zenodo
tar -xf ../zenodo/uploads/bcdb_prediction_embedding_cache.tar -C ../zenodo
tar -xf ../zenodo/uploads/checkpoints_bcdb_FRPN.tar -C ../zenodo
tar -xf ../zenodo/uploads/checkpoints_md_final1640_v2_FRPN.tar -C ../zenodo
```

This creates `../zenodo/data_cache/`, `../zenodo/figures_source_cache/` and the selected `../zenodo/checkpoints/` subdirectories used in the reproduction commands. Unpack the `pretrained` archive for the BCDB initialization checkpoint or other model archives for their corresponding variants. Exact archive names and SHA-256 values are listed in the archive manifest.

`reproduce/docs/checkpoint_manifest.csv` maps each extracted checkpoint to its dataset, model, target and fold. `release_path` is relative to the extracted `checkpoints/` directory. The archived MD lookup layout is `<model>/<label>/<model>__<label>__foldN.pt`; the exporter recognizes this layout directly.

The checkpoint manifest includes the main BCDB models and capacity controls (including FRPN-tiny), the three homopolymer models, four MD models across seven targets, and the initialization checkpoints. Additional volume/anchor ablation results are retained in `reproduce/results/ablations/bcdb_si_s11/`.

Verify the complete packed or extracted asset set from the repository root:

```bash
python reproduce/tools/verify_release.py --zenodo ../zenodo --lammps-hashes
```

Add `--archive-hashes` to check packed checkpoint TAR hashes, or `--checkpoint-hashes` to check extracted checkpoint hashes. The complete-set verifier expects all manifest entries. For a selected-model download, use the manifest to check the requested files and run its inference command.
