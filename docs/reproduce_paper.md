# Reproducing the manuscript

Run commands from the GitHub repository root. Use the included folds and the same model settings when comparing results. The retained OOF predictions allow table/figure reproduction without training. Checkpoint export and model training are separate executable routes.

## 1. Prepare the three datasets

BCDB's target is `is_target`; the CSV already includes its five-fold identifiers.

```bash
mkdir -p data/processed
python -m frpn.linear.preprocess \
  --task finetune --csv data/raw/bcdb/bcdb_2SMILES.csv \
  --labels is_target --output data/processed/BCDB_chain_withvolume.pkl \
  --kfold 5 --seed 42 --volume-mode with --export-splits
```

For the BCDB ablation without volume fraction, use `--volume-mode without` and a separate output filename.

```bash
python -m frpn.linear.preprocess \
  --task finetune --csv data/raw/homopolymer/homopolymer.csv \
  --labels density,Rg,self-diffusion,Cp,dielectric_const_dc,refractive_index \
  --outroot data/processed/HOMO_372 --dataset-name homopolymer \
  --kfold 5 --seed 42 --volume-mode with --export-splits
python -m frpn.linear.prepare_targets \
  --input data/processed/HOMO_372/homopolymer/main/split.pkl \
  --output-dir data/processed/HOMO_372/homopolymer/main
```

MD consumes the released `with_fold.csv` directly, preserving the final row-to-fold mapping.

```bash
python -m frpn.md.preprocess \
  --task finetune --csv data/splits/md_final1640_v2/with_fold.csv \
  --outroot data/processed/MD --dataset-name MD_FINAL1640_V2 \
  --labels density,Rg,D,S_q_peak,nematic_order,dielectric_constant,refractive_index \
  --workers 4 --kfold 5 --seed 42 --fold-source column --export-splits
```

Existing processed caches are also supplied in Zenodo's `data_cache/`: BCDB PKLs are under `bcdb/`; homopolymer and MD PKLs under `<dataset>/main/`. Pass those paths directly through `--pkl_path` to avoid preprocessing.

## 2. Train from data

These are GPU commands; they are not part of the quick verification. Repeat fold indices 0–4. BCDB's original FRPN run used four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m frpn.linear.train \
  --main_task chain --fold 0 --dataset_name BCDB \
  --pkl_path data/processed/BCDB_chain_withvolume.pkl \
  --epochs 100 --lr 1e-4 --weight_decay 1e-2 \
  --task_type cls --batch_size 8 --num_tasks 2 --wo_triopm --amp \
  --seed 42 --max_chain_tokens 1536 \
  --weight_path ../zenodo/checkpoints/pretrained/checkpoint_adaptive.pt \
  --results_root results/retrained_bcdb --distributed
```

BCDB Uni-Macro uses `--main_task finetune`. Stage2-Only uses `--main_task stage2_only --stage2_only_repr smiles_only`; FRPN-FP uses `--main_task stage2_only --stage2_only_repr fp_morgan2048 --fp_csv data/raw/bcdb/bcdb.csv`. Consult the corresponding run configuration for each model's epochs, learning rate and dropout instead of assuming identical hyperparameters across models.

The FRPN-tiny capacity control uses the dimensions and 40-epoch schedule in `configs/slurm/bcdb/BCDB_frpn_tiny_withvol_5fold_40ep_array.job`; its retained five-fold predictions and metrics are in `results/capacity/bcdb_frpn_tiny/`.

Homopolymer example, FRPN/density:

```bash
python -m frpn.linear.train \
  --dataset_name HOMO_372/frpn/label0_density --main_task chain \
  --task_type reg --num_tasks 1 --fold 0 \
  --pkl_path data/processed/HOMO_372/homopolymer/main/split_density.pkl \
  --results_root results/retrained_homopolymer \
  --epochs 100 --batch_size 24 --lr 1e-4 --weight_decay 1e-4 \
  --dropout 0.1 --amp --seed 42 --early_stop_patience 40 \
  --label_zscore --log10_labels self-diffusion --log10_clamp_min 1e-16 \
  --wo_triopm --max_chain_tokens 1536 --stage2_only_repr full \
  --save_finetune_checkpoint
```

The single-target cache aliases are `density`, `rg`, `self_diffusion`, `cp`, `dielectric_const` and `refractive_index`. Change the target cache and dataset output name for each property. Uni-Macro and Stage2-Only use the corresponding `main_task` options above. The homopolymer run starts from scratch when no `--weight_path` is supplied.

MD example, FRPN/Rg:

```bash
python -m frpn.md.train \
  --dataset_name MD_FINAL1640_V2/FRPN/label1_Rg --fold 0 \
  --pkl_path data/processed/MD/MD_FINAL1640_V2/main/split.pkl \
  --results_root results/retrained_md --main_task chain \
  --task_type reg --num_tasks 1 --label_index 1 \
  --epochs 100 --batch_size 24 --lr 1e-4 --weight_decay 1e-4 \
  --dropout 0.1 --amp --seed 42 --early_stop_patience 40 \
  --early_stop_start_epoch 10 --wo_triopm --chain_only_repr smiles_embed \
  --log10_labels D --d_clamp_min 1e-16 --no_pretrained
```

MD target indices are density=0, Rg=1, D=2, S_q_peak=3, nematic_order=4, dielectric_constant=5 and refractive_index=6. Uni-Macro uses `--main_task finetune --finetune_optimizer_stage1_only`; Stage2-Only uses `--main_task chain_only --chain_only_repr smiles_embed`; FRPN-FP uses `--main_task chain_only --chain_only_repr fp_morgan2048`. All seven targets use the numerical training settings in the example; each is trained as a separate single-output model.

The original scheduler jobs under `configs/slurm/` document the run settings. Their cluster account/partition/environment paths require local adaptation; the portable commands above are the primary entry points.

## 3. Export predictions from released checkpoints

The checkpoint manifest maps exact model, target and fold identities to the files. These commands actually load a checkpoint and run held-out inference.

BCDB FRPN, fold 0:

```bash
python -m frpn.linear.export \
  --checkpoint ../zenodo/checkpoints/bcdb/FRPN/frpn_anchor_withvol_fold0_recover_ep58.pt \
  --pkl_path ../zenodo/data_cache/bcdb/BCDB_chain_withvolume.pkl \
  --fold 0 --output_npz reproduced/bcdb/fold_0.npz \
  --device cpu --batch_size 32 --num_workers 0
```

For BCDB FRPN-FP export, `--fp_csv` defaults to the released monomer CSV; an explicit path can be supplied for a different dataset.

MD FRPN, all seven targets and all five folds:

```bash
python -m frpn.md.export \
  --raw_csv data/raw/md_final1640_v2/final_1640dataset.csv \
  --pkl_path ../zenodo/data_cache/md_final1640_v2/main/split.pkl \
  --results_root ../zenodo/checkpoints/md_final1640_v2 \
  --labels density,Rg,D,S_q_peak,nematic_order,dielectric_constant,refractive_index \
  --models FRPN --out_dir reproduced/md_oof --cpu --num_workers 0
```

The MD exporter uses the full target ordering to match each checkpoint's `label_index`; the example exports all seven targets. To export all four models, set `--models Uni_Macro,Stage2_Only,FRPN_FP,FRPN`. Omit `--cpu` on a CUDA host. Target means, standard deviations and log transforms are read from each checkpoint.

## 4. Reproduce from the retained outputs

The retained MD predictions are in `results/oof_predictions/md_final1640_v2/`; BCDB's compact six-model predictions and aligned metadata are in `results/oof_predictions/bcdb/`; the combined prediction/embedding cache used by the figure script is in Zenodo's `figures_source_cache/bcdb/`; homopolymer per-target caches are in `results/oof_predictions/homopolymer/`.

```bash
python -m frpn.analysis.bcdb_figures \
  --meta_csv results/oof_predictions/bcdb/sample_meta.csv \
  --cache6_npz ../zenodo/figures_source_cache/bcdb/cache_6models_oof_5fold_v1.npz \
  --out_dir reproduced/bcdb_figures
python -m frpn.analysis.homopolymer_figures \
  --cache-dir results/oof_predictions/homopolymer \
  --out_dir reproduced/homopolymer_figures
python -m frpn.analysis.structure_pairs \
  --output reproduced/structure_pairs
```

The structure-pair script reproduces the final 1,417 pairs: equal temperature and exact chain length, composition overlap at least 0.95, and a difference in topology, ordering, or both. `results/diagnostics/md/` also contains the 29,027 chemistry-control pairs and the saved additive-null and leave-one-model-out residual-stacking tables.

The MD diagnostic entry point regenerates matched pairs and cross-fitted residual comparisons and writes a separate t-SNE launcher. It uses the final structure-pair definition:

```bash
python -m frpn.analysis.md_diagnostics \
  --raw_csv data/raw/md_final1640_v2/final_1640dataset.csv \
  --with_fold_csv data/splits/md_final1640_v2/with_fold.csv \
  --preds_dir results/oof_predictions/md_final1640_v2 \
  --embeds_dir ../zenodo/figures_source_cache/md_oof_embeddings \
  --out_dir reproduced/md_diagnostics --chem_k 6 --ridge_alpha 1.0 \
  --topology_overlap 0.95 --chemistry_overlap 0.60 \
  --tsne_perplexity 50 --tsne_seed 42
```

For a direct CPU t-SNE run, for example FRPN/density:

```bash
python -m frpn.analysis.md_tsne \
  --raw_csv data/raw/md_final1640_v2/final_1640dataset.csv \
  --embeds_dir ../zenodo/figures_source_cache/md_oof_embeddings \
  --preds_dir results/oof_predictions/md_final1640_v2 \
  --model FRPN --label density --out_dir reproduced/md_tsne \
  --value y_pred_raw --cmap blue_navy --perplexity 50 --max_iter 2000 --seed 42
```

Repeat the model and label selections to recreate the other panels. Existing t-SNE coordinates are available under Zenodo's `figures_source_cache/md_diagnostics/embedding_tsne_pred_colored/`. The script's legacy filenames containing `topology_control` identify the current structure-control panel; they are not SI table numbers. The manuscript's current diagnostic tables are S17–S20, with source columns recorded in `results/README.md`.
