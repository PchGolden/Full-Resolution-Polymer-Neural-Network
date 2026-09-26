# Environments and run configurations

`requirements.txt` supplies the Python dependencies. `environment.yml` supplies a Conda CPU environment for preprocessing, evaluation and diagnostics. Use CUDA-enabled PyTorch on a GPU host for training.

`slurm/bcdb/`, `slurm/homopolymer/` and `slurm/md/` contain manuscript run configurations, including model-specific hyperparameters, seeds, target selection and folds. The two BCDB PerioGT jobs use the separate implementation described in [third-party baselines](../docs/third_party_baselines.md).

Use the [portable commands](../README.md) for a single run. For Slurm, adapt the partition, GPU resources and Conda environment to your cluster, then run from the repository root:

```bash
mkdir -p reproduce/outputs/logs/slurm/BCDB_frpn_tiny_withvol_40ep
export FRPN_ROOT="$PWD"
export FRPN_ENV=frpn
sbatch reproduce/configs/slurm/bcdb/BCDB_frpn_tiny_withvol_5fold_40ep_array.job
```

For other jobs, create the parent directories in their `#SBATCH --output` and `--error` directives before submission. Slurm opens these files before the script starts.

Jobs use the same `frpn.bcdb`, `frpn.md` and `reproduce.evaluate` modules as the portable commands. Generated training results and logs are written under `reproduce/outputs/`. Match each model's recorded settings when reproducing a run.
