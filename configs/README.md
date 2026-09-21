# Run configurations

`slurm/bcdb/`, `slurm/homopolymer/` and `slurm/md/` contain the manuscript run configurations. Each retains its original hyperparameters, target selection, seeds, folds and job-array layout. The two BCDB PerioGT jobs run the separately obtained third-party implementation described in [third_party_baselines.md](../docs/third_party_baselines.md).

Use the portable commands in [reproduce_paper.md](../docs/reproduce_paper.md) for a single run. To use a Slurm job, adapt its partition, GPU resources and Conda environment to your cluster. From the repository root:

```bash
mkdir -p sbatch/sbatch_log
export FRPN_ROOT="$PWD"
export FRPN_ENV=frpn
sbatch configs/slurm/bcdb/BCDB_frpn_tiny_withvol_5fold_40ep_array.job
```

Jobs invoke the same `frpn.linear`, `frpn.md` and `frpn.analysis` modules as the portable commands. Output directories are defined by each job. Match the model-specific settings when reproducing a run, including the recorded seed rather than assuming every job uses the same seed.
