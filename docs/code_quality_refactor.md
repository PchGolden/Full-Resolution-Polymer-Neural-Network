# Model and training refactor

The refactor consolidates repeated model primitives and separates training configuration, target processing and epoch execution. The comparison baseline is the reorganized release, commit `0a59b0414d0d3c0aa071bb3417c35e0d5755ac2e`.

## Model implementation

- `frpn.common` owns the shared token/pair Transformer, linear and pair-update layers, geometry kernels, attention masks and reconstruction heads. The unused duplicate `linear/models/pair_layers.py` has been removed.
- `common.modeling` supplies monomer encoding and atom reconstruction. Model constructors retain parameter ownership and initialization order, so existing state dictionaries load with the same keys and shapes.
- Each model has a single chain-prediction path. The MD model's unreachable duplicate `chain_only` branch and repeated attention-mask calculations have been removed.
- Chain construction stays family-specific: repeat-count expansion and proportional allocation for linear polymers, explicit supplied graph topology for MD. Fingerprint inputs, symmetry-breaking options, capacity controls and ablations remain available.

## Training implementation

- `linear.config` and `md.config` define workflow arguments using shared groups. CLI flags, defaults and choices are retained.
- `linear.engine` and `md.engine` implement supervised epoch updates and evaluation. Their `targets` modules retain the different label selection, transforms and inverse transforms.
- `frpn.training` provides common runtime setup, gradient update ordering, train-only normalization fitting, checkpoint serialization and masked reconstruction pretraining.
- The supervised schedulers, accumulation behavior, stopping rules, saved metrics, output paths and checkpoint fields remain compatible. The legacy pretraining resume filename `checkpoint_randomrandom.pt` is retained.
- Obsolete commented implementations and unreachable initialization alternatives have been removed. Training-process settings are applied at command startup instead of module import.

## Targeted fixes

1. **MD global-feature masking:** a tensor-valued batch mask was summed across all samples when deciding whether one sample had a valid global token. The decision now uses that sample's row, so another sample's available features cannot expose its missing token.
2. **Single-process MD pretraining:** an early-stopping broadcast ran without an initialized distributed process group. The shared loop broadcasts only during initialized distributed training. A CPU regression test exercises the complete single-process checkpoint-saving path.

These fixes change the two faulty boundary cases. Complete-feature model cases retain their numerical behavior.

## Validation

The one-time baseline comparison used fixed seeds and tiny CPU fixtures:

| Check | Coverage | Result |
| --- | --- | --- |
| Model comparison | 19 linear and 16 MD configurations, including branch modes, fingerprints, geometry, pair updates, anchors, capacity controls, pretraining and train-mode stochastic layers | Initial states, loaded states, predictions, gradients and one AdamW update matched exactly |
| Training and argument comparison | 26 cases covering CLI contracts, target normalization, supervised accumulation/evaluation, optimizer parameter order and pretraining evaluation | Exact match |
| Complete trainer loops | 20 cases covering classification/regression, both supervised schedulers, scripted validation scores for checkpoint selection and stopping, pretraining accumulation and checkpoint resume | Saved tensors, optimizer/scheduler states, metrics and output choices matched exactly |
| Regression suite | 16 tests, including missing global features, tied embeddings, checkpoint initialization, train-only normalization and padded reconstruction targets | Passed |
| Command integration | 23 module help commands and 17 Slurm syntax checks | Passed |
| Checkpoint export | BCDB, BCDB fingerprints and MD inference through their public commands | 15 output artifacts matched the baseline exactly |
| Release assets | 1898 tracked data/result files and 1640 LAMMPS checksums | Unchanged files and valid checksums |

The complete-loop comparison redirects the baseline's CUDA transfers to CPU. It stubs the baseline MD broadcast bug solely to compare update arithmetic, while the new single-process test checks that no broadcast is called. These checks exercise code paths and state transitions, without running a new benchmark training experiment. Multi-GPU execution and mixed-precision GPU numerics were not exercised.

Run the maintained regression suite with:

```bash
python -m unittest discover -s tests -v
```

See [the source map](architecture.md) for module responsibilities and [the reproduction guide](reproduce_paper.md) for training and export commands.
