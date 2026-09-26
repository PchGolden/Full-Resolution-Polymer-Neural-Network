# Three-directory layout verification

Baseline: `2410d8869b3fe2a19b88dd02c2bf6f8eadf2ed78`.

The release is organized into `datasets/`, `src/` and `reproduce/`. The installed
model package uses `frpn.bcdb` and `frpn.md`; repository evaluation commands use
`reproduce.evaluate`. The [source map](architecture.md) lists the entry-point
migration, and the [reproduction guide](../README.md) gives complete commands.

Changes are limited to file placement, imports, resource/output paths, packaging
and documentation. Generated training results, figures and logs belong under
`reproduce/outputs/`. The MD preprocessing job now reads the released fixed-fold
CSV directly. The BCDB figure default points to the existing Zenodo cache that
contains both predictions and embeddings. External PerioGT code is selected
through `PERIOGT_ROOT`, defaulting to a sibling checkout.

| Check | Result |
| --- | --- |
| Core implementation | 45 Python files have identical executable ASTs after accounting for namespace and path-literal changes |
| Model comparison | 19 BCDB and 16 MD configurations match initialization, strict state loading, predictions, gradients and an AdamW update exactly |
| Checkpoint inference | BCDB, FRPN-FP and MD exporters produce 15 matching artifacts through the new entry points |
| Data and retained results | All 1894 non-Markdown payload files retain their SHA-256 checksums |
| Regression suite | All 16 tests pass after editable installation |
| Package installation | Both model families import outside the repository without setting `PYTHONPATH` |
| CLI and jobs | All 23 module help commands and 17 Slurm syntax checks pass; scheduling resources and array definitions are unchanged |
| Generated launcher | The t-SNE launcher resolves the new module and paths, including paths containing spaces, for all 28 model/target combinations |
| README commands | Structure-control reproduction yields 1417 pairs; homopolymer figures complete with no missing inputs |
| Fingerprint defaults | BCDB fingerprint inference resolves the relocated monomer CSV without an explicit `--fp_csv` |
| Release assets | The MD mappings, retained OOF coverage and all 1640 LAMMPS hashes pass release verification |

Numerical comparisons used the existing small fixtures and synthetic checkpoints
on CPU with Python 3.9.20 and PyTorch 2.5.0. Full training, GPU/AMP/DDP execution
and the complete released checkpoint collection were not rerun. The generated
t-SNE submission script was inspected and exercised with a command recorder;
t-SNE itself was not run.

BCDB and homopolymer chains retain proportional rescaling above the default
1536-repeat-token budget. MD retains its explicit input graph. The allocation
implementation is unchanged, and its eight regression tests are included in the
16-test suite.

Run the maintained checks from the repository root after installation:

```bash
python -m unittest discover -s reproduce/tests -t . -v
python reproduce/tools/verify_release.py --lammps-hashes
python reproduce/tools/file_manifest.py
```
