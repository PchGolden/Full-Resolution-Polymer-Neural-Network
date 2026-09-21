# FRPN Python package

- `linear/`: BCDB classification and linear homopolymer regression. `preprocess.py`, `train.py` and `export.py` are the workflow entry points. `prepare_targets.py` produces the six homopolymer target caches.
- `md/`: explicit-graph MD preprocessing, training, checkpoint export, target transforms and fold preparation.
- `common/`: shared attention and molecular graph primitives.
- `analysis/`: plots, summaries, matched pairs and residual diagnostics. These modules read retained outputs or exported checkpoint predictions.

Run entry points as modules, for example `python -m frpn.linear.train --help` or `python -m frpn.md.export --help`. Both model families can be imported in the same Python process.

The [source map](../docs/architecture.md) explains dependencies and auxiliary utilities. The [reproduction guide](../docs/reproduce_paper.md) provides end-to-end commands.
