# Code and data availability

The GitHub package contains source code for preprocessing, FRPN model implementation, training, evaluation and predictive-behavior diagnostics; the BCDB, homopolymer and MD datasets with their split identifiers; target-processing scripts; retained OOF predictions; and final LAMMPS data files mapped to all 1,640 MD records.

Model checkpoints for inference and weight initialization, processed caches, and larger embedding/figure caches are prepared for Zenodo. The Zenodo record has not yet been published; its DOI and download links will be added to the repository after the author's deposit. `checkpoint_manifest.csv` provides the per-model, target and fold file mapping with checksums.

The generalized force-field assignment workflow is being developed for a separate release. This package supplies final atomistic inputs and the transformations of the provided labels.
