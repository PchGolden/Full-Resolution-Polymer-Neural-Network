# Retained results and source tables

`oof_predictions/` supplies BCDB's compact six-model OOF predictions and labels, 18 homopolymer model-target caches, and 28 MD pooled model-target CSVs plus per-fold exports. `summaries/` contains the manuscript benchmark summaries. Larger embeddings and processed caches are in the Zenodo bundle.

`diagnostics/md/` uses the final manuscript definitions:

| Manuscript item | Source |
| --- | --- |
| Table S17 structure control | `structure_variation_delta_metrics_by_label.csv` (1417 pairs) |
| Table S18 chemistry control | `chemistry_control_delta_metrics_by_label.csv` (29027 pairs) |
| Table S19 calibrated additive-null residuals | `calibrated_residual_metrics_by_label.csv` |
| Table S20 leave-one-model-out residual stacking | `leave_one_out_stack_metrics_by_label.csv` |

The paired row IDs themselves are supplied alongside these metrics. `figure_source_tables/md_final1640_v2/` provides compact plot inputs. The structure-control source replaces the earlier topology-only matching table. No new model fitting is performed by copying or reading these retained experimental outputs.
