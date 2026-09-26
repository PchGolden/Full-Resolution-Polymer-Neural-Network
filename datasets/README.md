# Benchmark inputs

- `raw/bcdb/`: BCDB tables, labels and monomer-name dictionaries.
- `raw/homopolymer/homopolymer.csv`: 372-record linear homopolymer benchmark.
- `raw/md_final1640_v2/final_1640dataset.csv`: 1640-record MD benchmark in the released final row order.
- `splits/`: retained five-fold assignments, including MD's complete `with_fold.csv` and one-based dataset-record mapping.
- `lammps/`: one final atomistic data file per MD record; each contains atom types, charges, masses and the nonbonded/bonded coefficients used for that record.

The MD CSV specifies chemistry with `SMILES0`–`SMILES7` and monomer counts with `seg0_feat0`–`seg7_feat0`. `chain_node_seg_id`, `chain_node_types` and `chain_edges` define the model graph; `topology` and `mix_mode` record topology and sequence ordering. `glob_feat0` records temperature. The public preprocessor reads these graph fields directly.

MD labels are density, Rg, D, S_q_peak, nematic_order, dielectric_constant and refractive_index. `src/frpn/md/normalize_targets.py` applies log10(max(D, 1e-16)) and then training-fold standardization. It also implements the inverse transform. The corresponding transformations are integrated into the training and checkpoint-export pipelines.

Raw graph specifications, final atomistic files and labels are provided. The general force-field assignment system and complete trajectory-production service are developed separately; the target-processing script operates on the provided labels.

For LAMMPS, `dataset_record_mapping.csv` uses one-based record numbers and zero-based CSV row indices. File paths are relative to `datasets/lammps/`; SHA-256 hashes identify the exact simulation inputs. No released MD record is designated as a failed simulation.
