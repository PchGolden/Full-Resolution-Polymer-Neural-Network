# Final LAMMPS files for 1640 MD records

Each ordinary `.data` file contains the atomistic system, masses, charges and embedded force-field coefficients used to initialize its simulation. The directory names retain the three archived workflow domains.

`dataset_record_mapping.csv` maps each one-based dataset record and zero-based row of `../raw/md_final1640_v2/final_1640dataset.csv` to a file here. `manifest.csv` adds temperature, topology, size, SHA-256 and charge summaries. `audit_summary.json` and `validation_summary.json` record the completed parameter/charge and coefficient-coverage audits. `conflict_resolution_records/` contains the retained ambiguity-resolution records.

Verify all file hashes from the repository root with:

```bash
python reproduce/tools/verify_release.py --lammps-hashes
```
