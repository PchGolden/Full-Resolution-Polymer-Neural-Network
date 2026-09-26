"""Check released paths, MD row mappings and optional file hashes (no training)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zenodo', type=Path, help='Optional local Zenodo asset directory')
    parser.add_argument('--lammps-hashes', action='store_true')
    parser.add_argument('--archive-hashes', action='store_true', help='Hash packed checkpoint archives when checking an unextracted Zenodo bundle')
    parser.add_argument('--checkpoint-hashes', action='store_true', help='Read every checkpoint; can take substantial time')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw = rows(root / 'datasets/raw/md_final1640_v2/final_1640dataset.csv')
    folds = rows(root / 'datasets/splits/md_final1640_v2/with_fold.csv')
    assert len(raw) == len(folds) == 1640
    assert {int(r['fold']) for r in folds} == set(range(5))
    mapping = rows(root / 'datasets/lammps/dataset_record_mapping.csv')
    assert {int(r['dataset_row']) for r in mapping} == set(range(1640))
    for row in mapping:
        path = root / 'datasets/lammps' / row['lammps_file']
        assert path.is_file(), path
        if args.lammps_hashes:
            assert sha256(path) == row['sha256'], path
    oof = list((root / 'reproduce/results/oof_predictions/md_final1640_v2').glob('oof_*.csv'))
    assert len(oof) == 28, len(oof)
    for path in oof:
        assert len(rows(path)) == 1640, path
    assert len(rows(root / 'reproduce/results/diagnostics/md/structure_variation_pairs.csv')) == 1417
    assert len(rows(root / 'reproduce/results/diagnostics/md/chemistry_control_pairs.csv')) == 29027
    manifest = rows(root / 'reproduce/docs/checkpoint_manifest.csv')
    checkpoint_paths_checked = False
    archive_paths_checked = False
    if args.zenodo:
        if (args.zenodo / 'checkpoints').is_dir():
            for row in manifest:
                path = args.zenodo / 'checkpoints' / row['release_path']
                assert path.is_file(), path
                assert path.stat().st_size == int(row['size_bytes']), path
                if args.checkpoint_hashes:
                    assert sha256(path) == row['sha256'], path
            checkpoint_paths_checked = True
        else:
            candidates = [args.zenodo / 'manifests/checkpoint_archives.csv', args.zenodo / 'uploads/checkpoint_archives.csv']
            archive_manifest = next((path for path in candidates if path.is_file()), None)
            assert archive_manifest, 'Extract the checkpoints or include checkpoint_archives.csv with the packed bundle'
            assert not args.checkpoint_hashes, '--checkpoint-hashes requires extracted checkpoints; use --archive-hashes for packed archives'
            for row in rows(archive_manifest):
                path = args.zenodo / row['archive_path']
                assert path.is_file(), path
                assert path.stat().st_size == int(row['size_bytes']), path
                if args.archive_hashes:
                    assert sha256(path) == row['sha256'], path
            archive_paths_checked = True
    print(json.dumps(dict(md_records=1640, mapped_lammps_files=len(mapping), oof_model_targets=len(oof), structure_pairs=1417, chemistry_pairs=29027, checkpoint_entries=len(manifest), lammps_hashes_checked=args.lammps_hashes, checkpoint_paths_checked=checkpoint_paths_checked, archive_paths_checked=archive_paths_checked, checkpoint_hashes_checked=args.checkpoint_hashes, archive_hashes_checked=args.archive_hashes), indent=2))

if __name__ == '__main__':
    main()
