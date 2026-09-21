"""Verify or refresh the SHA-256 manifest of Git-tracked release files."""

import argparse
import csv
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/github_file_manifest.csv"


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return sorted({name.decode() for name in result.stdout.split(b"\0") if name} - {MANIFEST})


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh after staging added, moved or removed files")
    args = parser.parse_args()
    paths = tracked_files()
    actual = []
    for name in paths:
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"Missing file or unsupported symlink: {name}. Stage intended removals before refreshing.")
        actual.append(dict(path=name, size_bytes=str(path.stat().st_size), sha256=digest(path)))
    if args.write:
        with (ROOT / MANIFEST).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "size_bytes", "sha256"], lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(actual)
        print(f"Wrote {len(actual)} entries (the manifest excludes itself).")
        return
    with (ROOT / MANIFEST).open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    expected_by_path = {row["path"]: row for row in expected}
    actual_by_path = {row["path"]: row for row in actual}
    if len(expected_by_path) != len(expected):
        raise SystemExit("Duplicate manifest paths.")
    differences = [name for name in sorted(expected_by_path.keys() | actual_by_path.keys())
                   if expected_by_path.get(name) != actual_by_path.get(name)]
    if differences:
        raise SystemExit("Manifest differences:\n" + "\n".join(differences))
    print(f"Verified {len(actual)} tracked files: paths, sizes and SHA-256 checksums.")


if __name__ == "__main__":
    main()
