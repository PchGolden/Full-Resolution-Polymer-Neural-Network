"""Export validation predictions using the packaged BCDB or MD exporter."""
from pathlib import Path
import runpy
import sys

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m frpn.cli.export_oof {bcdb|md} [exporter arguments]\nUse bcdb --help or md --help for dataset-specific options.")
        return
    dataset = sys.argv.pop(1)
    root = Path(__file__).resolve().parents[2]
    if dataset == "bcdb":
        pipeline = root / "frpn/pipelines/bcdb"
        sys.path.insert(0, str(pipeline))
        script = pipeline / "export_bcdb_probs.py"
    elif dataset == "md":
        script = root / "scripts/evaluate/export_md_oof.py"
    else:
        raise SystemExit("Choose bcdb or md")
    runpy.run_path(str(script), run_name="__main__")
if __name__ == "__main__":
    main()
