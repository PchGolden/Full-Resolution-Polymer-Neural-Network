"""Invoke the canonical md_final1640_v2 preprocessor."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[2] / "frpn/pipelines/md_final1640_v2/preprocess"
sys.path.insert(0, str(root))
runpy.run_path(str(root / "preprocessing_polymer.py"), run_name="__main__")
