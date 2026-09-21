"""Protect against cross-workflow imports and CLI execution during imports."""

import importlib
from pathlib import Path
import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_both_model_families_coexist(self):
        from frpn.linear import data as linear_data, train as linear_train
        from frpn.linear.models.model import MultiMolModel as LinearModel
        from frpn.md import data as md_data, train as md_train
        from frpn.md.models.model import MultiMolModel as MDModel

        self.assertIs(linear_train.MultiMolModel, LinearModel)
        self.assertIs(md_train.MultiMolModel, MDModel)
        self.assertIsNot(LinearModel, MDModel)
        self.assertIs(linear_train.build_dataloader, linear_data.build_dataloader)
        self.assertIs(md_train.build_dataloader, md_data.build_dataloader)
        self.assertIsNot(linear_data.collate_fn, md_data.collate_fn)

    def test_all_modules_import_without_running_a_command(self):
        # A parser executed at import time would consume unittest's arguments.
        root = Path(__file__).resolve().parents[1]
        original_path = list(sys.path)
        for path in sorted((root / "frpn").rglob("*.py")):
            if path.name == "__init__.py":
                continue
            name = ".".join(path.relative_to(root).with_suffix("").parts)
            with self.subTest(module=name):
                importlib.import_module(name)
        self.assertEqual(sys.path, original_path)
        for alias in ("models", "dataloader", "main", "molecular_features"):
            self.assertNotIn(alias, sys.modules)


if __name__ == "__main__":
    unittest.main()
