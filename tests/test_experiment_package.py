import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_experiment import load_manifest, prepare


class ExperimentPackageTests(unittest.TestCase):
    def test_all_seven_sources_are_unchanged_and_compile(self):
        manifest = load_manifest()
        self.assertEqual(len(manifest), 7)
        for entry in manifest.values():
            with self.subTest(experiment=entry["id"]):
                source = (ROOT / entry["path"]).read_bytes()
                self.assertEqual(len(source), entry["bytes"])
                self.assertEqual(hashlib.sha256(source).hexdigest(), entry["sha256"])
                compile(source, entry["path"], "exec")
                tree = ast.parse(source)
                main = next(node for node in tree.body if isinstance(node, ast.If)
                            and "__main__" in ast.unparse(node.test))
                batch = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Name) and node.func.id.startswith("run_batch_analysis"))
                settings = {kw.arg: ast.literal_eval(kw.value) for kw in batch.keywords if kw.arg != "base_folder"}
                self.assertEqual(settings["fractions"], tuple(entry["main_settings"]["fractions"]))
                self.assertEqual(settings["max_workers"], entry["main_settings"]["max_workers"])

    def test_staging_all_variants_preserves_sources_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for identifier, entry in load_manifest().items():
                with self.subTest(experiment=identifier):
                    target, _ = prepare(identifier, Path(directory) / identifier)
                    self.assertEqual((target / "experiment.py").read_bytes(), (ROOT / entry["path"]).read_bytes())
                    for index in range(1, 5):
                        name = f"OSS_{index}.xlsx"
                        self.assertEqual((target / entry["input_folder"] / name).read_bytes(),
                                         (ROOT / "data/OSS" / name).read_bytes())
                    self.assertFalse((target / entry["output_folder"]).exists())

    def test_existing_run_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                prepare("single-fixed", Path(directory))

    def test_unknown_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                prepare("unknown", Path(directory) / "new")


if __name__ == "__main__":
    unittest.main()
