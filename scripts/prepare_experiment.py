"""Stage OSS inputs beside an unchanged experiment script; do not execute it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_manifest():
    manifest = json.loads((ROOT / "experiments/source_manifest.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in manifest["experiments"]}


def prepare(identifier, work_dir):
    manifest = load_manifest()
    if identifier not in manifest:
        raise ValueError(f"Unknown experiment: {identifier}")
    entry = manifest[identifier]
    source = ROOT / entry["path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Experiment source differs from its preserved original")
    arrays = [ROOT / "data/OSS" / f"OSS_{i}.xlsx" for i in range(1, 5)]
    for array in arrays:
        if not array.is_file():
            raise FileNotFoundError(array)
    # Refuse reuse so variants with identical output names cannot overwrite a prior run.
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    input_dir = work_dir / entry["input_folder"]
    input_dir.mkdir()
    for array in arrays:
        shutil.copy2(array, input_dir / array.name)
    shutil.copy2(source, work_dir / "experiment.py")
    if (work_dir / "experiment.py").read_bytes() != source.read_bytes():
        raise ValueError("Staged script differs from its preserved original")
    return work_dir, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=sorted(load_manifest()))
    parser.add_argument("--list", action="store_true", help="List the seven preserved experiment variants")
    parser.add_argument("--work-dir", type=Path, help="New, non-existing run directory")
    args = parser.parse_args(argv)
    if args.list:
        for identifier, entry in load_manifest().items():
            print(f"{identifier}: {entry['path']} (input: {entry['input_folder']})")
        return 0
    if not args.experiment:
        parser.error("choose an experiment or use --list")
    try:
        target, entry = prepare(args.experiment, args.work_dir or ROOT / "runs" / args.experiment)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print("Prepared only; no experiment was executed and no numerical setting was overridden.")
    print(f"Source SHA-256: {entry['sha256']}")
    print(f'Next: cd "{target}"')
    print("Then run: python experiment.py")
    print(f"Original output folder: {entry['output_folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
