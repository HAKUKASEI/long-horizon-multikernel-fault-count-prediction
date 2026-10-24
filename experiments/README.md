# Original Experiment Programs

The seven supplied programs are preserved **byte-for-byte**. Only their repository filenames and locations have been organized. No algorithms, constants, bandwidth searches, weights, kernels, worker limits, observation fractions, output names, or main-entry settings have been edited.

[`source_manifest.json`](source_manifest.json) records each original filename, repository path, byte count, SHA-256 digest, input directory, output directory, and main-entry settings. Git attributes prevent line-ending normalization of these source files. Tests verify their hashes and compile their syntax without importing or running the experiments.

## Program Map

| Preparation ID | Method | Original filename | Preserved source |
|---|---|---|---|
| `single-dynamic` | Single kernel, sequential bandwidth | 单核动态窗宽.py | [Source](single_kernel/dynamic_bandwidth.py) |
| `single-fixed` | Single kernel, fixed bandwidth | 单核固定窗宽.py | [Source](single_kernel/fixed_bandwidth.py) |
| `multi-dynamic-dynamic` | Multi-kernel, dynamic bandwidth and weights | 动态窗宽动态参数.py | [Source](multi_kernel/dynamic_bandwidth_dynamic_weights.py) |
| `multi-dynamic-fixed` | Multi-kernel, dynamic bandwidth and fixed weights | 动态窗宽固定权重.py | [Source](multi_kernel/dynamic_bandwidth_fixed_weights.py) |
| `multi-fixed-dynamic` | Multi-kernel, fixed bandwidth and dynamic weights | 固定窗宽动态权重.py | [Source](multi_kernel/fixed_bandwidth_dynamic_weights.py) |
| `multi-fixed-dynamic-underscore` | Separately supplied fixed-bandwidth/dynamic-weight variant | 固定窗宽_动态权重.py | [Source](multi_kernel/fixed_bandwidth_dynamic_weights_underscore_variant.py) |
| `multi-fixed-fixed` | Multi-kernel, fixed bandwidth and weights | 固定窗宽固定权重.py | [Source](multi_kernel/fixed_bandwidth_fixed_weights.py) |

The two fixed-bandwidth/dynamic-weight files are **not duplicates**. Their prediction and bandwidth-related functions differ. Both are retained without merging, and neither is designated as the definitive version for a particular paper result table. Select the variant that was used for the result being reproduced.

## Prepare and Run

From the repository root, in a suitable Python environment:

```bash
python -m pip install -r requirements-experiments.txt
python scripts/prepare_experiment.py --list
python scripts/prepare_experiment.py single-dynamic
cd runs/single-dynamic
python experiment.py
```

Replace `single-dynamic` with another preparation ID in the table. The preparation command does not execute the experiment. It creates a new run directory containing:

```text
runs/<preparation-id>/
|-- experiment.py      # Byte-identical copy of the selected original program
`-- SDATA/ or DATA/    # The input directory expected by that original program
    |-- OSS_1.xlsx
    |-- OSS_2.xlsx
    |-- OSS_3.xlsx
    `-- OSS_4.xlsx
```

Single-kernel programs look for `SDATA` in the current working directory; multi-kernel programs look for `DATA`. The helper accommodates those paths without rewriting the code, importing it, or overriding any settings. The input filenames use the paper's OSS IDs; their numeric contents are unchanged. Generated result labels therefore use `OSS_1` through `OSS_4`.

Run `experiment.py` **from its prepared run directory**. Do not launch the preserved programs from the repository root: on Windows, the existing lowercase `data` directory can be confused with uppercase `DATA`, causing the wrong workbook to be read. Do not place the combined metadata workbook in an experiment input folder.

Existing run directories are never overwritten by the helper. For another run, select a new directory:

```bash
python scripts/prepare_experiment.py single-dynamic --work-dir runs/single-dynamic-repeat-2
```

Separate run directories also prevent the two fixed-bandwidth/dynamic-weight variants, which use the same original output-folder name, from overwriting each other's results. Experiment outputs stay inside the run directory under the original output-folder name. The `runs/` directory is ignored by Git.

## Preserved Entry-Point Settings

All seven scripts use observation fractions `(0.2, 0.5, 0.8)`, the `spawn` multiprocessing start method, and worker warm-up. The single-kernel entry points use `v_grid_size=1000`. The multi-kernel entry points use `estimator_types=("ordinary",)` and methods `("LLCV", "LSCV", "Campbell")`.

| Preparation ID | Input folder | Requested maximum workers |
|---|---|---:|
| `single-dynamic` | `SDATA` | 6 |
| `single-fixed` | `SDATA` | 6 |
| `multi-dynamic-dynamic` | `DATA` | 60 |
| `multi-dynamic-fixed` | `DATA` | 30 |
| `multi-fixed-dynamic` | `DATA` | 15 |
| `multi-fixed-dynamic-underscore` | `DATA` | 15 |
| `multi-fixed-fixed` | `DATA` | 60 |

These are the original requested limits, not a recommendation for every machine. The programs retain their own worker-selection logic. Full runs may consume substantial CPU, memory, and time. Run one variant at a time and inspect the source before starting it on a shared system.

## Dependencies and Notifications

The experiment programs import NumPy, pandas, SciPy, Matplotlib, Numba, and openpyxl. These are listed separately in [`requirements-experiments.txt`](../requirements-experiments.txt), so data-only validation does not need the full numerical stack. The original experiment environment/version lock was not supplied; the experiment dependency list is not a claim to reproduce that original environment exactly.

The multi-kernel programs retain their optional `classicUtils.SRMtool` email-notification calls and original recipient. That optional module is not bundled or installed by the requirements file. As written in the original code, notifications are skipped when the module is unavailable. If it is installed and configured in your environment, running the original main entry point may send start/finish notifications. Packaging and validation do not execute those calls.

## What Has Been Verified

- All seven source files match the supplied files byte-for-byte, including comments and line endings.
- All seven programs compile syntactically.
- Every preparation ID stages the unchanged program and the four unchanged OSS inputs in the required folder.
- The existing data checks still validate all 1,029 OSS time values.

The release checks do **not** execute the long-running forecasting jobs, reproduce the paper's numerical tables, or validate every combination of numerical-library versions. The four closed-source datasets are not included in this release; the preparation helper stages only the four published OSS datasets.
