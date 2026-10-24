# Data Dictionary

## Time Convention

The four datasets contain elapsed calendar times, in whole seconds. Each individual issue creation time is treated as a fault-detection time proxy. They are not monthly observations, CPU times, interarrival times, or issue closure times.

For each project, the earliest archived issue defines `t0`. Its event index is zero. The model array contains only subsequent issues, with indices 1 through N. Thus, an observation range reported as `0` to `T` has N model events but N+1 archived records. A zero is not prepended to the arrays.

## Complete Workbook

File: [`JSS_OSS_time_datasets_OSS1_OSS4.xlsx`](../data/JSS_OSS_time_datasets_OSS1_OSS4.xlsx).

| Sheet | Contents |
|---|---|
| `Dataset_Metadata` | Paper IDs, projects, event counts, time ranges, origin issues, timestamps, source sheet names, source hashes, and label-coverage counts. |
| `OSS_1` to `OSS_4` | One-column, headerless model arrays, in calendar seconds. |
| `Raw_Issue_Records` | All 1,033 archived issue records, including the four time-origin records. |
| `README` | Definitions, construction steps, checks, repository address, and limitations. |

### Raw Issue Fields

| Field | Type | Meaning |
|---|---|---|
| `Dataset ID` | Text | `OSS_1`, `OSS_2`, `OSS_3`, or `OSS_4`. |
| `Issue Number` | Integer | Issue number within the corresponding repository. |
| `Title` | Text | Archived original issue title. |
| `Created At (UTC)` | Excel datetime | Issue creation timestamp, displayed in UTC. Excel has no native timezone field. |
| `Categories` | Text | Original archived labels, without a new filtering step. |
| `Issue URL` | URL | Repository issue address associated with the archived issue number. |
| `Source Excel Row` | Integer | One-based row in the archived crawler sheet, including its header row. |
| `Event Index` | Integer | Zero for the origin record, then 1 through N for model events. |
| `Elapsed Calendar Seconds` | Integer formula | Rounded whole-second difference between the typed timestamp and that project's time origin. |
| `Record Role` | Text | Time origin, excluded from the model array, or model event. |
| `Has Bug Label` | Boolean | Whether the archived labels contain `bug`, case-insensitively. This is not independent defect verification. |
| `Original Created At Text` | Text | Exact archived UTC timestamp string, e.g. `2022-12-08T03:45:30Z`. This is the reconstruction script's authoritative timestamp field. |

The typed Excel datetime may be affected by floating-point storage at millisecond precision. Original UTC text is preserved exactly. The reconstruction script uses that text, and all exported elapsed times are exact whole-second integers.

### Metadata Counts

`Model Events` excludes the origin issue. `Raw Records` includes it. `Start (sec)` is the observation origin, not the minimum value of the model array. `First Model Event (sec)` gives the first positive array value. `Model Records Without Bug Label` excludes the origin record.

The source SHA-256 values identify the historical crawler workbooks, not the combined release workbook. Those larger crawler workbooks are not bundled in this repository; the selected source records and original timestamp strings are included.

## Standalone Excel and CSV Files

Files in [`data/OSS`](../data/OSS) are unchanged copies of the original experiment inputs, mapped to paper IDs. Each has one worksheet, one numeric column, no header, and N rows.

The reconstruction script writes matching headerless CSV files to [`data/processed_time`](../data/processed_time). The summary CSV has a header and includes the event count, time origin, first model time, maximum time, raw-record count, and count of model records without a bug label.

## Selection Limitation

The release preserves the existing archived selection to reproduce the paper's analysis. It does not apply a new bug-label filter or claim a verified defect-only census. Among model events, the numbers without a label containing `bug` are 0, 6, 17, and 2, respectively. Refiltering those records would define different datasets and require reevaluating the analysis.
