# Obtaining the data

This repository contains **code only**. No human participant data is included.
All hormone recordings analyzed in the paper were previously published by other
groups, and they remain governed by the terms of their original deposits. This
file tells you where to get each dataset and exactly where to put it so that
`python run_all.py` works.

## Sources

| Dataset key | Study | What it is | How to obtain |
|---|---|---|---|
| `habs` | Upton et al. 2023, *Sci. Transl. Med.* 15(701):eadg8464 | Paired serum ACTH + cortisol, n = 7, 20 min sampling | Public deposit: **[doi:10.18710/5TW8YF](https://doi.org/10.18710/5TW8YF)**, files `Data files/HABS 1.csv` ... `HABS 7.csv` |
| `habs_microdialysis_cortisol` | Upton et al. 2023 (tissue arm) | Subcutaneous microdialysis cortisol, n = 213, 20 min sampling | Same deposit, file `Data files/md_data_controls.csv` |
| `digitize_2019` | Henley et al. 2009, *J. Med. Eng. Technol.* 33(3):199-208 | Serum ACTH + cortisol, n = 2, 10 min sampling | Digitized from the published figures; request from the original authors |
| `all_digitized` | Young et al. 2004, *Front. Neuroendocrinol.* 25(2):69-76 | Serum total cortisol, n = 62, 10-15 min sampling | Digitized from the published figures; request from the original authors |

The Upton et al. deposit is deidentified and openly available under the terms
stated in its own `00_README.txt`. Please cite the original study, not this
repository, when using any of these recordings.

## Where the code looks for it

The registry (`code/src/hpa_model/data/registry.py`) resolves every dataset to:

```
code/data/catalog/datasets/<dataset_key>/<variant>/data_shifted.csv   # variant != "raw"
code/data/catalog/datasets/<dataset_key>/<variant>/data_raw.csv       # variant == "raw"
```

The figure pipeline reads the **`shifted_12h`** variant for the three serum
datasets and the **`shifted`** variant for the microdialysis dataset. Fig 8 also
reads one raw file directly:

```
code/data/raw_data_input/Upton et al. (2023) blood.csv
```

This is the seven paired serum/microdialysis recordings concatenated, with an
`ID` column added, and it is used only to fit the diffusion rate `k`. Columns
consumed are `ID`, `Time`, `ACTH`, `Cortisol`, and `mCortisol`.

## Expected schema

Column names are load-bearing; the registry maps them per dataset. `time_min`
is minutes from the start of the series. `Time` is a `HH:MM` clock string.

| Dataset key | ID column | Signal columns | Full column list |
|---|---|---|---|
| `habs` | `ID` | `Cortisol`, `ACTH` | `ID, Time, Cortisol, ACTH, time_min` |
| `all_digitized` | `ID` | `cortisol` | `ID, Time, cortisol, time_min` |
| `digitize_2019` | `series_id` | `value` (cortisol), `ACTH` | `series_id, Time, value, ACTH, time_min` |
| `habs_microdialysis_cortisol` | `MasterID` | `Cortisol` | `MasterID, Time, Cortisol, time_min` |

For the `raw` variant, drop the `Time` column.

## Verifying your catalog

Once assembled, these are the shapes the paper was produced from:

| Dataset key | Variant | Rows | Series | Median dt |
|---|---|---|---|---|
| `habs` | `shifted_12h` | 504 | 7 | 20 min |
| `all_digitized` | `shifted_12h` | 7248 | 62 | 15 min |
| `digitize_2019` | `shifted_12h` | 288 | 2 | 10 min |
| `habs_microdialysis_cortisol` | `shifted` | 15336 | 213 | 20 min |

The three serum datasets together give the 71 series and 518 detected pulses
reported in the paper.

## Building the catalog from the source files

Two helpers turn source files into the catalog the figure scripts read.

**Step 1, source files to `raw` variants.**
`code/src/hpa_model/data/package_datasets.py` expects these exact filenames in
`code/data/raw_data_input/`:

| Source filename | Produces dataset key |
|---|---|
| `Upton et al. (2023) blood.csv` | `habs` |
| `Young et al. (2004).csv` | `all_digitized` |
| `Russell & Lightman.csv` | `digitize_2019` |

The version of `package_datasets.py` shipped here covers those three **serum**
datasets only. The microdialysis catalog is not built by it: assemble
`habs_microdialysis_cortisol` yourself from the deposit's `md_data_controls.csv`
by keeping the `MasterID` and `Cortisol` columns, converting `SampleTime` to
`Time` (`HH:MM`) and `time_min`, and then applying the same 24 h + 12 h alignment
to a `shifted/` variant. Target shapes are in the verification table above.

**Step 2, `raw` to `shifted_12h`.** The aligned variant is not a separate
measurement, it is derived: each series is placed on a common circadian phase by
fitting a 24 h + 12 h harmonic and shifting so the circadian maximum falls at
10:00 (`target_peak_min = 600`).

```bash
python code/scripts/make_shifted_12h.py
```

This covers the three **serum** datasets (`habs`, `all_digitized`,
`digitize_2019`) and reproduces the packaged `shifted_12h` catalog byte for byte.
It writes `data_shifted.csv` plus a `shift_params.csv` recording the fitted
harmonic coefficients and the applied shift per series, so the alignment is
auditable. The microdialysis `shifted` variant is produced by
`package_datasets.py` rather than by this script.
