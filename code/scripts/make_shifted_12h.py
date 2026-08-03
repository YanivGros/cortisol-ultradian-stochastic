"""Regenerate the `shifted_12h` dataset variant from `raw/data_raw.csv`.

This is the raw -> shifted step of the bundle: each subject's fitted 24-hour
cortisol acrophase is aligned to 10:00 (target_peak_min=600) via the canonical
two-harmonic shift, with a 12-hour second harmonic (720 min). Reproduces the
packaged catalog `shifted_12h` byte-for-byte.

Self-contained: uses the vendored `two_harmonic_shift.shift_dataframe_by_peak24`
and a verbatim copy of `_complete_native_grid` from the repo's package_datasets.py
(so we don't vendor package_datasets, which reads original digitized source files).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hpa_model.data.registry import DATASETS_ROOT  # noqa: E402
from hpa_model.data.two_harmonic_shift import (  # noqa: E402
    PERIOD_MIN, infer_native_timestep, minutes_to_hhmm, shift_dataframe_by_peak24,
)

TARGET_PEAK_MIN = 600.0          # 10:00 canonical acrophase alignment
SECOND_PERIOD_MIN = 720.0        # 12-hour second harmonic

# (id_col, fit_value_col, value_cols) per dataset — from package_datasets.PACKAGE_SPECS.
SPECS = {
    "habs":          ("ID",        "Cortisol", ("Cortisol", "ACTH")),
    "all_digitized": ("ID",        "cortisol", ("cortisol",)),
    "digitize_2019": ("series_id", "value",    ("value", "ACTH")),
}


def _complete_native_grid(frame, *, id_col, time_col, value_cols, time_label_col=None):
    """Verbatim from hpa_model.data.package_datasets._complete_native_grid."""
    completed_groups = []
    for series_id, group in frame.groupby(id_col, sort=False):
        work = group.copy().sort_values(time_col)
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).copy()
        work = work.groupby([id_col, time_col], as_index=False).agg({c: "mean" for c in value_cols})
        times = work[time_col].to_numpy(dtype=float)
        step = infer_native_timestep(times)
        full_day = bool(times.size > 1 and (float(times.max()) - float(times.min()) >= PERIOD_MIN - 2.0 * step))
        target = (np.arange(0.0, PERIOD_MIN, step, dtype=float) if full_day
                  else np.arange(float(times.min()), float(times.max()) + 0.5 * step, step, dtype=float))
        completed = pd.DataFrame({id_col: series_id, time_col: target})
        for column in value_cols:
            values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not np.any(valid):
                completed[column] = np.nan
                continue
            src_times, src_values = times[valid], values[valid]
            if full_day:
                ext_t = np.concatenate([src_times - PERIOD_MIN, src_times, src_times + PERIOD_MIN])
                ext_v = np.concatenate([src_values, src_values, src_values])
                completed[column] = np.interp(target, ext_t, ext_v)
            else:
                completed[column] = np.interp(target, src_times, src_values)
        if time_label_col:
            completed[time_label_col] = completed[time_col].map(minutes_to_hhmm)
        ordered = [id_col] + ([time_label_col] if time_label_col else []) + [*value_cols, time_col]
        completed_groups.append(completed[ordered])
    return pd.concat(completed_groups, ignore_index=True)


def main() -> None:
    for name, (id_col, fit_col, value_cols) in SPECS.items():
        raw_path = DATASETS_ROOT / name / "raw" / "data_raw.csv"
        raw = pd.read_csv(raw_path)
        res = shift_dataframe_by_peak24(
            raw, id_col=id_col, time_col="time_min", fit_value_col=fit_col,
            value_cols=value_cols, output_value_cols=value_cols,
            target_peak_min=TARGET_PEAK_MIN, second_period_min=SECOND_PERIOD_MIN,
            time_label_col="Time",
        )
        completed = _complete_native_grid(
            res.shifted, id_col=id_col, time_col="time_min",
            value_cols=value_cols, time_label_col="Time",
        )
        out = DATASETS_ROOT / name / "shifted_12h"
        out.mkdir(parents=True, exist_ok=True)
        completed.to_csv(out / "data_shifted.csv", index=False)
        res.metadata.to_csv(out / "shift_params.csv", index=False)
        aligned = (res.metadata["peak24_min"] + res.metadata["applied_shift_min"]) % 1440
        print(f"{name:<14} rows={len(completed):>5} series={res.metadata.shape[0]:>3} "
              f"24h-peak~{minutes_to_hhmm(float(aligned.mean()) % 1440)} -> {out}")


if __name__ == "__main__":
    main()
