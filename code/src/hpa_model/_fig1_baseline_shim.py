"""Minimal baseline-reconstruction helpers for the standalone figure bundle.

Verbatim extraction of the four helpers used by Figure 1 / Figure 2 peak
detection, lifted from ``analysis/plotting/ultradian_demodulated_diagnostics.py``
(which is NOT vendored here — it is 1200+ lines and drags in scipy.signal /
flow-field code the figures don't need). Only the two-harmonic baseline path is
needed: the Cortisol branch evaluates the per-subject sidecar harmonic params;
the ACTH (Fig 2 SI) branch re-fits the two-harmonic on the signal.
"""
from __future__ import annotations

import numpy as np

from .data.registry import get_dataset_spec, load_shift_params
from .data.two_harmonic_shift import evaluate_two_harmonic, fit_two_harmonic_params


def _normalize_series_id(spec_id_col: str, series_id: object) -> object:
    if spec_id_col == "ID":
        return int(series_id)
    return str(series_id)


def _load_shift_param_rows(dataset_name: str, variant: str) -> dict[object, dict[str, float | str]]:
    spec = get_dataset_spec(dataset_name)
    frame = load_shift_params(dataset_name, variant=variant)
    rows: dict[object, dict[str, float | str]] = {}
    for _, row in frame.iterrows():
        key = _normalize_series_id(spec.id_col, row[spec.id_col])
        rows[key] = row.to_dict()
    return rows


def _evaluate_sidecar_baseline(time_min: np.ndarray, shift_row: dict[str, float | str]) -> np.ndarray:
    params = {
        "a24": float(shift_row["a24"]),
        "phase24": float(shift_row["phase24"]),
        "a12": float(shift_row["a12"]),
        "phase12": float(shift_row["phase12"]),
        "c": float(shift_row["c"]),
        "period_min": float(shift_row["period_min"]),
        "second_period_min": float(shift_row["second_period_min"]),
    }
    # Subtract applied_shift_min because time_min is already shifted,
    # but the harmonic parameters were fitted to raw time.
    shift = float(shift_row.get("applied_shift_min", 0.0))
    return evaluate_two_harmonic(np.asarray(time_min, dtype=float) - shift, params)


def _fit_signal_specific_baseline(time_min: np.ndarray, values: np.ndarray, shift_row: dict[str, float | str]) -> np.ndarray:
    params = fit_two_harmonic_params(
        np.asarray(time_min, dtype=float),
        np.asarray(values, dtype=float),
        period_min=float(shift_row["period_min"]),
        second_period_min=float(shift_row["second_period_min"]),
    )
    return evaluate_two_harmonic(np.asarray(time_min, dtype=float), params)


def reconstruct_signal_baseline(
    *,
    dataset_name: str,
    signal_name: str,
    time_min: np.ndarray,
    values: np.ndarray,
    shift_row: dict[str, float | str],
) -> tuple[np.ndarray, str]:
    if signal_name == "Cortisol":
        return _evaluate_sidecar_baseline(time_min, shift_row), "shift_sidecar"
    return _fit_signal_specific_baseline(time_min, values, shift_row), "signal_specific_two_harmonic_fit"
