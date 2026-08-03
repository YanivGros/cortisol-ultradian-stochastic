"""Independent-phase two-harmonic shift utilities for time-of-day series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


PERIOD_MIN = 24.0 * 60.0
SECOND_PERIOD_MIN = PERIOD_MIN / 2.0


def wrap_phase_positive(phase: float) -> float:
    return float(phase % (2.0 * math.pi))


def minutes_to_hhmm(minutes: float) -> str:
    minute_of_day = int(round(minutes)) % int(PERIOD_MIN)
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def _harmonic_model(
    t_eval: np.ndarray,
    *,
    a24: float,
    phi24: float,
    a12: float,
    phi12: float,
    c: float,
    period_min: float,
    second_period_min: float,
) -> np.ndarray:
    w24 = 2.0 * math.pi / float(period_min)
    w12 = 2.0 * math.pi / float(second_period_min)
    return (
        float(a24) * np.sin(w24 * t_eval + float(phi24))
        + float(a12) * np.sin(w12 * t_eval + float(phi12))
        + float(c)
    )


def fit_sine_params(
    time_min: np.ndarray,
    y: np.ndarray,
    *,
    period_min: float = PERIOD_MIN,
) -> tuple[float, float, float, float] | None:
    """Fit y = a*sin(wt) + b*cos(wt) + c and return peak time in minutes."""
    w = 2.0 * math.pi / float(period_min)
    t = np.asarray(time_min, dtype=float)
    values = np.asarray(y, dtype=float)
    if t.size == 0 or values.size == 0:
        return None
    X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    try:
        coeffs, *_ = np.linalg.lstsq(X, values, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b, c = coeffs
    amp = np.hypot(a, b)
    if not np.isfinite(amp) or amp <= 1e-12:
        return None
    phi = np.arctan2(b, a)
    t_peak = (np.pi / 2.0 - phi) / w
    return float(a), float(b), float(c), float(t_peak % period_min)


def fit_two_harmonic_params(
    time_min: np.ndarray,
    y: np.ndarray,
    *,
    period_min: float = PERIOD_MIN,
    second_period_min: float = SECOND_PERIOD_MIN,
    peak_grid_min: float = 1.0,
) -> dict[str, float] | None:
    """Fit y = a24*sin(w24*t + phi24) + a12*sin(w12*t + phi12) + c."""
    t = np.asarray(time_min, dtype=float)
    values = np.asarray(y, dtype=float)
    if t.size == 0 or values.size == 0:
        return None

    single_fit = fit_sine_params(t, values, period_min=period_min)
    c0 = float(np.mean(values))
    amp_scale = float(max(np.std(values), 1e-6))
    if single_fit is None:
        a24_0 = amp_scale
        phi24_0 = 0.0
    else:
        a_sin, b_cos, _, _ = single_fit
        a24_0 = float(np.hypot(a_sin, b_cos))
        phi24_0 = float(np.arctan2(b_cos, a_sin))
    x0 = np.array([a24_0, phi24_0, 0.5 * amp_scale, 0.0, c0], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        return _harmonic_model(
            t,
            a24=float(params[0]),
            phi24=float(params[1]),
            a12=float(params[2]),
            phi12=float(params[3]),
            c=float(params[4]),
            period_min=period_min,
            second_period_min=second_period_min,
        ) - values

    try:
        res = least_squares(residuals, x0=x0)
    except ValueError:
        return None
    if not res.success:
        return None

    a24, phi24, a12, phi12, c = (float(x) for x in res.x)
    amp24 = abs(a24)
    amp12 = abs(a12)
    if (amp24 <= 1e-12 and amp12 <= 1e-12) or not np.isfinite(amp24) or not np.isfinite(amp12):
        return None

    if a24 < 0.0:
        a24 = -a24
        phi24 += math.pi
    if a12 < 0.0:
        a12 = -a12
        phi12 += math.pi

    phi24 = wrap_phase_positive(phi24)
    phi12 = wrap_phase_positive(phi12)
    w24 = 2.0 * math.pi / float(period_min)
    w12 = 2.0 * math.pi / float(second_period_min)
    t_eval = np.arange(0.0, float(period_min), float(peak_grid_min), dtype=float)
    y_eval = _harmonic_model(
        t_eval,
        a24=a24,
        phi24=phi24,
        a12=a12,
        phi12=phi12,
        c=c,
        period_min=period_min,
        second_period_min=second_period_min,
    )
    combined_peak_min = float(t_eval[int(np.argmax(y_eval))] % period_min)
    peak24_min = float(((math.pi / 2.0 - phi24) / w24) % period_min)
    peak12_min = float(((math.pi / 2.0 - phi12) / w12) % second_period_min)

    return {
        "a24": float(a24),
        "amp24": float(amp24),
        "phase24": float(phi24),
        "peak24_min": peak24_min,
        "a12": float(a12),
        "amp12": float(amp12),
        "phase12": float(phi12),
        "peak12_min": peak12_min,
        "c": float(c),
        "combined_peak_min": combined_peak_min,
        "period_min": float(period_min),
        "second_period_min": float(second_period_min),
        "cost": float(res.cost),
    }


def evaluate_two_harmonic(
    time_min: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    return _harmonic_model(
        np.asarray(time_min, dtype=float),
        a24=float(params["a24"]),
        phi24=float(params["phase24"]),
        a12=float(params["a12"]),
        phi12=float(params["phase12"]),
        c=float(params["c"]),
        period_min=float(params["period_min"]),
        second_period_min=float(params["second_period_min"]),
    )


def infer_native_timestep(time_min: Sequence[float]) -> float:
    values = np.asarray(time_min, dtype=float)
    if values.size < 2:
        return 1.0
    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 1.0
    rounded = np.round(diffs, 6)
    unique, counts = np.unique(rounded, return_counts=True)
    return float(unique[int(np.argmax(counts))])


def compute_signed_shift(
    peak_min: float,
    *,
    target_peak_min: float,
    native_timestep_min: float,
    period_min: float = PERIOD_MIN,
) -> float:
    delta_raw = float(target_peak_min) - float(peak_min)
    signed = ((delta_raw + period_min / 2.0) % period_min) - period_min / 2.0
    step = max(float(native_timestep_min), 1e-9)
    return float(np.round(signed / step) * step)


@dataclass(frozen=True)
class ShiftResult:
    shifted: pd.DataFrame
    metadata: pd.DataFrame


def shift_dataframe_by_peak24(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    fit_value_col: str,
    value_cols: Sequence[str],
    output_value_cols: Sequence[str] | None = None,
    target_peak_min: float = 600.0,
    period_min: float = PERIOD_MIN,
    second_period_min: float = SECOND_PERIOD_MIN,
    time_label_col: str | None = "Time",
) -> ShiftResult:
    """Shift each series so the 24-hour component peak lands at target_peak_min."""
    metadata_rows: list[dict[str, float | str | object]] = []
    shifted_groups: list[pd.DataFrame] = []
    output_value_cols = tuple(value_cols if output_value_cols is None else output_value_cols)

    for series_id, group in df.groupby(id_col, sort=False):
        work = group.copy()
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).copy()
        if work.empty:
            continue
        native_step = infer_native_timestep(work[time_col].to_numpy(dtype=float))
        fit_source = work[[time_col, fit_value_col]].copy()
        fit_source[fit_value_col] = pd.to_numeric(fit_source[fit_value_col], errors="coerce")
        fit_source = fit_source.dropna(subset=[fit_value_col]).copy()
        fallback_mode = "two_harmonic"
        fit_params = None
        if not fit_source.empty:
            fit_params = fit_two_harmonic_params(
                fit_source[time_col].to_numpy(dtype=float),
                fit_source[fit_value_col].to_numpy(dtype=float),
                period_min=period_min,
                second_period_min=second_period_min,
            )
        if fit_params is None and not fit_source.empty:
            sine_fit = fit_sine_params(
                fit_source[time_col].to_numpy(dtype=float),
                fit_source[fit_value_col].to_numpy(dtype=float),
                period_min=period_min,
            )
            if sine_fit is not None:
                a, b, c, peak_min = sine_fit
                fit_params = {
                    "a24": float(np.hypot(a, b)),
                    "amp24": float(np.hypot(a, b)),
                    "phase24": float(np.arctan2(b, a) % (2.0 * math.pi)),
                    "peak24_min": float(peak_min),
                    "a12": float("nan"),
                    "amp12": float("nan"),
                    "phase12": float("nan"),
                    "peak12_min": float("nan"),
                    "c": float(c),
                    "combined_peak_min": float(peak_min),
                    "period_min": float(period_min),
                    "second_period_min": float(second_period_min),
                    "cost": float("nan"),
                }
                fallback_mode = "single_sine"
        if fit_params is None:
            values = pd.to_numeric(work[fit_value_col], errors="coerce")
            if values.notna().any():
                max_value = float(values.max())
                peak_min = float(work.loc[values == max_value, time_col].min())
            else:
                peak_min = 0.0
            fit_params = {
                "a24": float("nan"),
                "amp24": float("nan"),
                "phase24": float("nan"),
                "peak24_min": float(peak_min),
                "a12": float("nan"),
                "amp12": float("nan"),
                "phase12": float("nan"),
                "peak12_min": float("nan"),
                "c": float("nan"),
                "combined_peak_min": float(peak_min),
                "period_min": float(period_min),
                "second_period_min": float(second_period_min),
                "cost": float("nan"),
            }
            fallback_mode = "observed_max"

        applied_shift = compute_signed_shift(
            float(fit_params["peak24_min"]),
            target_peak_min=target_peak_min,
            native_timestep_min=native_step,
            period_min=period_min,
        )
        work[time_col] = (work[time_col].to_numpy(dtype=float) + applied_shift) % period_min
        for column in value_cols:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        agg_map = {column: "mean" for column in value_cols}
        shifted_group = (
            work.groupby([id_col, time_col], as_index=False)
            .agg(agg_map)
            .sort_values([id_col, time_col])
        )
        if time_label_col:
            shifted_group[time_label_col] = shifted_group[time_col].map(minutes_to_hhmm)
        ordered_cols = [id_col]
        if time_label_col:
            ordered_cols.append(time_label_col)
        ordered_cols.extend(output_value_cols)
        ordered_cols.append(time_col)
        shifted_groups.append(shifted_group[ordered_cols].copy())

        metadata_rows.append(
            {
                id_col: series_id,
                "native_timestep_min": float(native_step),
                "applied_shift_min": float(applied_shift),
                "fallback_mode": fallback_mode,
                **fit_params,
            }
        )

    shifted = pd.concat(shifted_groups, ignore_index=True) if shifted_groups else pd.DataFrame()
    metadata = pd.DataFrame(metadata_rows)
    return ShiftResult(shifted=shifted, metadata=metadata)
