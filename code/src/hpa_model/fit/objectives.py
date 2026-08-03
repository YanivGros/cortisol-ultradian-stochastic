"""Objective functions and target structures for peak-statistic fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalTargets:
    signal: str
    value_col: str
    bins: list[str]
    count: np.ndarray
    amplitude: np.ndarray
    ibi: np.ndarray
    cv: np.ndarray
    amplitude_sem: np.ndarray
    ibi_sem: np.ndarray
    cv_sem: np.ndarray
    # Optional scalar — relative residual amplitude:
    # mean over subjects of (residual_std_raw / mean(baseline_raw)).
    # If NaN, the absolute-amplitude objective term is skipped.
    rel_residual_amp: float = float("nan")
    # Optional scalar — full-signal CV: mean over subjects of
    # std(full signal) / mean(full signal). If NaN, the term is skipped.
    full_signal_cv: float = float("nan")


@dataclass(frozen=True)
class HabsTargets:
    frame: pd.DataFrame
    id_col: str
    time_col: str
    signals: dict[str, SignalTargets]

    @property
    def bins(self) -> list[str]:
        return next(iter(self.signals.values())).bins


def _safe_scale(values: np.ndarray, minimum: float = 1e-4) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        scale: np.ndarray = np.nan_to_num(values, nan=minimum)
    scale[scale < minimum] = minimum
    return scale


def build_residual_vector(
    *,
    targets: HabsTargets,
    sim_profiles: dict[str, pd.DataFrame],
    selected_signals: tuple[str, ...],
    cv_weight: float,
    rel_amp_weight: float = 0.0,
    full_cv_weight: float = 0.0,
    count_weight: float = 0.0,
) -> np.ndarray:
    """Build the flattened residual vector for LS optimization.

    If ``rel_amp_weight > 0`` and ``targets.signals[<signal>].rel_residual_amp``
    is finite, an extra scalar term is appended to the residual that compares
    the simulated relative residual amplitude to the target. This anchors the
    absolute (non-z-scored) variability of the noise so the fit cannot trade
    raw amplitude for shape.

    If ``count_weight > 0`` a per-bin term is appended comparing the simulated
    mean peak count (``num_peaks``) to the target count, scaled by the target
    count. This constrains pulse frequency directly rather than only through
    the inter-peak interval.
    """
    residuals: list[np.ndarray] = []
    for signal in selected_signals:
        target = targets.signals[signal]
        sim_profile = sim_profiles[signal].reindex(target.bins)

        target_amp = np.nan_to_num(target.amplitude, nan=0.0)
        target_ibi = np.nan_to_num(target.ibi, nan=0.0)
        target_cv = np.nan_to_num(target.cv, nan=0.0)

        amp_scale = _safe_scale(target_amp, 0.1)
        ibi_scale = _safe_scale(target_ibi, 1.0)
        cv_scale = _safe_scale(target_cv, 0.1)

        residuals.extend(
            [
                (np.nan_to_num(sim_profile["mean_amplitude"].to_numpy(dtype=float)) - target_amp) / amp_scale,
                (np.nan_to_num(sim_profile["mean_ibi"].to_numpy(dtype=float)) - target_ibi) / ibi_scale,
                cv_weight * ((np.nan_to_num(sim_profile["cv"].to_numpy(dtype=float)) - target_cv) / cv_scale),
            ]
        )

        if count_weight > 0.0 and "num_peaks" in sim_profile.columns:
            target_count = np.nan_to_num(target.count, nan=0.0)
            count_scale = _safe_scale(target_count, 1.0)
            sim_count = np.nan_to_num(sim_profile["num_peaks"].to_numpy(dtype=float))
            residuals.append(
                count_weight * ((sim_count - target_count) / count_scale),
            )

        if rel_amp_weight > 0.0 and np.isfinite(target.rel_residual_amp):
            target_rel = float(target.rel_residual_amp)
            if "rel_residual_amp" in sim_profile.columns:
                sim_rel_vals = sim_profile["rel_residual_amp"].to_numpy(dtype=float)
                sim_rel_vals = sim_rel_vals[np.isfinite(sim_rel_vals)]
                sim_rel = float(np.mean(sim_rel_vals)) if sim_rel_vals.size else 0.0
            else:
                sim_rel = 0.0
            scale = max(target_rel, 1e-3)
            residuals.append(
                np.array([rel_amp_weight * (sim_rel - target_rel) / scale]),
            )

        if full_cv_weight > 0.0 and np.isfinite(target.full_signal_cv):
            target_fcv = float(target.full_signal_cv)
            if "full_signal_cv" in sim_profile.columns:
                sim_fcv_vals = sim_profile["full_signal_cv"].to_numpy(dtype=float)
                sim_fcv_vals = sim_fcv_vals[np.isfinite(sim_fcv_vals)]
                sim_fcv = float(np.mean(sim_fcv_vals)) if sim_fcv_vals.size else 0.0
            else:
                sim_fcv = 0.0
            scale = max(abs(target_fcv), 1e-3)
            residuals.append(
                np.array([full_cv_weight * (sim_fcv - target_fcv) / scale]),
            )
    return np.concatenate(residuals)
