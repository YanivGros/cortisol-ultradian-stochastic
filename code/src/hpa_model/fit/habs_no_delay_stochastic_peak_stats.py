"""No-delay stochastic HABS peak-statistic fitting.

This fitter keeps the canonical three-state nonlinear model but fixes
``tau_min = 0``. It is built for stochastic peak-stat objectives: every
candidate parameter vector is scored against the same schedule of replicate
seeds, so the optimizer sees a deterministic Monte Carlo estimate rather than
one lucky or unlucky simulation.

Two stats modes are supported (set via ``fit.stats_mode``):

* ``legacy`` (default): z-scores the full simulated signal and detects peaks
  with a relative prominence factor — the original pipeline.
* ``residual``: fits a two-harmonic circadian baseline to the simulated
  signal, z-scores the residual, then detects peaks with a fixed absolute
  prominence threshold (``fit.prom_sigma``, default 0.3 σ).  Targets are
  loaded from pre-computed ``peak_amplitude_samples.csv`` CSVs produced by
  ``peak_amplitude_direct_rayleigh_fit``, exactly matching the statistics
  shown in Figure 2.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks as _sp_find_peaks

from ..config import dump_yaml
from ..data.registry import PROJECT_ROOT, get_dataset_spec, load_dataset
from ..data.two_harmonic_shift import evaluate_two_harmonic, fit_two_harmonic_params
from ..model.three_state_gr_delay import build_drive
from ..simulate.engine import simulate_trajectory_fit_arrays
from ..analysis.peak_stats.plot_results import plot_config_trajectory_comparison, plot_goodness_of_fit
from .habs_dual_peak_stats import (
    CONFIG_FREE_PARAM_PATHS,
    SITE_PARAM_TO_LOCATION,
    _build_comparison_rows,
    _build_config_trajectory_comparison_frame,
    _build_model,
    _calculate_peak_metrics,
    _config_peak_stat_options,
    _config_runtime_noise_epsilons,
    _config_runtime_noise_locations,
    _failed_simulation_penalty,
    _has_too_many_empty_bins,
    _selected_signals,
    _set_nested,
    _subsample_without_interpolation,
    _aggregate_config_simulation_stats,
    load_and_calculate_targets,
)
from .objectives import HabsTargets, SignalTargets, build_residual_vector


ALLOWED_FREE_PARAMS = {"kgr", "epsilon", "baseline", *SITE_PARAM_TO_LOCATION}
DEFAULT_FREE_PARAMS = ("kgr", "epsilon_x1", "epsilon_x2", "epsilon_x3")

# Bin edges/labels matching Figure 2 (merged 16-24 evening bin) and the
# peak_amplitude_direct_rayleigh_fit pipeline.
_RESIDUAL_BIN_EDGES: list[float] = [0.0, 240.0, 480.0, 720.0, 960.0, 1440.0]
_RESIDUAL_BIN_LABELS: list[str] = ["00-04", "04-08", "08-12", "12-16", "16-24"]

# Default amplitude column for residual-mode targets. The new canonical metric
# is the peak − previous-trough difference in within-subject Z-score units.
DEFAULT_AMP_COL: str = "peak_amplitude_prev_dip_sigma"


# ---------------------------------------------------------------------------
# Residual-mode helpers (new pipeline matching Figure 2 / peak_amplitude CSV)
# ---------------------------------------------------------------------------

def _binned_stats_residual(
    peak_times: np.ndarray,
    peak_amps: np.ndarray,
    *,
    bin_edges: list[float],
    bin_labels: list[str],
) -> pd.DataFrame:
    """Per-bin amplitude mean, CV, and IPI for one series' peaks.

    Matches Figure 2 exactly:
    - Amplitude: mean of positive peak_amps per bin (min 1 peak)
    - CV: std(ddof=1)/mean per bin (min 2 peaks)
    - IPI: diffs of consecutive peak times, assigned to the FIRST peak's bin
    """
    empty = pd.DataFrame(
        [{"bin": b, "num_peaks": 0.0, "mean_amplitude": np.nan, "cv": np.nan, "mean_ibi": np.nan}
         for b in bin_labels]
    )
    if peak_times.size == 0:
        return empty

    tod = peak_times % 1440.0
    bins_cut = pd.cut(
        pd.Series(tod),
        bins=bin_edges,
        labels=bin_labels,
        right=True,
        include_lowest=True,
    )

    # IPI: diffs assigned to first-peak bin
    sorted_idx = np.argsort(peak_times)
    s_times = peak_times[sorted_idx]
    ipi_by_bin: dict[str, list[float]] = {b: [] for b in bin_labels}
    if s_times.size >= 2:
        ipis = np.diff(s_times)
        first_tods = s_times[:-1] % 1440.0
        first_bins = pd.cut(
            pd.Series(first_tods),
            bins=bin_edges,
            labels=bin_labels,
            right=True,
            include_lowest=True,
        )
        for ipi, fb in zip(ipis, first_bins):
            if ipi > 0 and fb is not None and not pd.isna(fb):
                ipi_by_bin[str(fb)].append(float(ipi))

    rows = []
    for label in bin_labels:
        mask = (bins_cut == label).to_numpy()
        amps = peak_amps[mask]
        amps = amps[np.isfinite(amps) & (amps > 0)]
        n = len(amps)
        mean_amp = float(np.mean(amps)) if n >= 1 else np.nan
        cv = float(amps.std(ddof=1) / amps.mean()) if n >= 2 else np.nan
        ipi_list = ipi_by_bin[label]
        rows.append({
            "bin": label,
            "num_peaks": float(n),
            "mean_amplitude": mean_amp,
            "cv": cv,
            "mean_ibi": float(np.mean(ipi_list)) if ipi_list else np.nan,
        })

    return pd.DataFrame(rows)


def _build_signal_targets_residual(
    peaks_df: pd.DataFrame,
    *,
    bin_edges: list[float],
    bin_labels: list[str],
    signal_name: str,
    amp_col: str = DEFAULT_AMP_COL,
    full_signal_cv_target: float = float("nan"),
) -> SignalTargets:
    """Build a SignalTargets by averaging per-subject per-bin stats."""
    if amp_col not in peaks_df.columns:
        raise KeyError(
            f"Peaks CSV is missing required amplitude column {amp_col!r}; "
            "regenerate it with peak_amplitude_direct_rayleigh_fit."
        )
    per_subject: list[pd.DataFrame] = []
    per_subject_rel: list[float] = []
    for _uid, grp in peaks_df.groupby("series_uid", sort=True):
        times = grp["time_min"].to_numpy(float)
        amps = grp[amp_col].to_numpy(float)
        sort_idx = np.argsort(times)
        df = _binned_stats_residual(
            times[sort_idx], amps[sort_idx],
            bin_edges=bin_edges, bin_labels=bin_labels,
        )
        per_subject.append(df)
        # Relative residual amplitude = std(residual_raw) / mean(baseline_raw).
        # residual_std_raw is constant within a subject; baseline_raw varies.
        if "residual_std_raw" in grp.columns and "baseline_raw" in grp.columns:
            rs = float(grp["residual_std_raw"].iloc[0])
            mb = float(np.nanmean(grp["baseline_raw"].to_numpy(float)))
            if np.isfinite(rs) and np.isfinite(mb) and abs(mb) > 0:
                per_subject_rel.append(rs / abs(mb))

    if not per_subject:
        raise ValueError(f"No peaks found for signal {signal_name!r}")
    rel_residual_amp = (
        float(np.mean(per_subject_rel)) if per_subject_rel else float("nan")
    )

    full_df = pd.concat(per_subject, ignore_index=True)
    grouped = full_df.groupby("bin", sort=False)
    avg = grouped.mean(numeric_only=True).reindex(bin_labels)
    sem = grouped.sem(numeric_only=True).reindex(bin_labels)

    return SignalTargets(
        signal=signal_name,
        value_col=amp_col,
        bins=bin_labels,
        count=avg["num_peaks"].to_numpy(dtype=float),
        amplitude=avg["mean_amplitude"].to_numpy(dtype=float),
        ibi=avg["mean_ibi"].to_numpy(dtype=float),
        cv=avg["cv"].to_numpy(dtype=float),
        amplitude_sem=np.nan_to_num(sem["mean_amplitude"].to_numpy(dtype=float)),
        ibi_sem=np.nan_to_num(sem["mean_ibi"].to_numpy(dtype=float)),
        cv_sem=np.nan_to_num(sem["cv"].to_numpy(dtype=float)),
        rel_residual_amp=rel_residual_amp,
        full_signal_cv=float(full_signal_cv_target),
    )


def _load_targets_residual(
    cortisol_csv: Path,
    acth_csv: Path | None,
    *,
    habs_variant: str,
    bin_edges: list[float],
    bin_labels: list[str],
    signal_mode: str = "both",
    amp_col: str = DEFAULT_AMP_COL,
    full_signal_cv_target: float = float("nan"),
) -> tuple[HabsTargets, pd.DataFrame]:
    """Load residual-mode targets from pre-computed peak CSVs.

    ``signal_mode`` controls which signals are included:
    - ``"both"`` (default): load cortisol and ACTH targets
    - ``"cortisol"``: load only cortisol targets (acth_csv is ignored)

    Returns (HabsTargets, habs_frame) where habs_frame is the HABS shifted
    data used for simulation drive and trajectory comparison.
    """
    cortisol_df = pd.read_csv(cortisol_csv)

    signals: dict[str, Any] = {
        "Cortisol": _build_signal_targets_residual(
            cortisol_df, bin_edges=bin_edges, bin_labels=bin_labels,
            signal_name="Cortisol", amp_col=amp_col,
            full_signal_cv_target=full_signal_cv_target,
        ),
    }

    if signal_mode == "both":
        if acth_csv is None:
            raise ValueError("acth_csv is required when signal_mode='both'")
        acth_df = pd.read_csv(acth_csv)
        signals["ACTH"] = _build_signal_targets_residual(
            acth_df, bin_edges=bin_edges, bin_labels=bin_labels,
            signal_name="ACTH", amp_col=amp_col,
        )

    spec = get_dataset_spec("habs")
    habs_frame = load_dataset("habs", habs_variant)

    return HabsTargets(
        frame=habs_frame,
        id_col=spec.id_col,
        time_col=spec.time_col,
        signals=signals,
    ), habs_frame


def _series_list_from_peaks_csv(peaks_csv: Path) -> list[tuple[str, Any]]:
    """Return deduplicated (dataset_name, series_id) pairs from a peaks CSV."""
    df = pd.read_csv(peaks_csv, usecols=["dataset", "series_id"])
    seen: set[tuple[str, Any]] = set()
    pairs: list[tuple[str, Any]] = []
    for dataset_name, series_id in df.itertuples(index=False):
        key = (str(dataset_name), series_id)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _simulate_residual_stats_for_series(
    config: dict[str, Any],
    *,
    series_id: int | str,
    peak_options: dict[str, Any],
    seed: int,
    dataset_name: str | None = None,
    signals: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Simulate one series and compute residual-mode binned peak stats.

    Pipeline: simulate → fit 24h+12h baseline → z-score residual →
    find_peaks(prominence=prom_sigma) → _binned_stats_residual.

    ``dataset_name`` overrides the dataset in drive params (for multi-dataset mode).
    ``signals`` restricts which signals are computed (e.g. ``("Cortisol",)``).
    """
    model = _build_model(config)
    drive_params = dict(config["drive"]["params"])
    # Only inject the per-subject shift-row lookup when the config opts in
    # (i.e. the YAML already specifies `drive.params.dataset`). Otherwise
    # treat the drive as a global / population-level shape and leave the
    # explicit (a24, phase24, a12, phase12, baseline) literals untouched.
    if "dataset" in drive_params:
        drive_params["series_id"] = series_id
        if dataset_name is not None:
            drive_params["dataset"] = dataset_name
    drive = build_drive(str(config["drive"]["kind"]), drive_params)
    solver = config["solver"]
    traj = simulate_trajectory_fit_arrays(
        model, drive,
        dt_min=float(solver["dt_min"]),
        warmup_min=float(solver["warmup_min"]),
        duration_min=float(solver["duration_min"]),
        seed=seed,
        noise_locations=_config_runtime_noise_locations(config),
        noise_epsilons=_config_runtime_noise_epsilons(config),
        noise_form=str(config.get("runtime", {}).get("noise_form", "multiplicative")),
    )
    traj_time = traj["time_min"]
    traj_x2 = traj["x2"]
    traj_x3 = traj["x3"]

    subsample_min = float(peak_options["subsample_min"])
    duration_min = float(solver["duration_min"])
    if subsample_min > 0:
        time_min, x2 = _subsample_without_interpolation(traj_time, traj_x2, subsample_min=subsample_min, duration_min=duration_min)
        _, x3 = _subsample_without_interpolation(traj_time, traj_x3, subsample_min=subsample_min, duration_min=duration_min)
    else:
        time_min, x2, x3 = traj_time, traj_x2, traj_x3

    prom_sigma = float(peak_options.get("prom_sigma", 0.3))
    min_distance_min = float(peak_options["min_distance_min"])
    bin_edges: list[float] = list(peak_options["bin_edges"])
    bin_labels: list[str] = list(peak_options["bin_labels"])

    dt = float(np.median(np.diff(time_min))) if time_min.size > 1 else 1.0
    min_dist_samples = max(1, int(round(min_distance_min / dt)))

    empty_df = pd.DataFrame(
        [{"bin": b, "num_peaks": 0.0, "mean_amplitude": np.nan,
          "cv": np.nan, "mean_ibi": np.nan, "rel_residual_amp": np.nan,
          "full_signal_cv": np.nan}
         for b in bin_labels]
    )
    all_signal_pairs = (("ACTH", x2), ("Cortisol", x3))
    active_pairs = [(s, v) for s, v in all_signal_pairs if signals is None or s in signals]

    result: dict[str, pd.DataFrame] = {}
    for signal_name, values in active_pairs:
        finite = np.isfinite(values)
        if int(np.sum(finite)) < 10:
            result[signal_name] = empty_df.copy()
            continue

        params = fit_two_harmonic_params(time_min, values, period_min=1440.0, second_period_min=720.0)
        if params is not None:
            baseline = evaluate_two_harmonic(time_min, params)
        else:
            baseline = np.full_like(values, float(np.nanmean(values)))

        residual = values - baseline
        res_std = float(np.nanstd(residual))
        if res_std <= 0 or not np.isfinite(res_std):
            result[signal_name] = empty_df.copy()
            continue
        residual_z = (residual - float(np.nanmean(residual))) / res_std

        # Relative residual amplitude (raw scale, unitless): std(residual) /
        # |mean(baseline)|. Same definition as on the data side.
        baseline_mean = float(np.nanmean(baseline))
        sim_rel_amp = (res_std / abs(baseline_mean)
                       if np.isfinite(baseline_mean) and abs(baseline_mean) > 0
                       else float("nan"))

        # Full-signal CV: std(full signal) / |mean(full signal)| over the whole
        # trace (raw, baseline-inclusive). Matches the data-side target.
        sig_mean = float(np.nanmean(values))
        sig_std = float(np.nanstd(values))
        full_signal_cv = (sig_std / abs(sig_mean)
                          if np.isfinite(sig_mean) and abs(sig_mean) > 0
                          else float("nan"))

        peaks, _ = _sp_find_peaks(residual_z, distance=min_dist_samples, prominence=prom_sigma)
        if peaks.size == 0:
            result[signal_name] = empty_df.copy()
            continue

        # Match the data-side metric: amplitude = peak Z − previous-trough Z,
        # where the previous trough is the minimum between this peak and the
        # prior peak (or the start of the trace for the first peak).
        prev_dip_z = np.empty(peaks.size, dtype=float)
        for i, p in enumerate(peaks):
            lo = 0 if i == 0 else int(peaks[i - 1])
            prev_dip_z[i] = float(residual_z[lo:p].min()) if p > lo else 0.0
        amp_sigma = residual_z[peaks] - prev_dip_z

        binned = _binned_stats_residual(
            time_min[peaks], amp_sigma,
            bin_edges=bin_edges, bin_labels=bin_labels,
        )
        binned["rel_residual_amp"] = sim_rel_amp
        binned["full_signal_cv"] = full_signal_cv
        result[signal_name] = binned

    return result


def _aggregate_residual_simulation_stats(
    config: dict[str, Any],
    *,
    series_ids: list[int],
    peak_options: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Same interface as _aggregate_config_simulation_stats, residual mode."""
    rep_stats: dict[str, list[pd.DataFrame]] = {"ACTH": [], "Cortisol": []}
    for rep in range(int(peak_options["n_reps"])):
        rep_seed = seed + rep
        for series_idx, series_id in enumerate(series_ids):
            signal_frames = _simulate_residual_stats_for_series(
                config,
                series_id=int(series_id),
                peak_options=peak_options,
                seed=rep_seed + series_idx * 1000,
            )
            for signal, df in signal_frames.items():
                if not df.empty and "bin" in df.columns:
                    rep_stats[signal].append(df)

    averaged: dict[str, pd.DataFrame] = {}
    for signal, frames in rep_stats.items():
        if not frames:
            averaged[signal] = pd.DataFrame()
            continue
        grouped = pd.concat(frames, ignore_index=True).groupby("bin", sort=False)
        means = grouped.mean(numeric_only=True)
        sems = grouped.sem(numeric_only=True).rename(columns=lambda x: f"{x}_sem")
        averaged[signal] = pd.concat([means, sems], axis=1)
    return averaged


def _aggregate_residual_simulation_stats_multi(
    config: dict[str, Any],
    *,
    series_ids: list[tuple[str, Any]],
    peak_options: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Multi-dataset residual-mode aggregate; ``series_ids`` is a list of (dataset_name, series_id) tuples."""
    active_signals: tuple[str, ...] = tuple(peak_options.get("active_signals", ("ACTH", "Cortisol")))
    rep_stats: dict[str, list[pd.DataFrame]] = {s: [] for s in active_signals}
    for rep in range(int(peak_options["n_reps"])):
        rep_seed = seed + rep
        for series_idx, (dataset_name, series_id) in enumerate(series_ids):
            signal_frames = _simulate_residual_stats_for_series(
                config,
                dataset_name=dataset_name,
                series_id=series_id,
                peak_options=peak_options,
                seed=rep_seed + series_idx * 1000,
                signals=active_signals,
            )
            for signal, df in signal_frames.items():
                if signal in rep_stats and not df.empty and "bin" in df.columns:
                    rep_stats[signal].append(df)

    averaged: dict[str, pd.DataFrame] = {}
    for signal, frames in rep_stats.items():
        if not frames:
            averaged[signal] = pd.DataFrame()
            continue
        grouped = pd.concat(frames, ignore_index=True).groupby("bin", sort=False)
        means = grouped.mean(numeric_only=True)
        sems = grouped.sem(numeric_only=True).rename(columns=lambda x: f"{x}_sem")
        averaged[signal] = pd.concat([means, sems], axis=1)
    return averaged


# ---------------------------------------------------------------------------
# Residual-mode NaN penalty
# ---------------------------------------------------------------------------

def _fill_residual_nan_penalty(
    sim_profiles: dict[str, pd.DataFrame],
    targets: HabsTargets,
    *,
    ibi_fill_multiplier: float = 5.0,
) -> dict[str, pd.DataFrame]:
    """Fix two NaN-handling failure modes in the residual-mode objective.

    Problem A — near-zero noise collapse: NaN sim IBI (no peaks) gets filled
    with 0 in build_residual_vector, which is closer to target than a real but
    wrong IBI of ~500 min, so the optimizer prefers ε→0 solutions.
    Fix: NaN sim IBI → ibi_fill_multiplier × target_ibi when target IS valid.

    Problem B — zero-target explosion: if a target bin has NaN IBI or NaN CV
    (not enough peaks/pairs in the data for that bin), build_residual_vector
    converts NaN → 0 with scale = 1.0 (IBI) or 0.1 (CV). Simulation values
    there can be large, creating residuals of 75+ or 23+ that dominate the
    entire 36-term objective.
    Fix: when target stat is NaN (no data), set sim stat = 0 so the residual
    cancels to zero — equivalent to masking that bin out of the objective.
    """
    filled: dict[str, pd.DataFrame] = {}
    for signal, df in sim_profiles.items():
        if df.empty:
            filled[signal] = df
            continue
        target = targets.signals.get(signal)
        if target is None:
            filled[signal] = df
            continue
        df_copy = df.copy()
        bin_to_t_ibi = dict(zip(target.bins, target.ibi.tolist()))
        bin_to_t_cv  = dict(zip(target.bins, target.cv.tolist()))
        bin_to_t_amp = dict(zip(target.bins, target.amplitude.tolist()))

        for b in df_copy.index:
            b_str = str(b)
            t_ibi = float(bin_to_t_ibi.get(b_str, np.nan))
            t_cv  = float(bin_to_t_cv.get(b_str, np.nan))
            t_amp = float(bin_to_t_amp.get(b_str, np.nan))

            # IBI column
            if "mean_ibi" in df_copy.columns:
                sim_ibi = float(df_copy.loc[b, "mean_ibi"]) if b in df_copy.index else np.nan
                if not (np.isfinite(t_ibi) and t_ibi > 0):
                    df_copy.loc[b, "mean_ibi"] = 0.0          # no target data → suppress penalty
                elif not np.isfinite(sim_ibi):
                    df_copy.loc[b, "mean_ibi"] = t_ibi * ibi_fill_multiplier  # Problem A fix

            # CV column
            if "cv" in df_copy.columns:
                if not (np.isfinite(t_cv) and t_cv > 0):
                    df_copy.loc[b, "cv"] = 0.0                 # no target data → suppress penalty

            # Amplitude column
            if "mean_amplitude" in df_copy.columns:
                if not (np.isfinite(t_amp) and t_amp > 0):
                    df_copy.loc[b, "mean_amplitude"] = 0.0     # no target data → suppress penalty

        filled[signal] = df_copy
    return filled


# ---------------------------------------------------------------------------
# Core fitter functions
# ---------------------------------------------------------------------------

def _resolve_free_params(config: dict[str, Any]) -> list[str]:
    free_params = [str(name) for name in config.get("fit", {}).get("free_params", DEFAULT_FREE_PARAMS)]
    if not free_params:
        raise ValueError("fit.free_params must contain at least one parameter")
    if len(set(free_params)) != len(free_params):
        raise ValueError(f"Duplicate free params are not allowed: {free_params}")
    invalid = set(free_params) - ALLOWED_FREE_PARAMS
    if invalid:
        raise ValueError(f"Unsupported no-delay stochastic free params: {sorted(invalid)}")
    return free_params


def _resolve_bounds(config: dict[str, Any], free_params: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds_cfg = config.get("fit", {}).get("bounds", {})
    lower: list[float] = []
    upper: list[float] = []
    x0: list[float] = []
    for name in free_params:
        if name not in bounds_cfg:
            raise ValueError(f"fit.bounds.{name} is required for no-delay stochastic fitting")
        lo, hi = (float(v) for v in bounds_cfg[name])
        if hi <= lo:
            raise ValueError(f"Invalid bounds for {name}: upper must exceed lower")
        current = _get_param_value(config, name)
        lower.append(lo)
        upper.append(hi)
        x0.append(float(np.clip(current, lo, hi)))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), np.asarray(x0, dtype=float)


def _get_param_value(config: dict[str, Any], name: str) -> float:
    target: Any = config
    for key in CONFIG_FREE_PARAM_PATHS[name]:
        target = target[key]
    return float(target)


def _trial_config(config: dict[str, Any], free_params: list[str], theta: np.ndarray) -> dict[str, Any]:
    trial = deepcopy(config)
    trial["model"]["params"]["tau_min"] = 0.0
    for idx, name in enumerate(free_params):
        _set_nested(trial, CONFIG_FREE_PARAM_PATHS[name], float(theta[idx]))
    return trial


def _peak_options_for_reps(config: dict[str, Any], *, n_reps: int) -> dict[str, Any]:
    config_for_options = deepcopy(config)
    optimizer_cfg = config_for_options.setdefault("fit", {}).setdefault("optimizer", {})
    if optimizer_cfg.get("maxiter") is None:
        optimizer_cfg["maxiter"] = 1
    options = dict(_config_peak_stat_options(config_for_options))
    options["n_reps"] = int(n_reps)
    return options


def _evaluate_config(
    config: dict[str, Any],
    *,
    targets: HabsTargets,
    series_ids: list[int],
    peak_options: dict[str, Any],
    selected_signals: tuple[str, ...],
    seed: int,
    aggregate_sim_stats: Any = None,
    postprocess_sim_profiles: Any = None,
) -> dict[str, Any]:
    if aggregate_sim_stats is None:
        aggregate_sim_stats = _aggregate_config_simulation_stats
    sim_profiles = aggregate_sim_stats(
        config,
        series_ids=series_ids,
        peak_options=peak_options,
        seed=int(seed),
    )
    if postprocess_sim_profiles is not None:
        sim_profiles = postprocess_sim_profiles(sim_profiles)
    if any(sim_profiles[s].empty for s in selected_signals) or _has_too_many_empty_bins(sim_profiles, selected_signals):
        residual_vector = _failed_simulation_penalty(targets, selected_signals)
        metrics = _calculate_peak_metrics(targets, sim_profiles, selected_signals)
    else:
        residual_vector = build_residual_vector(
            targets=targets,
            sim_profiles=sim_profiles,
            selected_signals=selected_signals,
            cv_weight=float(peak_options["cv_weight"]),
            rel_amp_weight=float(peak_options.get("rel_amp_weight", 0.0)),
            full_cv_weight=float(peak_options.get("full_cv_weight", 0.0)),
            count_weight=float(peak_options.get("count_weight", 0.0)),
        )
        metrics = _calculate_peak_metrics(targets, sim_profiles, selected_signals)

    return {
        "sim_profiles": sim_profiles,
        "residual_vector": residual_vector,
        "objective_value": float(np.mean(np.asarray(residual_vector, dtype=float) ** 2)),
        "metrics": metrics,
    }


def _objective(
    theta: np.ndarray,
    *,
    base_config: dict[str, Any],
    free_params: list[str],
    targets: HabsTargets,
    series_ids: list[int],
    peak_options: dict[str, Any],
    selected_signals: tuple[str, ...],
    seed: int,
    aggregate_sim_stats: Any = None,
    postprocess_sim_profiles: Any = None,
) -> float:
    trial = _trial_config(base_config, free_params, np.asarray(theta, dtype=float))
    evaluation = _evaluate_config(
        trial,
        targets=targets,
        series_ids=series_ids,
        peak_options=peak_options,
        selected_signals=selected_signals,
        seed=seed,
        aggregate_sim_stats=aggregate_sim_stats,
        postprocess_sim_profiles=postprocess_sim_profiles,
    )
    return float(evaluation["objective_value"])


def _build_params(config: dict[str, Any], free_params: list[str]) -> dict[str, float]:
    params = {
        "kgr": float(config["model"]["params"]["kgr"]),
        "tau_min": 0.0,
    }
    for name in SITE_PARAM_TO_LOCATION:
        params[name] = float(config.get("runtime", {}).get("noise_epsilons", {}).get(SITE_PARAM_TO_LOCATION[name], 0.0))
    for name in free_params:
        if name not in params:
            params[name] = _get_param_value(config, name)
    return params


def _write_readme(out_dir: Path, summary_row: dict[str, Any], config: dict[str, Any]) -> None:
    free_params = ", ".join(str(v) for v in config["fit"]["free_params"])
    noise_locations = ", ".join(_config_runtime_noise_locations(config))
    stats_mode = str(config.get("fit", {}).get("stats_mode", "legacy"))
    lines = [
        "# No-Delay Stochastic HABS Peak-Statistic Fit",
        "",
        "This run fits the nonlinear three-state HPA model to HABS shifted ACTH/cortisol peak statistics with `tau_min` fixed at 0 min.",
        "",
        "The objective is stochastic-aware: each candidate is evaluated by averaging fixed-seed replicate simulations, and the saved artifacts use the same replicate schedule as the optimizer search.",
        "",
        "Highlights:",
        f"- Optimizer: {config['fit']['optimizer']['name']}",
        f"- Free parameters: {free_params}",
        f"- Noise form: {config['runtime'].get('noise_form', 'multiplicative')}",
        f"- Noise locations: {noise_locations}",
        f"- Stats mode: {stats_mode}",
        f"- Training replicates per candidate: {int(config['fit']['n_reps'])}",
        f"- Artifact evaluation replicates: {int(summary_row['final_n_reps'])}",
        f"- Final objective value: {float(summary_row['objective_value']):.6g}",
        "",
        "Artifacts:",
        "- `artifacts/fit_summary.csv`",
        "- `artifacts/fit_params.csv`",
        "- `artifacts/peak_profile_comparison.csv`",
        "- `artifacts/trajectory_comparison.csv`",
        "- `artifacts/fitted_config.yaml`",
        "",
        "Figures:",
        "- `figures/habs_no_delay_stochastic_goodness_of_fit.png`",
        "- `figures/habs_no_delay_stochastic_trajectory_comparison.png`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def _resolve_maxiter(optimizer_cfg: dict[str, Any]) -> int:
    raw_maxiter = optimizer_cfg.get("maxiter", 16)
    if raw_maxiter is None or str(raw_maxiter).lower() in {"none", "null", "unbounded"}:
        return 1_000_000
    maxiter = int(raw_maxiter)
    if maxiter < 1:
        raise ValueError("fit.optimizer.maxiter must be positive, null, or 'unbounded'")
    return maxiter


def _resolve_workers(optimizer_cfg: dict[str, Any]) -> int:
    workers = int(optimizer_cfg.get("workers", 1))
    if workers == 0 or workers < -1:
        raise ValueError("fit.optimizer.workers must be positive or -1 for all available workers")
    return workers


def _bounded_polish(
    theta: np.ndarray,
    *,
    objective: Any,
    bounds: list[tuple[float, float]],
    optimizer_cfg: dict[str, Any],
    logger: Any | None,
) -> Any:
    polish_maxiter = int(optimizer_cfg.get("polish_maxiter", 30))
    polish_maxfun = int(optimizer_cfg.get("polish_maxfun", max(20, polish_maxiter * (len(theta) + 1))))
    if polish_maxiter < 1 or polish_maxfun < 1:
        raise ValueError("fit.optimizer.polish_maxiter and polish_maxfun must be positive")
    if logger is not None:
        logger.info(
            "Starting bounded L-BFGS-B polish maxiter=%d maxfun=%d",
            polish_maxiter,
            polish_maxfun,
        )
    result = minimize(
        objective,
        np.asarray(theta, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": polish_maxiter,
            "maxfun": polish_maxfun,
            "ftol": float(optimizer_cfg.get("polish_ftol", 1e-6)),
        },
    )
    if logger is not None:
        logger.info(
            "Completed bounded polish: success=%s fun=%.6g nfev=%s nit=%s message=%s",
            bool(result.success),
            float(result.fun),
            getattr(result, "nfev", "NA"),
            getattr(result, "nit", "NA"),
            str(result.message),
        )
    return result


def fit_habs_no_delay_stochastic_peak_stats_from_config(
    config: dict[str, Any],
    out_dir: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    if logger is not None:
        logger.info("Fitting no-delay stochastic HABS peak statistics")

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / "artifacts"
    figure_dir = out_dir / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fit_cfg = config["fit"]
    optimizer_cfg = dict(fit_cfg.get("optimizer", {}))
    free_params = _resolve_free_params(config)
    lower, upper, x0 = _resolve_bounds(config, free_params)
    peak_options = _peak_options_for_reps(config, n_reps=int(fit_cfg.get("n_reps", 8)))

    stats_mode = str(fit_cfg.get("stats_mode", "legacy"))
    signal_mode = str(fit_cfg.get("signal_mode", "both"))  # "both" or "cortisol"

    if stats_mode == "residual":
        cortisol_csv = Path(fit_cfg["cortisol_peaks_csv"])
        if not cortisol_csv.is_absolute():
            cortisol_csv = PROJECT_ROOT / cortisol_csv

        if signal_mode == "cortisol":
            # Multi-dataset cortisol-only mode: targets and series come from the peaks CSV
            acth_csv = None
            targets, frame = _load_targets_residual(
                cortisol_csv, acth_csv,
                habs_variant=str(config["dataset"]["variant"]),
                bin_edges=_RESIDUAL_BIN_EDGES,
                bin_labels=_RESIDUAL_BIN_LABELS,
                signal_mode="cortisol",
                full_signal_cv_target=float(
                    fit_cfg.get("loss", {}).get("full_signal_cv_target", float("nan"))
                ),
            )
            series_ids = _series_list_from_peaks_csv(cortisol_csv)
            peak_options["active_signals"] = ("Cortisol",)
            aggregate_sim_stats_fn = _aggregate_residual_simulation_stats_multi
            if logger is not None:
                logger.info(
                    "Residual cortisol-only mode: %d series across %d datasets from %s",
                    len(series_ids),
                    len({ds for ds, _ in series_ids}),
                    cortisol_csv,
                )
        else:
            acth_csv = Path(fit_cfg["acth_peaks_csv"])
            if not acth_csv.is_absolute():
                acth_csv = PROJECT_ROOT / acth_csv
            targets, frame = _load_targets_residual(
                cortisol_csv, acth_csv,
                habs_variant=str(config["dataset"]["variant"]),
                bin_edges=_RESIDUAL_BIN_EDGES,
                bin_labels=_RESIDUAL_BIN_LABELS,
                signal_mode="both",
            )
            series_ids = [int(sid) for sid in sorted(frame[targets.id_col].dropna().unique().tolist())]
            aggregate_sim_stats_fn = _aggregate_residual_simulation_stats
            if logger is not None:
                logger.info(
                    "Residual stats mode: cortisol=%s acth=%s",
                    cortisol_csv, acth_csv,
                )

        # Common residual-mode peak options
        peak_options["prom_sigma"] = float(fit_cfg.get("prom_sigma", 0.3))
        peak_options["bin_edges"] = _RESIDUAL_BIN_EDGES
        peak_options["bin_labels"] = _RESIDUAL_BIN_LABELS
        ibi_nan_fill = float(fit_cfg.get("ibi_nan_fill", 5.0))
        postprocess_sim_profiles_fn = partial(
            _fill_residual_nan_penalty,
            targets=targets,
            ibi_fill_multiplier=ibi_nan_fill,
        )
        if logger is not None:
            logger.info("prom_sigma=%.2f ibi_nan_fill=%.1f×target", peak_options["prom_sigma"], ibi_nan_fill)
    else:
        targets, frame = load_and_calculate_targets(
            variant=str(config["dataset"]["variant"]),
            bin_size_min=float(peak_options["bin_size_min"]),
            min_distance_min=float(peak_options["min_distance_min"]),
            acth_prom_factor=float(peak_options["acth_prom_factor"]),
            cortisol_prom_factor=float(peak_options["cortisol_prom_factor"]),
        )
        series_ids = [int(sid) for sid in sorted(frame[targets.id_col].dropna().unique().tolist())]
        aggregate_sim_stats_fn = _aggregate_config_simulation_stats
        postprocess_sim_profiles_fn = None

    selected_signals = ("Cortisol",) if signal_mode == "cortisol" else _selected_signals("both")

    base_config = deepcopy(config)
    base_config["model"]["params"]["tau_min"] = 0.0
    base_config["model"]["free_params"] = []
    base_config["fit"]["free_params"] = free_params

    optimizer_name = str(optimizer_cfg.get("name", "differential_evolution"))
    if optimizer_name != "differential_evolution":
        raise ValueError("fit_habs_no_delay_stochastic_peak_stats currently requires optimizer.name=differential_evolution")
    bounds = list(zip(lower, upper, strict=True))
    seed = int(config["runtime"]["seed"])
    maxiter = _resolve_maxiter(optimizer_cfg)
    tol = float(optimizer_cfg.get("tol", 0.01))
    atol = float(optimizer_cfg.get("atol", 0.0))
    workers = _resolve_workers(optimizer_cfg)
    should_polish = bool(optimizer_cfg.get("polish", True))
    log_every = max(1, int(optimizer_cfg.get("log_every", 1)))
    generation_counter = {"value": 0}

    def _callback(xk: np.ndarray, convergence: float) -> bool:
        del xk
        generation_counter["value"] += 1
        generation = generation_counter["value"]
        if logger is not None and generation % log_every == 0:
            logger.info("DE generation %d convergence=%.6g", generation, float(convergence))
        return False

    objective = partial(
        _objective,
        base_config=base_config,
        free_params=free_params,
        targets=targets,
        series_ids=series_ids,
        peak_options=peak_options,
        selected_signals=selected_signals,
        seed=seed,
        aggregate_sim_stats=aggregate_sim_stats_fn,
        postprocess_sim_profiles=postprocess_sim_profiles_fn,
    )
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=int(optimizer_cfg.get("popsize", 6)),
        polish=False,
        seed=seed,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
        x0=x0,
        tol=tol,
        atol=atol,
        callback=_callback,
    )

    best_theta = np.asarray(result.x, dtype=float)
    best_fun = float(result.fun)
    polish_result = None
    if should_polish:
        polish_result = _bounded_polish(
            best_theta,
            objective=objective,
            bounds=bounds,
            optimizer_cfg=optimizer_cfg,
            logger=logger,
        )
        if np.isfinite(float(polish_result.fun)) and float(polish_result.fun) <= best_fun:
            best_theta = np.asarray(polish_result.x, dtype=float)
            best_fun = float(polish_result.fun)

    fitted_config = _trial_config(base_config, free_params, best_theta)
    search_evaluation = _evaluate_config(
        fitted_config,
        targets=targets,
        series_ids=series_ids,
        peak_options=peak_options,
        selected_signals=selected_signals,
        seed=seed,
        aggregate_sim_stats=aggregate_sim_stats_fn,
        postprocess_sim_profiles=postprocess_sim_profiles_fn,
    )
    sim_profiles = search_evaluation["sim_profiles"]
    comparison_rows = _build_comparison_rows(targets, sim_profiles, selected_signals)
    peak_profile_frame = pd.concat(comparison_rows, ignore_index=True)
    trajectory_frame = _build_config_trajectory_comparison_frame(fitted_config, frame=frame, seed=seed)

    params = _build_params(fitted_config, free_params)
    summary_row = {
        "dataset": str(fitted_config["dataset"]["name"]),
        "variant": str(fitted_config["dataset"]["variant"]),
        "signal_mode": signal_mode,
        "stats_mode": stats_mode,
        "model_variant": "three_state_gr_delay_tau_fixed_0",
        "optimizer": optimizer_name,
        "success": bool(result.success),
        "message": str(result.message),
        "optimizer_objective_value": float(result.fun),
        "best_objective_value": float(best_fun),
        "objective_value": float(search_evaluation["objective_value"]),
        "nfev": float(result.nfev),
        "nit": float(result.nit),
        "maxiter": int(maxiter),
        "tol": float(tol),
        "atol": float(atol),
        "workers": int(workers),
        "polish": bool(should_polish),
        "polish_nfev": float(getattr(polish_result, "nfev", 0) if polish_result is not None else 0),
        "polish_nit": float(getattr(polish_result, "nit", 0) if polish_result is not None else 0),
        "polish_success": bool(getattr(polish_result, "success", False) if polish_result is not None else False),
        "training_n_reps": int(peak_options["n_reps"]),
        "final_n_reps": int(peak_options["n_reps"]),
        "noise_form": str(fitted_config.get("runtime", {}).get("noise_form", "multiplicative")),
        "noise_locations": ",".join(_config_runtime_noise_locations(fitted_config)),
        **params,
        **search_evaluation["metrics"],
    }

    pd.DataFrame([summary_row]).to_csv(artifact_dir / "fit_summary.csv", index=False)
    pd.DataFrame([{"parameter": name, "value": value} for name, value in params.items()]).to_csv(
        artifact_dir / "fit_params.csv",
        index=False,
    )
    peak_profile_frame.to_csv(artifact_dir / "peak_profile_comparison.csv", index=False)
    trajectory_frame.to_csv(artifact_dir / "trajectory_comparison.csv", index=False)
    (artifact_dir / "fitted_config.yaml").write_text(dump_yaml(fitted_config))

    plot_goodness_of_fit(
        targets,
        sim_profiles,
        selected_signals=selected_signals,
        out_path=figure_dir / "habs_no_delay_stochastic_goodness_of_fit.png",
        title="No-delay stochastic HABS peak-stat fit",
    )
    plot_config_trajectory_comparison(
        trajectory_frame,
        out_path=figure_dir / "habs_no_delay_stochastic_trajectory_comparison.png",
        title="No-delay stochastic HABS trajectory comparison",
    )
    _write_readme(out_dir, summary_row, fitted_config)

    if logger is not None:
        logger.info(
            "Completed no-delay stochastic fit: objective=%.4f kgr=%.3f eps=(%.3f, %.3f, %.3f)",
            float(summary_row["objective_value"]),
            float(params["kgr"]),
            float(params.get("epsilon_x1", 0.0)),
            float(params.get("epsilon_x2", 0.0)),
            float(params.get("epsilon_x3", 0.0)),
        )

    return {
        "summary_row": summary_row,
        "params": params,
        "metrics": search_evaluation["metrics"],
        "targets": targets,
        "sim_profiles": sim_profiles,
        "peak_profile_frame": peak_profile_frame,
        "trajectory_frame": trajectory_frame,
        "fitted_config": fitted_config,
    }


def _run_direct() -> None:
    import argparse
    import json
    import platform
    import subprocess

    parser = argparse.ArgumentParser(description="Fit no-delay stochastic HABS peak statistics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = deepcopy(__import__("yaml").safe_load(args.config.read_text()))
    result = fit_habs_no_delay_stochastic_peak_stats_from_config(config, args.out)
    manifest = {
        "task": "fit_habs_no_delay_stochastic_peak_stats",
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(args.config.resolve()),
        "run_dir": str(args.out.resolve()),
        "python_version": platform.python_version(),
        "seed": int(config["runtime"]["seed"]),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dataset": config["dataset"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (args.out / "resolved_config.yaml").write_text(dump_yaml(result["fitted_config"]))


if __name__ == "__main__":
    _run_direct()
