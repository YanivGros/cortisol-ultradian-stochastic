"""HABS-only peak-statistic fitting for ACTH and cortisol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
import json
import logging
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "hpa_model-mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

try:
    import emcee
except ImportError:
    emcee = None

from ..config import dump_yaml
from ..data.registry import get_dataset_spec, load_dataset
from ..model.three_state_gr_delay import ThreeStateGRDelayModel, build_drive
from ..simulate.engine import sample_trajectory, simulate_trajectory, simulate_trajectory_fit_arrays
from .objectives import HabsTargets, SignalTargets, build_residual_vector
from ..analysis.peak_stats.plot_results import (
    plot_config_trajectory_comparison,
    plot_goodness_of_fit,
    plot_trajectory_comparison,
)
from .individual_datasets_new_model import (
    BIN_SIZE_MIN,
    DEFAULT_PHASE_MIN,
    DEFAULT_VARIANT,
    MIN_DISTANCE_MIN,
    N_REPS,
    PROM_FACTOR_CORTISOL,
    SIM_DURATION_MIN,
    SUBSAMPLE_MIN,
    WARMUP_MIN,
    _safe_scale,
    _subsample_without_interpolation,
    _zscore,
    calculate_binned_stats,
)


DEFAULT_PROM_FACTOR_ACTH = 0.15
DEFAULT_NOISE_MODE = "multiplicative"
FIT_PARAM_NAMES = ("drive_baseline", "drive_amplitude", "kgr", "tau_min", "epsilon")
CONFIG_FIT_PARAM_NAMES = ("kgr", "tau_min", "epsilon")
CONFIG_FREE_PARAM_PATHS: dict[str, tuple[str, ...]] = {
    "kgr": ("model", "params", "kgr"),
    "tau_min": ("model", "params", "tau_min"),
    "epsilon": ("drive", "params", "epsilon"),
    "baseline": ("drive", "params", "baseline"),
    "epsilon_x1": ("runtime", "noise_epsilons", "x1_secretion"),
    "epsilon_x2": ("runtime", "noise_epsilons", "x2_secretion"),
    "epsilon_x3": ("runtime", "noise_epsilons", "x3_secretion"),
}
SITE_PARAM_TO_LOCATION = {
    "epsilon_x1": "x1_secretion",
    "epsilon_x2": "x2_secretion",
    "epsilon_x3": "x3_secretion",
}
LOCATION_TO_SITE_PARAM = {value: key for key, value in SITE_PARAM_TO_LOCATION.items()}
ALLOWED_CONFIG_FREE_PARAMS = {"kgr", "tau_min", "epsilon", *SITE_PARAM_TO_LOCATION}


@dataclass(frozen=True)
class FitSettings:
    variant: str = DEFAULT_VARIANT
    signal_mode: str = "both"
    noise_location: str = "drive"
    n_reps: int = N_REPS
    max_nfev: int = 24
    seed: int = 42
    noise_mode: str = DEFAULT_NOISE_MODE
    bin_size_min: float = BIN_SIZE_MIN
    subsample_min: float = SUBSAMPLE_MIN
    duration_min: float = SIM_DURATION_MIN
    warmup_min: float = WARMUP_MIN
    min_distance_min: float = MIN_DISTANCE_MIN
    acth_prom_factor: float = DEFAULT_PROM_FACTOR_ACTH
    cortisol_prom_factor: float = PROM_FACTOR_CORTISOL
    phase_min: float = DEFAULT_PHASE_MIN
    cv_weight: float = 1.0
    kgr_max: float = 100.0
    tau_max_min: float = 180.0
    epsilon_max: float = 25.0



def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model.habs_peak_stats")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _signal_columns(signal: str) -> tuple[str, str]:
    if signal == "ACTH":
        return "ACTH", "acth"
    if signal == "Cortisol":
        return "Cortisol", "cortisol"
    raise KeyError(f"Unsupported signal: {signal}")


def _selected_signals(signal_mode: str) -> tuple[str, ...]:
    if signal_mode == "both":
        return ("ACTH", "Cortisol")
    if signal_mode == "acth":
        return ("ACTH",)
    if signal_mode == "cortisol":
        return ("Cortisol",)
    raise ValueError(f"Unsupported signal_mode: {signal_mode}")



def _default_run_config(settings: FitSettings) -> dict[str, Any]:
    model = ThreeStateGRDelayModel()
    drive_kind = "sine_noise" if settings.noise_location == "drive" else "sine"
    drive_params: dict[str, Any] = {
        "baseline": 1.0,
        "amplitude": 0.25,
        "phase_min": settings.phase_min,
        "period_min": 1440.0,
    }
    if settings.noise_location == "drive":
        drive_params["epsilon"] = 0.05
    return {
        "task": "fit_habs_dual_peak_stats",
        "dataset": {"name": "habs", "variant": settings.variant},
        "signal_mode": settings.signal_mode,
        "model": {
            "params": {
                **model.to_params_dict(),
            },
            "free_params": list(FIT_PARAM_NAMES),
        },
        "drive": {
            "kind": drive_kind,
            "params": drive_params,
        },
        "solver": {
            "dt_min": 1.0,
            "warmup_min": settings.warmup_min,
            "duration_min": settings.duration_min,
        },
        "fit": {
            "bin_size_min": settings.bin_size_min,
            "subsample_min": settings.subsample_min,
            "min_distance_min": settings.min_distance_min,
            "acth_prom_factor": settings.acth_prom_factor,
            "cortisol_prom_factor": settings.cortisol_prom_factor,
            "cv_weight": settings.cv_weight,
            "n_reps": settings.n_reps,
            "max_nfev": settings.max_nfev,
            "bounds": {
                "kgr_max": settings.kgr_max,
                "tau_max_min": settings.tau_max_min,
                "epsilon_max": settings.epsilon_max,
            },
            "loss": {"mode": "peak_stats"},
        },
        "runtime": {
            "seed": settings.seed,
            "noise_mode": settings.noise_mode,
            "noise_location": settings.noise_location,
            "noise_form": "multiplicative",
        },
    }


def _signal_prom_factor(signal: str, settings: FitSettings) -> float:
    if signal == "ACTH":
        return float(settings.acth_prom_factor)
    if signal == "Cortisol":
        return float(settings.cortisol_prom_factor)
    raise KeyError(signal)


def _infer_value_column(frame: pd.DataFrame, signal: str) -> str:
    value_col, _ = _signal_columns(signal)
    if value_col not in frame.columns:
        raise KeyError(f"Missing column {value_col} for {signal}")
    return value_col


def _build_signal_targets(
    frame: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    signal: str,
    bin_size_min: float,
    min_distance_min: float,
    prom_factor: float,
) -> SignalTargets:
    value_col = _infer_value_column(frame, signal)
    all_dfs: list[pd.DataFrame] = []

    for _, group in frame.groupby(id_col, sort=False):
        time = group[time_col].to_numpy(dtype=float)
        values = group[value_col].to_numpy(dtype=float)
        if len(values) < 10:
            continue
        bin_df = calculate_binned_stats(
            time,
            values,
            bin_size_min=bin_size_min,
            prom_factor=prom_factor,
            min_distance_min=min_distance_min,
        )
        if not bin_df.empty and "bin" in bin_df.columns:
            all_dfs.append(bin_df)

    if not all_dfs:
        raise ValueError(f"No valid bins calculated for signal {signal}.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    grouped = full_df.groupby("bin", sort=False)
    avg_profile = grouped.mean(numeric_only=True)
    sem_profile = grouped.sem(numeric_only=True)

    return SignalTargets(
        signal=signal,
        value_col=value_col,
        bins=avg_profile.index.tolist(),
        count=avg_profile["num_peaks"].to_numpy(dtype=float),
        amplitude=avg_profile["mean_amplitude"].to_numpy(dtype=float),
        ibi=avg_profile["mean_ibi"].to_numpy(dtype=float),
        cv=avg_profile["cv"].to_numpy(dtype=float),
        amplitude_sem=np.nan_to_num(sem_profile["mean_amplitude"].to_numpy(dtype=float)),
        ibi_sem=np.nan_to_num(sem_profile["mean_ibi"].to_numpy(dtype=float)),
        cv_sem=np.nan_to_num(sem_profile["cv"].to_numpy(dtype=float)),
    )


def load_and_calculate_targets(
    *,
    variant: str = DEFAULT_VARIANT,
    bin_size_min: float = BIN_SIZE_MIN,
    min_distance_min: float = MIN_DISTANCE_MIN,
    acth_prom_factor: float = DEFAULT_PROM_FACTOR_ACTH,
    cortisol_prom_factor: float = PROM_FACTOR_CORTISOL,
) -> tuple[HabsTargets, pd.DataFrame]:
    """Calculate peak-statistic targets for HABS ACTH and cortisol."""
    spec = get_dataset_spec("habs")
    frame = load_dataset("habs", variant)
    signals = {
        "ACTH": _build_signal_targets(
            frame,
            id_col=spec.id_col,
            time_col=spec.time_col,
            signal="ACTH",
            bin_size_min=bin_size_min,
            min_distance_min=min_distance_min,
            prom_factor=acth_prom_factor,
        ),
        "Cortisol": _build_signal_targets(
            frame,
            id_col=spec.id_col,
            time_col=spec.time_col,
            signal="Cortisol",
            bin_size_min=bin_size_min,
            min_distance_min=min_distance_min,
            prom_factor=cortisol_prom_factor,
        ),
    }
    return HabsTargets(frame=frame, id_col=spec.id_col, time_col=spec.time_col, signals=signals), frame


def _build_model(config: dict[str, Any]) -> ThreeStateGRDelayModel:
    params = config["model"]["params"]
    return ThreeStateGRDelayModel(
        a1=float(params["a1"]),
        a2=float(params["a2"]),
        a3=float(params["a3"]),
        b1=float(params["b1"]),
        b2=float(params["b2"]),
        b3=float(params["b3"]),
        kgr=float(params["kgr"]),
        tau_min=float(params["tau_min"]),
        x3_floor=float(params["x3_floor"]),
        hill_coeff=float(params["hill_coeff"]),
        initial_state=tuple(float(x) for x in params["initial_state"]),
    )


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: float) -> None:
    target = config
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = float(value)


def _get_nested(config: dict[str, Any], path: tuple[str, ...]) -> float:
    target: Any = config
    for key in path:
        target = target[key]
    return float(target)


def _config_peak_stat_options(config: dict[str, Any]) -> dict[str, Any]:
    fit_cfg = config["fit"]
    optimizer_cfg = fit_cfg.get("optimizer", {})
    return {
        "bin_size_min": float(fit_cfg.get("bin_size_min", BIN_SIZE_MIN)),
        "subsample_min": float(fit_cfg.get("subsample_min", SUBSAMPLE_MIN)),
        "min_distance_min": float(fit_cfg.get("min_distance_min", MIN_DISTANCE_MIN)),
        "acth_prom_factor": float(fit_cfg.get("acth_prom_factor", DEFAULT_PROM_FACTOR_ACTH)),
        "cortisol_prom_factor": float(fit_cfg.get("cortisol_prom_factor", PROM_FACTOR_CORTISOL)),
        "cv_weight": float(fit_cfg.get("loss", {}).get("cv_weight", 1.0)),
        "rel_amp_weight": float(fit_cfg.get("loss", {}).get("rel_amp_weight", 0.0)),
        "full_cv_weight": float(fit_cfg.get("loss", {}).get("full_signal_cv_weight", 0.0)),
        "count_weight": float(fit_cfg.get("loss", {}).get("count_weight", 0.0)),
        "n_reps": int(fit_cfg.get("n_reps", N_REPS)),
        "max_nfev": int(fit_cfg.get("max_nfev", 24)),
        "optimizer": {
            "name": str(optimizer_cfg.get("name", "least_squares")),
            "maxiter": int(optimizer_cfg.get("maxiter", 6)),
            "popsize": int(optimizer_cfg.get("popsize", 4)),
            "workers": int(optimizer_cfg.get("workers", 1)),
            "polish": bool(optimizer_cfg.get("polish", False)),
        },
    }


def _resolve_config_free_params(config: dict[str, Any]) -> list[str]:
    free_params = list(config["model"].get("free_params", []))
    if not free_params:
        raise ValueError("Expected at least one config free parameter")
    if len(set(free_params)) != len(free_params):
        raise ValueError(f"Duplicate free params are not allowed: {free_params}")
    invalid = set(free_params) - ALLOWED_CONFIG_FREE_PARAMS
    if invalid:
        raise ValueError(f"Unsupported config free params: {sorted(invalid)}")
    if "tau_min" not in free_params:
        raise ValueError("Config free params must include tau_min")
    has_drive_noise = "epsilon" in free_params
    has_secretion_noise = any(name in SITE_PARAM_TO_LOCATION for name in free_params)
    if has_drive_noise and has_secretion_noise:
        raise ValueError("Config free params cannot mix drive epsilon with secretion epsilon parameters")
    return free_params


def _config_runtime_noise_locations(config: dict[str, Any]) -> list[str]:
    return [str(location) for location in config.get("runtime", {}).get("noise_locations", [])]


def _config_runtime_noise_epsilons(config: dict[str, Any]) -> dict[str, float]:
    return {
        str(location): float(value)
        for location, value in config.get("runtime", {}).get("noise_epsilons", {}).items()
    }


def _config_peak_stat_residuals(
    theta: np.ndarray,
    *,
    config: dict[str, Any],
    free_params: list[str],
    series_ids: list[int],
    peak_options: dict[str, Any],
    targets: HabsTargets,
    selected_signals: tuple[str, ...],
) -> np.ndarray:
    trial = deepcopy(config)
    for idx, name in enumerate(free_params):
        _set_nested(trial, CONFIG_FREE_PARAM_PATHS[name], float(theta[idx]))
    sim_profiles = _aggregate_config_simulation_stats(
        trial,
        series_ids=series_ids,
        peak_options=peak_options,
        seed=int(trial["runtime"]["seed"]),
    )
    if any(sim_profiles[s].empty for s in selected_signals):
        return _failed_simulation_penalty(targets, selected_signals)
    if _has_too_many_empty_bins(sim_profiles, selected_signals):
        return _failed_simulation_penalty(targets, selected_signals)
    return build_residual_vector(
        targets=targets,
        sim_profiles=sim_profiles,
        selected_signals=selected_signals,
        cv_weight=float(peak_options["cv_weight"]),
    )


def _config_peak_stat_scalar_objective(
    theta: np.ndarray,
    *,
    config: dict[str, Any],
    free_params: list[str],
    series_ids: list[int],
    peak_options: dict[str, Any],
    targets: HabsTargets,
    selected_signals: tuple[str, ...],
) -> float:
    residual_vec = _config_peak_stat_residuals(
        np.asarray(theta, dtype=float),
        config=config,
        free_params=free_params,
        series_ids=series_ids,
        peak_options=peak_options,
        targets=targets,
        selected_signals=selected_signals,
    )
    return 0.5 * float(np.dot(residual_vec, residual_vec))


def _emcee_log_prior(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if np.all(theta >= lower) and np.all(theta <= upper):
        return 0.0
    return -np.inf


def _emcee_log_prob(
    theta: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    config: dict[str, Any],
    free_params: list[str],
    series_ids: list[int],
    peak_options: dict[str, Any],
    targets: HabsTargets,
    selected_signals: tuple[str, ...],
) -> float:
    lp = _emcee_log_prior(theta, lower, upper)
    if not np.isfinite(lp):
        return -np.inf
    cost = _config_peak_stat_scalar_objective(
        theta,
        config=config,
        free_params=free_params,
        series_ids=series_ids,
        peak_options=peak_options,
        targets=targets,
        selected_signals=selected_signals,
    )
    return lp - cost


def _config_secretion_noise_params(config: dict[str, Any]) -> dict[str, float]:
    noise_epsilons = _config_runtime_noise_epsilons(config)
    return {
        param_name: float(noise_epsilons.get(location, 0.0))
        for param_name, location in SITE_PARAM_TO_LOCATION.items()
    }


def _default_config_bounds(name: str, current: float) -> tuple[float, float]:
    if name == "kgr":
        return (1e-6, max(current * 5.0, 1.0))
    if name == "tau_min":
        return (0.0, max(current * 5.0, 60.0))
    if name == "epsilon" or name in SITE_PARAM_TO_LOCATION:
        return (0.0, max(current * 5.0, 1.0))
    raise KeyError(f"Unsupported config free parameter: {name}")


def _resolve_config_bound(
    name: str,
    current: float,
    bound: tuple[float, float] | list[float] | None,
) -> tuple[float, float, float]:
    lower, upper = _default_config_bounds(name, current) if bound is None else (float(bound[0]), float(bound[1]))
    x0 = min(max(current, lower), upper)
    return float(lower), float(upper), float(x0)


def _simulate_config_signal_stats_for_series(
    config: dict[str, Any],
    *,
    series_id: int,
    peak_options: dict[str, float | int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    model = _build_model(config)
    drive = build_drive(
        str(config["drive"]["kind"]),
        {
            **config["drive"]["params"],
            "series_id": int(series_id),
        },
    )
    solver = config["solver"]
    traj = simulate_trajectory_fit_arrays(
        model,
        drive,
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
        time_min, x2 = _subsample_without_interpolation(
            traj_time,
            traj_x2,
            subsample_min=subsample_min,
            duration_min=duration_min,
        )
        _, x3 = _subsample_without_interpolation(
            traj_time,
            traj_x3,
            subsample_min=subsample_min,
            duration_min=duration_min,
        )
    else:
        time_min = traj_time
        x2 = traj_x2
        x3 = traj_x3

    return {
        "ACTH": calculate_binned_stats(
            time_min,
            x2,
            bin_size_min=float(peak_options["bin_size_min"]),
            prom_factor=float(peak_options["acth_prom_factor"]),
            min_distance_min=float(peak_options["min_distance_min"]),
        ),
        "Cortisol": calculate_binned_stats(
            time_min,
            x3,
            bin_size_min=float(peak_options["bin_size_min"]),
            prom_factor=float(peak_options["cortisol_prom_factor"]),
            min_distance_min=float(peak_options["min_distance_min"]),
        ),
    }


def _aggregate_config_simulation_stats(
    config: dict[str, Any],
    *,
    series_ids: list[int],
    peak_options: dict[str, float | int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    rep_stats: dict[str, list[pd.DataFrame]] = {"ACTH": [], "Cortisol": []}
    for rep in range(int(peak_options["n_reps"])):
        rep_seed = seed + rep
        for series_idx, series_id in enumerate(series_ids):
            signal_frames = _simulate_config_signal_stats_for_series(
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


def _simulate_signal_stats(
    theta: np.ndarray,
    *,
    settings: FitSettings,
    seed: int,
) -> dict[str, pd.DataFrame]:
    baseline, amplitude, kgr, tau_min, epsilon = [float(x) for x in theta]
    model = ThreeStateGRDelayModel(
        kgr=max(1e-6, kgr),
        tau_min=max(0.0, tau_min),
    )
    drive_kind = "sine_noise" if settings.noise_location == "drive" else "sine"
    drive_params: dict[str, float] = {
        "baseline": max(0.0, baseline),
        "amplitude": max(0.0, amplitude),
        "phase_min": settings.phase_min,
        "period_min": 1440.0,
    }
    if settings.noise_location == "drive":
        drive_params["epsilon"] = max(0.0, epsilon)
    drive = build_drive(drive_kind, drive_params)
    traj = simulate_trajectory_fit_arrays(
        model,
        drive,
        dt_min=1.0,
        warmup_min=settings.warmup_min,
        duration_min=settings.duration_min,
        seed=seed,
        noise_location=None if settings.noise_location == "drive" else settings.noise_location,
        noise_epsilon=0.0 if settings.noise_location == "drive" else max(0.0, epsilon),
    )
    traj_time = traj["time_min"]
    traj_x2 = traj["x2"]
    traj_x3 = traj["x3"]

    if settings.subsample_min > 0:
        time_min, x2 = _subsample_without_interpolation(
            traj_time,
            traj_x2,
            subsample_min=settings.subsample_min,
            duration_min=settings.duration_min,
        )
        _, x3 = _subsample_without_interpolation(
            traj_time,
            traj_x3,
            subsample_min=settings.subsample_min,
            duration_min=settings.duration_min,
        )
    else:
        time_min = traj_time
        x2 = traj_x2
        x3 = traj_x3

    return {
        "ACTH": calculate_binned_stats(
            time_min,
            x2,
            bin_size_min=settings.bin_size_min,
            prom_factor=settings.acth_prom_factor,
            min_distance_min=settings.min_distance_min,
        ),
        "Cortisol": calculate_binned_stats(
            time_min,
            x3,
            bin_size_min=settings.bin_size_min,
            prom_factor=settings.cortisol_prom_factor,
            min_distance_min=settings.min_distance_min,
        ),
    }


def _aggregate_simulation_stats(theta: np.ndarray, *, settings: FitSettings) -> dict[str, pd.DataFrame]:
    rep_stats: dict[str, list[pd.DataFrame]] = {"ACTH": [], "Cortisol": []}
    for rep in range(settings.n_reps):
        signal_frames = _simulate_signal_stats(theta, settings=settings, seed=settings.seed + rep)
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




def _failed_simulation_penalty(targets: HabsTargets, selected_signals: tuple[str, ...]) -> np.ndarray:
    return np.ones(len(targets.bins) * 3 * len(selected_signals), dtype=float) * 1e6


def _count_empty_bins(sim_profiles: dict[str, pd.DataFrame], selected_signals: tuple[str, ...]) -> int:
    empty_bins = 0
    for signal in selected_signals:
        profile = sim_profiles[signal]
        if profile.empty or "num_peaks" not in profile.columns:
            continue
        num_peaks = pd.to_numeric(profile["num_peaks"], errors="coerce").to_numpy(dtype=float)
        empty_bins += int(np.count_nonzero(~np.isfinite(num_peaks) | (num_peaks <= 0.0)))
    return empty_bins


def _has_too_many_empty_bins(sim_profiles: dict[str, pd.DataFrame], selected_signals: tuple[str, ...]) -> bool:
    return _count_empty_bins(sim_profiles, selected_signals) > 1

def _calculate_peak_metrics(
    targets: HabsTargets,
    sim_profiles: dict[str, pd.DataFrame],
    selected_signals: tuple[str, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for signal in selected_signals:
        target = targets.signals[signal]
        sim_profile = sim_profiles[signal].reindex(target.bins)
        signal_key = signal.lower()
        sim_count = np.nan_to_num(sim_profile["num_peaks"].to_numpy(dtype=float))
        sim_amp = np.nan_to_num(sim_profile["mean_amplitude"].to_numpy(dtype=float))
        sim_ibi = np.nan_to_num(sim_profile["mean_ibi"].to_numpy(dtype=float))
        sim_cv = np.nan_to_num(sim_profile["cv"].to_numpy(dtype=float))
        tgt_count = np.nan_to_num(target.count, nan=0.0)
        tgt_amp = np.nan_to_num(target.amplitude, nan=0.0)
        tgt_ibi = np.nan_to_num(target.ibi, nan=0.0)
        tgt_cv = np.nan_to_num(target.cv, nan=0.0)
        metrics[f"rmse_num_peaks_{signal_key}"] = float(np.sqrt(np.mean((sim_count - tgt_count) ** 2)))
        metrics[f"rmse_mean_amplitude_{signal_key}"] = float(np.sqrt(np.mean((sim_amp - tgt_amp) ** 2)))
        metrics[f"rmse_mean_ibi_{signal_key}"] = float(np.sqrt(np.mean((sim_ibi - tgt_ibi) ** 2)))
        metrics[f"rmse_cv_{signal_key}"] = float(np.sqrt(np.mean((sim_cv - tgt_cv) ** 2)))
    return metrics

def _build_comparison_rows(
    targets: HabsTargets,
    sim_profiles: dict[str, pd.DataFrame],
    selected_signals: tuple[str, ...]
) -> list[pd.DataFrame]:
    comparison_rows: list[pd.DataFrame] = []
    for signal in selected_signals:
        target = targets.signals[signal]
        sim_profile = sim_profiles[signal].reindex(target.bins)
        comparison_rows.append(
            pd.DataFrame(
                {
                    "signal": signal,
                    "bin": target.bins,
                    "target_num_peaks": np.nan_to_num(target.count, nan=0.0),
                    "target_mean_amplitude": np.nan_to_num(target.amplitude, nan=0.0),
                    "target_mean_ibi": np.nan_to_num(target.ibi, nan=0.0),
                    "target_cv": np.nan_to_num(target.cv, nan=0.0),
                    "sim_num_peaks": sim_profile["num_peaks"].to_numpy(dtype=float),
                    "sim_mean_amplitude": sim_profile["mean_amplitude"].to_numpy(dtype=float),
                    "sim_mean_ibi": sim_profile["mean_ibi"].to_numpy(dtype=float),
                    "sim_cv": sim_profile["cv"].to_numpy(dtype=float),
                }
            )
        )
    return comparison_rows

def objective(theta: np.ndarray, targets: HabsTargets, settings: FitSettings) -> np.ndarray:
    selected_signals = _selected_signals(settings.signal_mode)
    sim_dfs = _aggregate_simulation_stats(theta, settings=settings)
    if any(sim_dfs[s].empty for s in selected_signals):
        return _failed_simulation_penalty(targets, selected_signals)
    if _has_too_many_empty_bins(sim_dfs, selected_signals):
        return _failed_simulation_penalty(targets, selected_signals)
    return build_residual_vector(
        targets=targets,
        sim_profiles=sim_dfs,
        selected_signals=selected_signals,
        cv_weight=float(settings.cv_weight),
    )


def evaluate_theta(theta: np.ndarray, targets: HabsTargets, settings: FitSettings) -> dict[str, Any]:
    """Evaluate one parameter vector against the peak-statistic objective."""
    selected_signals = _selected_signals(settings.signal_mode)
    sim_profiles = _aggregate_simulation_stats(theta, settings=settings)
    if any(sim_profiles[s].empty for s in selected_signals):
        residual_vector = _failed_simulation_penalty(targets, selected_signals)
        return {
            "selected_signals": selected_signals,
            "sim_profiles": sim_profiles,
            "residual_vector": residual_vector,
            "objective_value": float(np.mean(residual_vector ** 2)),
            "metrics": {},
        }
    if _has_too_many_empty_bins(sim_profiles, selected_signals):
        residual_vector = _failed_simulation_penalty(targets, selected_signals)
        metrics = _calculate_peak_metrics(targets, sim_profiles, selected_signals)
        return {
            "selected_signals": selected_signals,
            "sim_profiles": sim_profiles,
            "residual_vector": residual_vector,
            "objective_value": float(np.mean(residual_vector ** 2)),
            "metrics": metrics,
        }

    residual_vector = build_residual_vector(
        targets=targets,
        sim_profiles=sim_profiles,
        selected_signals=selected_signals,
        cv_weight=float(settings.cv_weight),
    )
    metrics = _calculate_peak_metrics(targets, sim_profiles, selected_signals)

    return {
        "selected_signals": selected_signals,
        "sim_profiles": sim_profiles,
        "residual_vector": residual_vector,
        "objective_value": float(np.mean(residual_vector ** 2)),
        "metrics": metrics,
    }


def _fit_theta(targets: HabsTargets, *, settings: FitSettings) -> tuple[np.ndarray, Any]:
    lower = np.array([0.0, 0.0, 1e-6, 0.0, 0.0], dtype=float)
    upper = np.array([50.0, 50.0, float(settings.kgr_max), float(settings.tau_max_min), float(settings.epsilon_max)], dtype=float)
    x0 = np.clip(np.array([1.0, 0.25, 15.0, 20.0, 0.05], dtype=float), lower, upper)

    result = least_squares(
        lambda th: objective(th, targets, settings),
        x0=x0,
        bounds=(lower, upper),
        max_nfev=settings.max_nfev,
        loss="soft_l1",
    )
    return result.x, result


def evaluate_habs_dual_peak_stats_from_config(config: dict[str, Any]) -> dict[str, Any]:
    peak_options = _config_peak_stat_options(config)
    targets, _ = load_and_calculate_targets(
        variant=str(config["dataset"]["variant"]),
        bin_size_min=float(peak_options["bin_size_min"]),
        min_distance_min=float(peak_options["min_distance_min"]),
        acth_prom_factor=float(peak_options["acth_prom_factor"]),
        cortisol_prom_factor=float(peak_options["cortisol_prom_factor"]),
    )
    selected_signals = _selected_signals("both")
    series_ids = sorted(targets.frame[targets.id_col].dropna().unique().tolist())
    sim_profiles = _aggregate_config_simulation_stats(
        config,
        series_ids=[int(series_id) for series_id in series_ids],
        peak_options=peak_options,
        seed=int(config["runtime"]["seed"]),
    )
    if any(sim_profiles[s].empty for s in selected_signals):
        residual_vector = _failed_simulation_penalty(targets, selected_signals)
        return {
            "targets": targets,
            "sim_profiles": sim_profiles,
            "selected_signals": selected_signals,
            "residual_vector": residual_vector,
            "objective_value": float(np.mean(residual_vector ** 2)),
            "metrics": {},
        }
    if _has_too_many_empty_bins(sim_profiles, selected_signals):
        residual_vector = _failed_simulation_penalty(targets, selected_signals)
        metrics = _calculate_peak_metrics(targets, sim_profiles, selected_signals)
        return {
            "targets": targets,
            "sim_profiles": sim_profiles,
            "selected_signals": selected_signals,
            "residual_vector": residual_vector,
            "objective_value": float(np.mean(residual_vector ** 2)),
            "metrics": metrics,
        }

    residual_vector = build_residual_vector(
        targets=targets,
        sim_profiles=sim_profiles,
        selected_signals=selected_signals,
        cv_weight=float(peak_options["cv_weight"]),
    )
    metrics = _calculate_peak_metrics(targets, sim_profiles, selected_signals)
    return {
        "targets": targets,
        "sim_profiles": sim_profiles,
        "selected_signals": selected_signals,
        "residual_vector": residual_vector,
        "objective_value": float(np.mean(residual_vector ** 2)),
        "metrics": metrics,
    }


def _build_config_trajectory_comparison_frame(
    config: dict[str, Any],
    *,
    frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    spec = get_dataset_spec(str(config["dataset"]["name"]))
    solver = config["solver"]
    model = _build_model(config)
    export_rows: list[dict[str, float | int]] = []

    for series_id, group in frame.groupby(spec.id_col, sort=True):
        obs = group.sort_values(spec.time_col).copy()
        obs_times = obs[spec.time_col].to_numpy(dtype=float)
        drive = build_drive(
            str(config["drive"]["kind"]),
            {
                **config["drive"]["params"],
                "series_id": int(series_id),
            },
        )
        traj = simulate_trajectory(
            model,
            drive,
            dt_min=float(solver["dt_min"]),
            warmup_min=float(solver["warmup_min"]),
            duration_min=float(solver["duration_min"]),
            seed=int(seed) + int(series_id),
            noise_locations=_config_runtime_noise_locations(config),
            noise_epsilons=_config_runtime_noise_epsilons(config),
            noise_form=str(config.get("runtime", {}).get("noise_form", "multiplicative")),
        )
        sampled = sample_trajectory(traj, obs_times)
        acth_obs = _zscore(obs["ACTH"].to_numpy(dtype=float))
        acth_sim = _zscore(sampled["x2"].to_numpy(dtype=float))
        cort_obs = _zscore(obs["Cortisol"].to_numpy(dtype=float))
        cort_sim = _zscore(sampled["x3"].to_numpy(dtype=float))

        export_rows.extend(
            {
                "ID": int(series_id),
                "time_min": float(t),
                "time_hr": float(t) / 60.0,
                "observed_acth_z": float(oa),
                "model_x2_z": float(sa),
                "observed_cortisol_z": float(oc),
                "model_x3_z": float(sc),
            }
            for t, oa, sa, oc, sc in zip(obs_times, acth_obs, acth_sim, cort_obs, cort_sim, strict=False)
        )

    return pd.DataFrame(export_rows)



def fit_habs_dual_peak_stats_from_config(
    config: dict[str, Any],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if logger is not None:
        logger.info("Fitting HABS peak stats with per-ID two-harmonic drive")

    dataset_dir = out_dir / "artifacts"
    figure_dir = out_dir / "figures"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    peak_options = _config_peak_stat_options(config)
    targets, frame = load_and_calculate_targets(
        variant=str(config["dataset"]["variant"]),
        bin_size_min=float(peak_options["bin_size_min"]),
        min_distance_min=float(peak_options["min_distance_min"]),
        acth_prom_factor=float(peak_options["acth_prom_factor"]),
        cortisol_prom_factor=float(peak_options["cortisol_prom_factor"]),
    )
    selected_signals = _selected_signals("both")
    series_ids = sorted(frame[targets.id_col].dropna().unique().tolist())
    free_params = _resolve_config_free_params(config)
    bounds_cfg = config["fit"].get("bounds", {})

    x0_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for name in free_params:
        current = _get_nested(config, CONFIG_FREE_PARAM_PATHS[name])
        lb, ub, x0_value = _resolve_config_bound(name, current, bounds_cfg.get(name))
        lower.append(lb)
        upper.append(ub)
        x0_values.append(x0_value)
    x0 = np.array(x0_values, dtype=float)

    residuals = lambda theta: _config_peak_stat_residuals(
        np.asarray(theta, dtype=float),
        config=config,
        free_params=free_params,
        series_ids=[int(series_id) for series_id in series_ids],
        peak_options=peak_options,
        targets=targets,
        selected_signals=selected_signals,
    )

    optimizer_cfg = dict(peak_options["optimizer"])
    optimizer_name = str(optimizer_cfg["name"])
    bounds_array = (np.array(lower, dtype=float), np.array(upper, dtype=float))
    if optimizer_name == "least_squares":
        optimizer_result = least_squares(
            residuals,
            x0=x0,
            bounds=bounds_array,
            max_nfev=int(peak_options["max_nfev"]),
            loss="soft_l1",
        )
        theta_opt = np.array(optimizer_result.x, dtype=float)
        optimizer_cost = float(optimizer_result.cost)
        optimizer_nfev = float(optimizer_result.nfev)
        optimizer_nit = np.nan
    elif optimizer_name == "differential_evolution":
        scalar_bounds = list(zip(lower, upper, strict=True))
        de_workers = int(optimizer_cfg["workers"])
        scalar_objective = partial(
            _config_peak_stat_scalar_objective,
            config=config,
            free_params=free_params,
            series_ids=[int(series_id) for series_id in series_ids],
            peak_options=peak_options,
            targets=targets,
            selected_signals=selected_signals,
        )

        optimizer_result = differential_evolution(
            scalar_objective,
            bounds=scalar_bounds,
            maxiter=int(optimizer_cfg["maxiter"]),
            popsize=int(optimizer_cfg["popsize"]),
            polish=bool(optimizer_cfg["polish"]),
            seed=int(config["runtime"]["seed"]),
            workers=de_workers,
            updating="deferred" if de_workers != 1 else "immediate",
        )
        theta_opt = np.array(optimizer_result.x, dtype=float)
        optimizer_cost = float(optimizer_result.fun)
        optimizer_nfev = float(optimizer_result.nfev)
        optimizer_nit = float(optimizer_result.nit)
    elif optimizer_name == "emcee":
        if emcee is None:
            raise ImportError("emcee is not installed. Please install it to use the emcee optimizer.")

        ndim = len(x0)
        nwalkers = int(optimizer_cfg.get("popsize", max(32, 2 * ndim)))
        maxiter = int(optimizer_cfg.get("maxiter", 100))
        de_workers = int(optimizer_cfg.get("workers", 1))

        pos = x0 + 1e-4 * np.random.randn(nwalkers, ndim)
        pos = np.clip(pos, bounds_array[0], bounds_array[1])

        log_prob_fn = partial(
            _emcee_log_prob,
            lower=bounds_array[0],
            upper=bounds_array[1],
            config=config,
            free_params=free_params,
            series_ids=[int(sid) for sid in series_ids],
            peak_options=peak_options,
            targets=targets,
            selected_signals=selected_signals,
        )

        from multiprocessing import Pool
        if de_workers > 1:
            with Pool(de_workers) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, pool=pool)
                sampler.run_mcmc(pos, maxiter, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)
            sampler.run_mcmc(pos, maxiter, progress=True)

        burnin = int(0.2 * maxiter) if maxiter > 10 else 0
        samples = sampler.get_chain(discard=burnin, flat=True)
        log_probs = sampler.get_log_prob(discard=burnin, flat=True)

        best_idx = np.argmax(log_probs)
        theta_opt = np.array(samples[best_idx], dtype=float)

        optimizer_cost = -float(log_probs[best_idx])
        
        # approximate total evaluations: nwalkers * maxiter
        optimizer_nfev = float(nwalkers * maxiter)
        optimizer_nit = float(maxiter)

        class _MCMCResult:
            success = True
            message = "MCMC completed"
        optimizer_result = _MCMCResult()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    fitted_config = deepcopy(config)
    for idx, name in enumerate(free_params):
        _set_nested(fitted_config, CONFIG_FREE_PARAM_PATHS[name], float(theta_opt[idx]))

    evaluation = evaluate_habs_dual_peak_stats_from_config(fitted_config)
    sim_profiles = evaluation["sim_profiles"]
    comparison_rows = _build_comparison_rows(targets, sim_profiles, selected_signals)

    params = {
        "kgr": float(fitted_config["model"]["params"]["kgr"]),
        "tau_min": float(fitted_config["model"]["params"]["tau_min"]),
    }
    if str(fitted_config.get("drive", {}).get("kind", "")).endswith("_noise"):
        params["epsilon"] = float(fitted_config["drive"]["params"]["epsilon"])
        params["sigma"] = float(fitted_config["drive"]["params"]["epsilon"])
    params.update(_config_secretion_noise_params(fitted_config))
    summary_row = {
        "dataset": str(fitted_config["dataset"]["name"]),
        "variant": str(fitted_config["dataset"]["variant"]),
        "signal_mode": "both",
        "drive_kind": str(fitted_config["drive"]["kind"]),
        "optimizer": optimizer_name,
        "optimizer_workers": float(optimizer_cfg.get("workers", 1)) if optimizer_name == "differential_evolution" else 1.0,
        "noise_form": str(fitted_config.get("runtime", {}).get("noise_form", "multiplicative")),
        "sigma_note": (
            "sigma = drive epsilon"
            if "epsilon" in params
            else "production noise uses site-specific epsilons"
        ),
        "noise_locations": ",".join(_config_runtime_noise_locations(fitted_config)),
        "success": bool(optimizer_result.success),
        "message": str(optimizer_result.message),
        "cost": optimizer_cost,
        "nfev": optimizer_nfev,
        "nit": optimizer_nit,
        **params,
        **evaluation["metrics"],
    }

    fit_summary_path = dataset_dir / "fit_summary.csv"
    fit_params_path = dataset_dir / "fit_params.csv"
    peak_profile_path = dataset_dir / "peak_profile_comparison.csv"
    trajectory_csv_path = dataset_dir / "trajectory_comparison.csv"
    fitted_config_path = dataset_dir / "fitted_config.yaml"

    pd.DataFrame([summary_row]).to_csv(fit_summary_path, index=False)
    pd.DataFrame(
        [{"parameter": name, "value": value} for name, value in params.items()]
    ).to_csv(fit_params_path, index=False)
    peak_profile_frame = pd.concat(comparison_rows, ignore_index=True)
    peak_profile_frame.to_csv(peak_profile_path, index=False)
    trajectory_frame = _build_config_trajectory_comparison_frame(
        fitted_config,
        frame=frame,
        seed=int(fitted_config["runtime"]["seed"]),
    )
    trajectory_frame.to_csv(trajectory_csv_path, index=False)
    fitted_config_path.write_text(dump_yaml(fitted_config))

    plot_goodness_of_fit(
        targets,
        sim_profiles,
        selected_signals=selected_signals,
        out_path=figure_dir / "habs_goodness_of_fit.png",
        title="HABS peak-stat fit with per-ID two-harmonic drive",
    )
    plot_config_trajectory_comparison(
        trajectory_frame,
        out_path=figure_dir / "habs_trajectory_comparison.png",
        title="HABS model vs data with fitted per-ID two-harmonic drive",
    )

    if logger is not None:
        noise_summary = ", ".join(
            f"{name}={float(value):.3f}"
            for name, value in params.items()
            if name.startswith("epsilon")
        )
        logger.info(
            "Completed two-harmonic peak-stat fit: cost=%.4f tau=%.3f kgr=%.3f %s",
            float(optimizer_result.cost),
            params["tau_min"],
            params["kgr"],
            noise_summary,
        )

    return {
        "summary_row": summary_row,
        "params": params,
        "metrics": evaluation["metrics"],
        "targets": targets,
        "sim_profiles": sim_profiles,
        "peak_profile_frame": peak_profile_frame,
        "trajectory_frame": trajectory_frame,
        "fitted_config": fitted_config,
    }


def fit_dataset(
    *,
    settings: FitSettings,
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if logger is not None:
        logger.info("Fitting HABS (%s)", settings.variant)

    dataset_dir = out_dir / "artifacts"
    figure_dir = out_dir / "figures"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    targets, frame = load_and_calculate_targets(
        variant=settings.variant,
        bin_size_min=settings.bin_size_min,
        min_distance_min=settings.min_distance_min,
        acth_prom_factor=settings.acth_prom_factor,
        cortisol_prom_factor=settings.cortisol_prom_factor,
    )
    selected_signals = _selected_signals(settings.signal_mode)

    theta, result = _fit_theta(targets, settings=settings)
    evaluation = evaluate_theta(theta, targets, settings)
    sim_profiles = evaluation["sim_profiles"]
    if any(sim_profiles[s].empty for s in selected_signals):
        raise RuntimeError("Could not simulate any peak statistics for the selected signal mode")

    comparison_rows = _build_comparison_rows(targets, sim_profiles, selected_signals)

    summary_metrics: dict[str, Any] = dict(evaluation["metrics"])
    params = {
        "drive_baseline": float(theta[0]),
        "drive_amplitude": float(theta[1]),
        "kgr": float(theta[2]),
        "tau_min": float(theta[3]),
        "epsilon": float(theta[4]),
    }

    summary_row = {
        "dataset": "habs",
        "variant": settings.variant,
        "signal_mode": settings.signal_mode,
        "noise_location": settings.noise_location,
        "signals_used": ",".join(selected_signals),
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "nfev": float(result.nfev),
        **params,
        **summary_metrics,
    }

    pd.DataFrame([summary_row]).to_csv(dataset_dir / "fit_summary.csv", index=False)
    pd.DataFrame([{"parameter": name, "value": params[name]} for name in FIT_PARAM_NAMES]).to_csv(
        dataset_dir / "fit_params.csv",
        index=False,
    )
    pd.concat(comparison_rows, ignore_index=True).to_csv(
        dataset_dir / "peak_profile_comparison.csv",
        index=False,
    )

    plot_goodness_of_fit(
        targets,
        sim_profiles,
        selected_signals=selected_signals,
        out_path=figure_dir / "habs_goodness_of_fit.png",
        title=f"HABS peak-statistic fit ({settings.signal_mode})",
    )
    plot_trajectory_comparison(
        frame,
        theta,
        targets=targets,
        selected_signals=selected_signals,
        settings=settings,
        out_path=figure_dir / "habs_trajectory_comparison.png",
        title=f"HABS trajectory comparison ({settings.signal_mode})",
    )

    if logger is not None:
        logger.info(
            "Completed HABS: cost=%.4f, tau=%.2f, baseline=%.3f, amplitude=%.3f",
            float(result.cost),
            params["tau_min"],
            params["drive_baseline"],
            params["drive_amplitude"],
        )

    return {
        "summary_row": summary_row,
        "params": params,
        "metrics": summary_metrics,
        "targets": targets,
        "sim_profiles": sim_profiles,
    }


def _build_manifest(
    *,
    config_path: Path,
    out_dir: Path,
    settings: FitSettings,
) -> dict[str, Any]:
    return {
        "task": "fit_habs_dual_peak_stats",
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(out_dir.resolve()),
        "python_version": platform.python_version(),
        "seed": settings.seed,
        "git_commit": _git_commit(),
        "dataset": {"name": "habs", "variant": settings.variant},
        "signal_mode": settings.signal_mode,
        "noise_location": settings.noise_location,
    }


def run_fit(settings: FitSettings, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(out_dir / "logs" / "run.log")
    config = _default_run_config(settings)
    (out_dir / "resolved_config.yaml").write_text(dump_yaml(config))
    manifest = _build_manifest(
        config_path=out_dir / "resolved_config.yaml",
        out_dir=out_dir,
        settings=settings,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    result = fit_dataset(settings=settings, out_dir=out_dir, logger=logger)
    highlights = [
        f"Signals used: {settings.signal_mode}.",
        f"Noise location: {settings.noise_location}.",
        "Artifacts: artifacts/{fit_summary.csv, fit_params.csv, peak_profile_comparison.csv}.",
        "Figures: figures/habs_goodness_of_fit.png and figures/habs_trajectory_comparison.png.",
    ]
    (out_dir / "README.md").write_text(
        "# fit_habs_dual_peak_stats\n\n## Highlights\n" + "\n".join(f"- {item}" for item in highlights) + "\n"
    )
    return result


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Fit HABS with ACTH/cortisol peak-statistics.")
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=["raw", "shifted"],
        help="Dataset variant to load.",
    )
    parser.add_argument(
        "--signal-mode",
        default="both",
        choices=["both", "acth", "cortisol"],
        help="Which signal statistics to include in the fit.",
    )
    parser.add_argument(
        "--noise-location",
        default="drive",
        choices=["drive", "x1_secretion", "x2_secretion", "x3_secretion"],
        help="Single location where multiplicative Gaussian noise is injected.",
    )
    parser.add_argument(
        "--out",
        default="experiments/runs/fit_habs_dual_peak_stats",
        help="Output run directory.",
    )
    parser.add_argument("--max-nfev", type=int, default=24, help="Maximum least-squares evaluations.")
    parser.add_argument("--n-reps", type=int, default=N_REPS, help="Number of stochastic replicates per objective call.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base.")
    parser.add_argument(
        "--duration-min",
        type=float,
        default=SIM_DURATION_MIN,
        help="Simulation duration used for peak-stat analysis.",
    )
    parser.add_argument(
        "--subsample-min",
        type=float,
        default=SUBSAMPLE_MIN,
        help="Sampling interval used when extracting simulated analysis points.",
    )
    parser.add_argument(
        "--acth-prom-factor",
        type=float,
        default=DEFAULT_PROM_FACTOR_ACTH,
        help="Peak prominence factor for ACTH.",
    )
    parser.add_argument(
        "--cortisol-prom-factor",
        type=float,
        default=PROM_FACTOR_CORTISOL,
        help="Peak prominence factor for cortisol.",
    )
    parser.add_argument(
        "--cv-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to the CV residual term.",
    )
    parser.add_argument("--kgr-max", type=float, default=100.0, help="Upper bound for kgr during fitting.")
    parser.add_argument("--tau-max-min", type=float, default=180.0, help="Upper bound for tau_min during fitting.")
    parser.add_argument("--epsilon-max", type=float, default=25.0, help="Upper bound for epsilon during fitting.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = FitSettings(
        variant=str(args.variant),
        signal_mode=str(args.signal_mode),
        noise_location=str(args.noise_location),
        n_reps=int(args.n_reps),
        max_nfev=int(args.max_nfev),
        seed=int(args.seed),
        duration_min=float(args.duration_min),
        subsample_min=float(args.subsample_min),
        acth_prom_factor=float(args.acth_prom_factor),
        cortisol_prom_factor=float(args.cortisol_prom_factor),
        cv_weight=float(args.cv_weight),
        kgr_max=float(args.kgr_max),
        tau_max_min=float(args.tau_max_min),
        epsilon_max=float(args.epsilon_max),
    )
    run_fit(settings, Path(args.out))


if __name__ == "__main__":
    main()
