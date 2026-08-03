"""Single-signal peak-statistic fitting for the delayed three-state HPA model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import platform
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from ..config import dump_yaml
from ..data.registry import get_dataset_spec, load_dataset
from ..model.three_state_gr_delay import ThreeStateGRDelayModel, build_drive
from ..simulate.engine import simulate_trajectory


BIN_SIZE_MIN = 240.0
SUBSAMPLE_MIN = 10.0
SIM_DURATION_MIN = 1440.0 * 2.0
WARMUP_MIN = 1440.0
N_REPS = 6
MIN_DISTANCE_MIN = 30.0
PROM_FACTOR_CORTISOL = 0.05
DEFAULT_PHASE_MIN = 120.0
DEFAULT_NOISE_MODE = "multiplicative"
DEFAULT_DATASETS = ("habs", "all_digitized", "digitize_2019")
DEFAULT_VARIANT = "shifted"

FIT_PARAM_NAMES = ("drive_baseline", "drive_amplitude", "tau_min", "epsilon")


@dataclass(frozen=True)
class FitSettings:
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    variant: str = DEFAULT_VARIANT
    n_reps: int = N_REPS
    max_nfev: int = 24
    seed: int = 42
    noise_mode: str = DEFAULT_NOISE_MODE
    bin_size_min: float = BIN_SIZE_MIN
    subsample_min: float = SUBSAMPLE_MIN
    duration_min: float = SIM_DURATION_MIN
    warmup_min: float = WARMUP_MIN
    min_distance_min: float = MIN_DISTANCE_MIN
    prom_factor: float = PROM_FACTOR_CORTISOL
    phase_min: float = DEFAULT_PHASE_MIN


@dataclass(frozen=True)
class DatasetTargets:
    dataset: str
    variant: str
    id_col: str
    time_col: str
    value_col: str
    frame: pd.DataFrame
    bins: list[str]
    count: np.ndarray
    amplitude: np.ndarray
    ibi: np.ndarray
    cv: np.ndarray
    amplitude_sem: np.ndarray
    ibi_sem: np.ndarray
    cv_sem: np.ndarray


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model.individual_fit")
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


def _infer_value_column(frame: pd.DataFrame) -> str:
    for candidate in ("Cortisol", "cortisol", "value"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("Could not infer cortisol/value column from dataset frame")


def _safe_scale(values: np.ndarray, floor: float) -> float:
    mean = float(np.nanmean(np.abs(np.asarray(values, dtype=float))))
    if not np.isfinite(mean) or mean <= 0.0:
        return float(floor)
    return float(max(floor, mean))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if std <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _subsample_without_interpolation(
    time_min: np.ndarray,
    values: np.ndarray,
    *,
    subsample_min: float,
    duration_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_min = np.asarray(time_min, dtype=float)
    values = np.asarray(values, dtype=float)
    target_times = np.arange(0.0, duration_min, subsample_min, dtype=float)
    if target_times.size == 0:
        return target_times, np.asarray([], dtype=float)

    indices = np.searchsorted(time_min, target_times)
    if np.any(indices >= time_min.size):
        raise ValueError("Subsample target exceeds simulated trajectory range")

    matched_times = time_min[indices]
    if not np.allclose(matched_times, target_times, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"subsample_min={subsample_min} is not aligned with the solver time grid"
        )
    return target_times, values[indices]


def detect_peaks(
    time_min: np.ndarray,
    values: np.ndarray,
    *,
    min_dist_min: float = MIN_DISTANCE_MIN,
    prom_factor: float = PROM_FACTOR_CORTISOL,
) -> pd.DataFrame:
    """Detect peaks in a one-dimensional signal."""
    time_min = np.asarray(time_min, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return pd.DataFrame(columns=["time_min", "amplitude", "prominence"])

    dt = float(np.median(np.diff(time_min))) if time_min.size > 1 else 1.0
    min_dist_samples = max(1, int(round(min_dist_min / dt)))

    std = float(np.std(values))
    prominence = float(prom_factor) * std if std > 0.0 else float(prom_factor)
    peaks, properties = find_peaks(values, distance=min_dist_samples, prominence=prominence)
    if peaks.size == 0:
        return pd.DataFrame(columns=["time_min", "amplitude", "prominence"])

    prominences = np.asarray(properties.get("prominences", np.full(peaks.size, np.nan)), dtype=float)
    if prominences.size < peaks.size:
        prominences = np.pad(prominences, (0, peaks.size - prominences.size), constant_values=np.nan)

    return pd.DataFrame(
        {
            "time_min": time_min[peaks],
            "amplitude": values[peaks],
            "prominence": prominences[: peaks.size],
        }
    )


def calculate_peak_stats(peaks_df: pd.DataFrame) -> dict[str, float]:
    if peaks_df.empty:
        return {"num_peaks": 0.0, "mean_amplitude": 0.0, "mean_ibi": 0.0}

    peaks_sorted = peaks_df.sort_values("time_min").reset_index(drop=True)
    ibis = np.diff(peaks_sorted["time_min"].to_numpy(dtype=float))
    return {
        "num_peaks": float(len(peaks_sorted)),
        "mean_amplitude": float(np.mean(peaks_sorted["amplitude"])),
        "mean_ibi": float(np.mean(ibis)) if ibis.size > 0 else 0.0,
    }


def calculate_binned_stats(
    time_min: np.ndarray,
    values: np.ndarray,
    *,
    bin_size_min: float = BIN_SIZE_MIN,
    prom_factor: float = PROM_FACTOR_CORTISOL,
    min_distance_min: float = MIN_DISTANCE_MIN,
) -> pd.DataFrame:
    """Compute peak amplitude, IBI, and CV in fixed time-of-day bins."""
    time_min = np.asarray(time_min, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return pd.DataFrame()

    std = float(np.std(values))
    if std <= 0.0:
        return pd.DataFrame()

    values_z = _zscore(values)
    peaks_df = detect_peaks(
        time_min,
        values_z,
        min_dist_min=min_distance_min,
        prom_factor=prom_factor,
    )

    duration_min = float(np.max(time_min) - np.min(time_min))
    n_days = max(1.0, duration_min / 1440.0)
    bins = np.arange(0.0, 1440.0 + bin_size_min, bin_size_min)
    results: list[dict[str, float | str]] = []

    if not peaks_df.empty:
        peaks_df = peaks_df.sort_values("time_min").reset_index(drop=True)
        peaks_df["ibi"] = peaks_df["time_min"].diff()
        peaks_df["tod_min"] = np.mod(peaks_df["time_min"], 1440.0)
        peaks_df["bin_idx"] = np.digitize(peaks_df["tod_min"], bins, right=False) - 1

    tod = np.mod(time_min, 1440.0)
    for idx in range(len(bins) - 1):
        bin_start = float(bins[idx])
        bin_end = float(bins[idx + 1])
        bin_label = f"{int(bin_start / 60):02d}-{int(bin_end / 60):02d}"

        bin_mask = (tod >= bin_start) & (tod < bin_end)
        w_vals = values[bin_mask]
        bin_peaks = pd.DataFrame()
        if not peaks_df.empty:
            bin_peaks = peaks_df.loc[peaks_df["bin_idx"] == idx]

        if not bin_peaks.empty:
            w_mean = float(np.mean(w_vals)) if w_vals.size > 0 else 0.0
            results.append(
                {
                    "bin": bin_label,
                    "num_peaks": float(len(bin_peaks)) / n_days,
                    "mean_amplitude": float(bin_peaks["amplitude"].mean()),
                    "mean_ibi": float(bin_peaks["ibi"].dropna().mean())
                    if not bin_peaks["ibi"].dropna().empty
                    else np.nan,
                    "cv": float(np.std(w_vals) / w_mean) if w_vals.size > 0 and w_mean > 0 else 0.0,
                }
            )
            continue

        w_mean = float(np.mean(w_vals)) if w_vals.size > 0 else 0.0
        results.append(
            {
                "bin": bin_label,
                "num_peaks": 0.0,
                "mean_amplitude": np.nan,
                "mean_ibi": np.nan,
                "cv": float(np.std(w_vals) / w_mean) if w_vals.size > 0 and w_mean > 0 else 0.0,
            }
        )

    return pd.DataFrame(results)


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


def _build_drive(config: dict[str, Any]):
    return build_drive(config["drive"]["kind"], config["drive"]["params"])


def _default_run_config(settings: FitSettings) -> dict[str, Any]:
    model = ThreeStateGRDelayModel()
    return {
        "task": "fit_individual_datasets_new_model",
        "datasets": {"names": list(settings.datasets), "variant": settings.variant},
        "model": {
            "params": {
                **model.to_params_dict(),
            },
            "free_params": list(FIT_PARAM_NAMES),
        },
        "drive": {
            "kind": "sine_noise",
            "params": {
                "baseline": 1.0,
                "amplitude": 0.25,
                "phase_min": settings.phase_min,
                "period_min": 1440.0,
                "epsilon": 0.05,
            },
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
            "prom_factor": settings.prom_factor,
            "n_reps": settings.n_reps,
            "max_nfev": settings.max_nfev,
            "loss": {"mode": "peak_stats"},
        },
        "runtime": {"seed": settings.seed, "noise_mode": settings.noise_mode},
    }


def load_and_calculate_targets(
    dataset_name: str,
    *,
    variant: str = DEFAULT_VARIANT,
    bin_size_min: float = BIN_SIZE_MIN,
    prom_factor: float = PROM_FACTOR_CORTISOL,
    min_distance_min: float = MIN_DISTANCE_MIN,
) -> tuple[DatasetTargets, pd.DataFrame]:
    """Calculate target profiles for a single dataset."""
    spec = get_dataset_spec(dataset_name)
    frame = load_dataset(dataset_name, variant)
    value_col = _infer_value_column(frame)

    all_dfs: list[pd.DataFrame] = []
    for _, group in frame.groupby(spec.id_col, sort=False):
        time = group[spec.time_col].to_numpy(dtype=float)
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
        raise ValueError(f"No valid bins calculated for dataset {dataset_name}.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    grouped = full_df.groupby("bin", sort=False)
    avg_profile = grouped.mean(numeric_only=True)
    sem_profile = grouped.sem(numeric_only=True)

    targets = DatasetTargets(
        dataset=dataset_name,
        variant=variant,
        id_col=spec.id_col,
        time_col=spec.time_col,
        value_col=value_col,
        frame=frame,
        bins=avg_profile.index.tolist(),
        count=avg_profile["num_peaks"].to_numpy(dtype=float),
        amplitude=avg_profile["mean_amplitude"].to_numpy(dtype=float),
        ibi=avg_profile["mean_ibi"].to_numpy(dtype=float),
        cv=avg_profile["cv"].to_numpy(dtype=float),
        amplitude_sem=np.nan_to_num(sem_profile["mean_amplitude"].to_numpy(dtype=float)),
        ibi_sem=np.nan_to_num(sem_profile["mean_ibi"].to_numpy(dtype=float)),
        cv_sem=np.nan_to_num(sem_profile["cv"].to_numpy(dtype=float)),
    )
    return targets, frame


def _simulate_single_replicate(
    theta: np.ndarray,
    *,
    settings: FitSettings,
    seed: int,
) -> pd.DataFrame:
    baseline, amplitude, tau_min, epsilon = [float(x) for x in theta]
    model = ThreeStateGRDelayModel(
        tau_min=max(0.0, tau_min),
    )
    drive = build_drive(
        "sine_noise",
        {
            "baseline": max(0.0, baseline),
            "amplitude": max(0.0, amplitude),
            "phase_min": settings.phase_min,
            "period_min": 1440.0,
            "epsilon": max(0.0, epsilon),
        },
    )
    traj = simulate_trajectory(
        model,
        drive,
        dt_min=1.0,
        warmup_min=settings.warmup_min,
        duration_min=settings.duration_min,
        seed=seed,
    )

    if settings.subsample_min > 0:
        analysis_times = np.arange(0.0, settings.duration_min, settings.subsample_min)
        values = np.interp(analysis_times, traj["time_min"].to_numpy(dtype=float), traj["x3"].to_numpy(dtype=float))
        time_min = analysis_times
    else:
        time_min = traj["time_min"].to_numpy(dtype=float)
        values = traj["x3"].to_numpy(dtype=float)

    return calculate_binned_stats(
        time_min,
        values,
        bin_size_min=settings.bin_size_min,
        prom_factor=settings.prom_factor,
        min_distance_min=settings.min_distance_min,
    )


def objective(theta: np.ndarray, targets: DatasetTargets, settings: FitSettings) -> np.ndarray:
    """Residual vector for the peak-statistic objective."""
    rep_dfs: list[pd.DataFrame] = []
    for rep in range(settings.n_reps):
        df = _simulate_single_replicate(theta, settings=settings, seed=settings.seed + rep)
        if not df.empty and "bin" in df.columns:
            rep_dfs.append(df)

    if not rep_dfs:
        return np.ones(len(targets.bins) * 3, dtype=float) * 1e6

    all_stats = pd.concat(rep_dfs, ignore_index=True)
    avg_stats = all_stats.groupby("bin", sort=False).mean(numeric_only=True).reindex(targets.bins)

    target_amp = np.nan_to_num(targets.amplitude, nan=0.0)
    target_ibi = np.nan_to_num(targets.ibi, nan=0.0)
    target_cv = np.nan_to_num(targets.cv, nan=0.0)

    amp_scale = _safe_scale(target_amp, 0.1)
    ibi_scale = _safe_scale(target_ibi, 1.0)
    cv_scale = _safe_scale(target_cv, 0.1)

    amp_sim = np.nan_to_num(avg_stats["mean_amplitude"].to_numpy(dtype=float))
    ibi_sim = np.nan_to_num(avg_stats["mean_ibi"].to_numpy(dtype=float))
    cv_sim = np.nan_to_num(avg_stats["cv"].to_numpy(dtype=float))

    residuals = [
        (amp_sim - target_amp) / amp_scale,
        (ibi_sim - target_ibi) / ibi_scale,
        (cv_sim - target_cv) / cv_scale,
    ]
    return np.concatenate(residuals)


def _fit_theta(
    targets: DatasetTargets,
    *,
    settings: FitSettings,
) -> tuple[np.ndarray, Any]:
    x0 = np.array([1.0, 0.25, 20.0, 0.05], dtype=float)
    lower = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    upper = np.array([50.0, 50.0, 180.0, 2.0], dtype=float)

    result = least_squares(
        lambda th: objective(th, targets, settings),
        x0=x0,
        bounds=(lower, upper),
        max_nfev=settings.max_nfev,
        loss="soft_l1",
    )
    return result.x, result


def _aggregate_simulation_stats(theta: np.ndarray, *, settings: FitSettings) -> pd.DataFrame:
    rep_dfs = [
        _simulate_single_replicate(theta, settings=settings, seed=settings.seed + rep)
        for rep in range(settings.n_reps)
    ]
    rep_dfs = [df for df in rep_dfs if not df.empty and "bin" in df.columns]
    if not rep_dfs:
        return pd.DataFrame()
    grouped = pd.concat(rep_dfs, ignore_index=True).groupby("bin", sort=False)
    means = grouped.mean(numeric_only=True)
    sems = grouped.sem(numeric_only=True).rename(columns=lambda x: f"{x}_sem")
    return pd.concat([means, sems], axis=1)


def plot_goodness_of_fit(targets: DatasetTargets, sim_stats: pd.DataFrame, out_path: Path, *, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    x_hours = np.arange(len(targets.bins), dtype=float) * (BIN_SIZE_MIN / 60.0)
    panels = [
        ("num_peaks", targets.count, None, "Peak Count (per day)", "Count"),
        ("mean_amplitude", targets.amplitude, targets.amplitude_sem, "Peak Amplitude (z-score)", "Amplitude"),
        ("mean_ibi", targets.ibi, targets.ibi_sem, "Inter-Burst Interval (min)", "IBI (min)"),
        ("cv", targets.cv, targets.cv_sem, "Coefficient of Variation", "CV"),
    ]

    for ax, (sim_col, target_values, target_sem, panel_title, ylabel) in zip(axes, panels):
        sim_values = np.nan_to_num(sim_stats.reindex(targets.bins)[sim_col].to_numpy(dtype=float))
        if target_sem is not None:
            ax.errorbar(x_hours, target_values, yerr=target_sem, fmt="ko-", markersize=4, capsize=3, label="Data")
        else:
            ax.plot(x_hours, target_values, "ko-", markersize=4, label="Data")
        ax.plot(x_hours, sim_values, "ro--", markersize=4, label="Model")
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time of Day (h)")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_comparison(
    frame: pd.DataFrame,
    targets: DatasetTargets,
    theta: np.ndarray,
    *,
    settings: FitSettings,
    out_path: Path,
    title: str,
) -> None:
    spec = get_dataset_spec(targets.dataset)
    valid_ids = (
        frame.groupby(spec.id_col)
        .filter(lambda x: (x[spec.time_col].max() - x[spec.time_col].min()) > 720)
        [spec.id_col]
        .unique()
    )
    if len(valid_ids) == 0:
        return

    rng = np.random.default_rng(settings.seed)
    subject_id = rng.choice(valid_ids)
    subj = frame.loc[frame[spec.id_col] == subject_id, [spec.time_col, targets.value_col]].dropna().copy()

    model = ThreeStateGRDelayModel(tau_min=float(theta[2]))
    drive = build_drive(
        "sine_noise",
        {
            "baseline": float(theta[0]),
            "amplitude": float(theta[1]),
            "phase_min": settings.phase_min,
            "period_min": 1440.0,
            "epsilon": float(theta[3]),
        },
    )
    traj = simulate_trajectory(
        model,
        drive,
        dt_min=1.0,
        warmup_min=settings.warmup_min,
        duration_min=1440.0,
        seed=settings.seed,
    )

    subj_values = subj[targets.value_col].to_numpy(dtype=float)
    subj_z = _zscore(subj_values)
    t_data = np.mod(subj[spec.time_col].to_numpy(dtype=float), 1440.0)

    sim_values = traj["x3"].to_numpy(dtype=float)
    sim_z = _zscore(sim_values)
    t_sim = traj["time_min"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_data / 60.0, subj_z, "k.-", markersize=4, label=f"Data (ID: {subject_id})")
    axes[0].set_ylabel("Cortisol (z-score)")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(t_sim / 60.0, sim_z, "r.-", markersize=4, label="Model (fitted)")
    axes[1].set_ylabel("Cortisol (z-score)")
    axes[1].set_xlabel("Time of Day (h)")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_dataset(
    dataset_name: str,
    *,
    settings: FitSettings,
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if logger is not None:
        logger.info("Fitting dataset %s (%s)", dataset_name, settings.variant)

    dataset_dir = out_dir / "artifacts" / dataset_name
    figure_dir = out_dir / "figures"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    targets, frame = load_and_calculate_targets(
        dataset_name,
        variant=settings.variant,
        bin_size_min=settings.bin_size_min,
        prom_factor=settings.prom_factor,
        min_distance_min=settings.min_distance_min,
    )

    theta, result = _fit_theta(targets, settings=settings)
    sim_stats = _aggregate_simulation_stats(theta, settings=settings)
    if sim_stats.empty:
        raise RuntimeError(f"Could not simulate any peak statistics for dataset {dataset_name}")

    sim_profile = sim_stats.reindex(targets.bins)
    target_count = np.nan_to_num(targets.count, nan=0.0)
    target_amp = np.nan_to_num(targets.amplitude, nan=0.0)
    target_ibi = np.nan_to_num(targets.ibi, nan=0.0)
    target_cv = np.nan_to_num(targets.cv, nan=0.0)
    metrics = {
        "rmse_num_peaks": float(
            np.sqrt(np.mean((np.nan_to_num(sim_profile["num_peaks"].to_numpy(dtype=float)) - target_count) ** 2))
        ),
        "rmse_mean_amplitude": float(
            np.sqrt(
                np.mean((np.nan_to_num(sim_profile["mean_amplitude"].to_numpy(dtype=float)) - target_amp) ** 2)
            )
        ),
        "rmse_mean_ibi": float(
            np.sqrt(np.mean((np.nan_to_num(sim_profile["mean_ibi"].to_numpy(dtype=float)) - target_ibi) ** 2))
        ),
        "rmse_cv": float(np.sqrt(np.mean((np.nan_to_num(sim_profile["cv"].to_numpy(dtype=float)) - target_cv) ** 2))),
    }

    params = {
        "drive_baseline": float(theta[0]),
        "drive_amplitude": float(theta[1]),
        "tau_min": float(theta[2]),
        "epsilon": float(theta[3]),
    }
    summary_row = {
        "dataset": dataset_name,
        "variant": settings.variant,
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "nfev": float(result.nfev),
        **params,
        **metrics,
    }

    pd.DataFrame([summary_row]).to_csv(dataset_dir / "fit_summary.csv", index=False)
    pd.DataFrame([{"parameter": name, "value": params[name]} for name in FIT_PARAM_NAMES]).to_csv(
        dataset_dir / "fit_params.csv",
        index=False,
    )
    pd.DataFrame(
        {
        "bin": targets.bins,
        "target_num_peaks": target_count,
        "target_mean_amplitude": target_amp,
        "target_mean_ibi": target_ibi,
        "target_cv": target_cv,
        "sim_num_peaks": sim_profile["num_peaks"].to_numpy(dtype=float),
        "sim_mean_amplitude": sim_profile["mean_amplitude"].to_numpy(dtype=float),
        "sim_mean_ibi": sim_profile["mean_ibi"].to_numpy(dtype=float),
        "sim_cv": sim_profile["cv"].to_numpy(dtype=float),
    }
    ).to_csv(dataset_dir / "peak_profile_comparison.csv", index=False)

    plot_goodness_of_fit(
        targets,
        sim_profile,
        figure_dir / f"{dataset_name}_goodness_of_fit.png",
        title=f"{dataset_name} peak-statistic fit",
    )
    plot_trajectory_comparison(
        frame,
        targets,
        theta,
        settings=settings,
        out_path=figure_dir / f"{dataset_name}_trajectory_comparison.png",
        title=f"{dataset_name} trajectory comparison",
    )

    if logger is not None:
        logger.info(
            "Completed %s: cost=%.4f, tau=%.2f, baseline=%.3f, amplitude=%.3f",
            dataset_name,
            float(result.cost),
            params["tau_min"],
            params["drive_baseline"],
            params["drive_amplitude"],
        )

    return {
        "summary_row": summary_row,
        "params": params,
        "metrics": metrics,
        "targets": targets,
        "sim_profile": sim_profile,
    }


def _build_manifest(
    *,
    config_path: Path,
    out_dir: Path,
    settings: FitSettings,
    datasets: list[str],
) -> dict[str, Any]:
    return {
        "task": "fit_individual_datasets_new_model",
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(out_dir.resolve()),
        "python_version": platform.python_version(),
        "seed": settings.seed,
        "git_commit": _git_commit(),
        "datasets": datasets,
        "variant": settings.variant,
    }


def run_fit(settings: FitSettings, out_dir: Path) -> list[dict[str, Any]]:
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
        datasets=list(settings.datasets),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    summary_rows: list[dict[str, Any]] = []
    for dataset_name in settings.datasets:
        try:
            result = fit_dataset(dataset_name, settings=settings, out_dir=out_dir, logger=logger)
        except Exception as exc:
            logger.exception("Failed dataset %s", dataset_name)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": settings.variant,
                    "success": False,
                    "message": str(exc),
                }
            )
            continue
        summary_rows.append(result["summary_row"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "artifacts" / "summary_params.csv", index=False)

    highlights = [
        f"Fitted datasets: {', '.join(settings.datasets)}.",
        "Artifacts: artifacts/<dataset>/{fit_summary.csv, fit_params.csv, peak_profile_comparison.csv}.",
        "Figures: figures/<dataset>_goodness_of_fit.png and figures/<dataset>_trajectory_comparison.png.",
    ]
    (out_dir / "README.md").write_text("# fit_individual_datasets_new_model\n\n## Highlights\n" + "\n".join(f"- {item}" for item in highlights) + "\n")
    return summary_rows


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Fit individual datasets with the delayed three-state HPA model.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Datasets to fit (default: habs all_digitized digitize_2019).",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=["raw", "shifted"],
        help="Dataset variant to load.",
    )
    parser.add_argument(
        "--out",
        default="experiments/runs/fit_individual_datasets_new_model",
        help="Output run directory.",
    )
    parser.add_argument("--max-nfev", type=int, default=24, help="Maximum least-squares evaluations per dataset.")
    parser.add_argument("--n-reps", type=int, default=N_REPS, help="Number of stochastic replicates per objective call.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base.")
    parser.add_argument(
        "--noise-mode",
        default=DEFAULT_NOISE_MODE,
        choices=["additive", "multiplicative"],
        help="Drive noise interpretation for the run metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = FitSettings(
        datasets=tuple(args.datasets),
        variant=str(args.variant),
        n_reps=int(args.n_reps),
        max_nfev=int(args.max_nfev),
        seed=int(args.seed),
        noise_mode=str(args.noise_mode),
    )
    run_fit(settings, Path(args.out))


if __name__ == "__main__":
    main()
