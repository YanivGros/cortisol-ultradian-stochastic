"""Plotting functions for HABS dual-peak fitting results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...data.registry import get_dataset_spec
from ...model.three_state_gr_delay import ThreeStateGRDelayModel, build_drive
from ...simulate.engine import simulate_trajectory
from ...fit.objectives import HabsTargets
from ...fit.individual_datasets_new_model import BIN_SIZE_MIN, _subsample_without_interpolation, _zscore


def plot_goodness_of_fit(
    targets: HabsTargets,
    sim_profiles: dict[str, pd.DataFrame],
    *,
    selected_signals: tuple[str, ...],
    out_path: Path,
    title: str,
) -> None:
    n_rows = len(selected_signals)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows), sharex="col")
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    x_hours = np.arange(len(targets.bins), dtype=float) * (BIN_SIZE_MIN / 60.0)
    panel_titles = [
        ("mean_amplitude", "Peak Amplitude (z-score)", "Amplitude"),
        ("mean_ibi", "Inter-Burst Interval (min)", "IBI (min)"),
        ("cv", "Coefficient of Variation", "CV"),
    ]

    for row_idx, signal in enumerate(selected_signals):
        target = targets.signals[signal]
        sim_profile = sim_profiles[signal].reindex(target.bins)
        target_arrays = [
            np.nan_to_num(target.amplitude, nan=0.0),
            np.nan_to_num(target.ibi, nan=0.0),
            np.nan_to_num(target.cv, nan=0.0),
        ]
        target_sems = [
            np.nan_to_num(target.amplitude_sem, nan=0.0),
            np.nan_to_num(target.ibi_sem, nan=0.0),
            np.nan_to_num(target.cv_sem, nan=0.0),
        ]
        sim_arrays = [
            np.nan_to_num(sim_profile["mean_amplitude"].to_numpy(dtype=float)),
            np.nan_to_num(sim_profile["mean_ibi"].to_numpy(dtype=float)),
            np.nan_to_num(sim_profile["cv"].to_numpy(dtype=float)),
        ]

        for col_idx, ((_, panel_title, ylabel), target_values, target_sem, sim_values) in enumerate(
            zip(panel_titles, target_arrays, target_sems, sim_arrays, strict=True)
        ):
            ax = axes[row_idx, col_idx]
            ax.errorbar(x_hours, target_values, yerr=target_sem, fmt="ko-", markersize=4, capsize=3, label="Data")
            ax.plot(x_hours, sim_values, "ro--", markersize=4, label="Model")
            ax.set_title(f"{signal} {panel_title}")
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
    theta: np.ndarray,
    *,
    targets: HabsTargets,
    selected_signals: tuple[str, ...],
    settings: Any,
    out_path: Path,
    title: str,
) -> None:
    spec = get_dataset_spec("habs")
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
    subj = frame.loc[frame[spec.id_col] == subject_id].copy()

    model = ThreeStateGRDelayModel(
        kgr=float(theta[2]),
        tau_min=float(theta[3]),
    )
    drive_kind = "sine_noise" if settings.noise_location == "drive" else "sine"
    drive_params: dict[str, float] = {
        "baseline": float(theta[0]),
        "amplitude": float(theta[1]),
        "phase_min": settings.phase_min,
        "period_min": 1440.0,
    }
    if settings.noise_location == "drive":
        drive_params["epsilon"] = float(theta[4])
    drive = build_drive(drive_kind, drive_params)
    traj = simulate_trajectory(
        model,
        drive,
        dt_min=1.0,
        warmup_min=settings.warmup_min,
        duration_min=1440.0,
        seed=settings.seed,
        noise_location=None if settings.noise_location == "drive" else settings.noise_location,
        noise_epsilon=0.0 if settings.noise_location == "drive" else float(theta[4]),
    )
    traj_time = traj["time_min"].to_numpy(dtype=float)

    rows = len(selected_signals)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 4.2 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, signal in zip(axes, selected_signals, strict=True):
        signal_targets = targets.signals[signal]
        value_col = signal_targets.value_col
        subj_vals = subj[value_col].dropna().to_numpy(dtype=float)
        if subj_vals.size == 0:
            continue
        subj_z = _zscore(subj_vals)
        t_data = np.mod(subj.loc[subj[value_col].notna(), spec.time_col].to_numpy(dtype=float), 1440.0)

        if signal == "ACTH":
            sim_vals = traj["x2"].to_numpy(dtype=float)
        else:
            sim_vals = traj["x3"].to_numpy(dtype=float)

        if settings.subsample_min > 0:
            t_sim, sim_vals = _subsample_without_interpolation(
                traj_time,
                sim_vals,
                subsample_min=settings.subsample_min,
                duration_min=1440.0,
            )
        else:
            t_sim = traj_time

        sim_z = _zscore(sim_vals)

        ax.plot(t_data / 60.0, subj_z, "k.-", markersize=4, label=f"Data {signal} (ID: {subject_id})")
        ax.plot(t_sim / 60.0, sim_z, "r.-", markersize=4, label=f"Model {signal}")
        ax.set_ylabel(f"{signal} (z-score)")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time of Day (h)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_config_trajectory_comparison(
    trajectory_frame: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
) -> None:
    ids = sorted(trajectory_frame["ID"].dropna().unique().tolist())
    fig, axes = plt.subplots(nrows=len(ids), ncols=2, figsize=(12, 2.4 * len(ids)), sharex=True)
    if len(ids) == 1:
        axes = np.array([axes])

    for row_idx, series_id in enumerate(ids):
        group = trajectory_frame.loc[trajectory_frame["ID"] == series_id].sort_values("time_min")
        time_hr = group["time_hr"].to_numpy(dtype=float)
        ax_acth = axes[row_idx, 0]
        ax_cort = axes[row_idx, 1]
        ax_acth.plot(
            time_hr,
            group["observed_acth_z"].to_numpy(dtype=float),
            "o-",
            color="#1f77b4",
            linewidth=1.2,
            markersize=3.0,
            label="Observed ACTH",
        )
        ax_acth.plot(
            time_hr,
            group["model_x2_z"].to_numpy(dtype=float),
            "-",
            color="#d62728",
            linewidth=1.6,
            label="Model x2",
        )
        ax_cort.plot(
            time_hr,
            group["observed_cortisol_z"].to_numpy(dtype=float),
            "o-",
            color="#1f77b4",
            linewidth=1.2,
            markersize=3.0,
            label="Observed Cortisol",
        )
        ax_cort.plot(
            time_hr,
            group["model_x3_z"].to_numpy(dtype=float),
            "-",
            color="#d62728",
            linewidth=1.6,
            label="Model x3",
        )
        ax_acth.set_ylabel(f"ID {int(series_id)}\nz-score")
        ax_acth.set_title("ACTH")
        ax_cort.set_title("Cortisol")
        for ax in (ax_acth, ax_cort):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(0.0, 24.0)
            ax.set_xticks([0, 6, 12, 18, 24])
        if row_idx == 0:
            ax_acth.legend(frameon=False, loc="upper right")
            ax_cort.legend(frameon=False, loc="upper right")

    axes[-1, 0].set_xlabel("Time (hours)")
    axes[-1, 1].set_xlabel("Time (hours)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
