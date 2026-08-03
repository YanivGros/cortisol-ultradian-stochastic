"""Hilbert-phase coherence analysis for ACTH and cortisol trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, hilbert, sosfiltfilt

from ...config import dump_yaml
from ...data.registry import get_dataset_spec, load_dataset


@dataclass(frozen=True)
class PhaseCoherenceSettings:
    dataset: str = "habs"
    variant: str = "shifted"
    normalize: str = "per_id_zscore"
    bandpass_min_period_hours: float = 1.0
    bandpass_max_period_hours: float = 3.0
    filter_order: int = 2
    detrend: bool = True
    edge_trim_hours: float = 2.0
    amplitude_mask_quantile: float = 0.10


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model.habs_phase_coherence")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _git_commit() -> str | None:
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


def _prepare_run_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _infer_dt_min(time_min: np.ndarray) -> float:
    time_min = np.asarray(time_min, dtype=float)
    diffs = np.diff(time_min)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        raise ValueError("At least two strictly increasing time points are required")
    return float(np.median(diffs))


def _bandpass_signal(
    values: np.ndarray,
    *,
    dt_hours: float,
    min_period_hours: float,
    max_period_hours: float,
    order: int,
) -> np.ndarray:
    if min_period_hours <= 0.0 or max_period_hours <= 0.0:
        raise ValueError("Bandpass periods must be positive")
    if max_period_hours <= min_period_hours:
        raise ValueError("bandpass_max_period_hours must exceed bandpass_min_period_hours")

    sample_rate_cph = 1.0 / dt_hours
    nyquist_cph = 0.5 * sample_rate_cph
    low_cph = 1.0 / max_period_hours
    high_cph = 1.0 / min_period_hours
    if high_cph >= nyquist_cph:
        raise ValueError(
            f"High cutoff {high_cph:.3f} cycles/hour exceeds Nyquist {nyquist_cph:.3f} cycles/hour"
        )
    normalized = [low_cph / nyquist_cph, high_cph / nyquist_cph]
    sos = butter(int(order), normalized, btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=float))


def _prepare_signal(
    values: np.ndarray,
    *,
    normalize: str,
    apply_detrend: bool,
    dt_hours: float,
    apply_bandpass: bool = True,
    min_period_hours: float = 1.0,
    max_period_hours: float = 3.0,
    filter_order: int = 2,
) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    if normalize == "per_id_zscore":
        signal = _zscore(signal)
    elif normalize != "raw":
        raise ValueError(f"Unsupported normalize mode: {normalize}")

    if apply_detrend:
        signal = detrend(signal, type="linear")

    if not apply_bandpass:
        return signal

    return _bandpass_signal(
        signal,
        dt_hours=dt_hours,
        min_period_hours=min_period_hours,
        max_period_hours=max_period_hours,
        order=filter_order,
    )


def analyze_phase_pair(
    time_min: np.ndarray,
    acth: np.ndarray,
    cortisol: np.ndarray,
    *,
    normalize: str = "per_id_zscore",
    apply_bandpass: bool = True,
    bandpass_min_period_hours: float = 1.0,
    bandpass_max_period_hours: float = 3.0,
    filter_order: int = 2,
    detrend_signals: bool = True,
    edge_trim_hours: float = 2.0,
    amplitude_mask_quantile: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    time_min = np.asarray(time_min, dtype=float)
    acth = np.asarray(acth, dtype=float)
    cortisol = np.asarray(cortisol, dtype=float)
    if time_min.ndim != 1 or acth.ndim != 1 or cortisol.ndim != 1:
        raise ValueError("time_min, acth, and cortisol must be one-dimensional")
    if not (len(time_min) == len(acth) == len(cortisol)):
        raise ValueError("time_min, acth, and cortisol must have equal length")
    if len(time_min) < 8:
        raise ValueError("At least 8 samples are required for phase analysis")

    finite = np.isfinite(time_min) & np.isfinite(acth) & np.isfinite(cortisol)
    if not np.all(finite):
        time_min = time_min[finite]
        acth = acth[finite]
        cortisol = cortisol[finite]
    if len(time_min) < 8:
        raise ValueError("Too few finite samples remain after dropping NaNs")

    order = np.argsort(time_min)
    time_min = time_min[order]
    acth = acth[order]
    cortisol = cortisol[order]

    dt_min = _infer_dt_min(time_min)
    dt_hours = dt_min / 60.0
    time_hours = time_min / 60.0

    acth_processed = _prepare_signal(
        acth,
        normalize=normalize,
        apply_detrend=detrend_signals,
        dt_hours=dt_hours,
        apply_bandpass=apply_bandpass,
        min_period_hours=bandpass_min_period_hours,
        max_period_hours=bandpass_max_period_hours,
        filter_order=filter_order,
    )
    cortisol_processed = _prepare_signal(
        cortisol,
        normalize=normalize,
        apply_detrend=detrend_signals,
        dt_hours=dt_hours,
        apply_bandpass=apply_bandpass,
        min_period_hours=bandpass_min_period_hours,
        max_period_hours=bandpass_max_period_hours,
        filter_order=filter_order,
    )

    acth_phase = np.unwrap(np.angle(hilbert(acth_processed)))
    cortisol_phase = np.unwrap(np.angle(hilbert(cortisol_processed)))
    delta_phase_unwrapped = cortisol_phase - acth_phase
    delta_phase = _wrap_to_pi(delta_phase_unwrapped)

    acth_inst_freq = np.gradient(acth_phase, time_hours) / (2.0 * np.pi)
    cortisol_inst_freq = np.gradient(cortisol_phase, time_hours) / (2.0 * np.pi)

    trim_samples = int(max(0, math.ceil(edge_trim_hours / dt_hours)))
    retained = np.ones_like(time_min, dtype=bool)
    if trim_samples > 0:
        retained[:trim_samples] = False
        retained[-trim_samples:] = False
    retained &= np.isfinite(delta_phase) & np.isfinite(delta_phase_unwrapped)
    retained &= np.isfinite(acth_inst_freq) & np.isfinite(cortisol_inst_freq)
    if int(np.sum(retained)) < 4:
        raise ValueError("Too few retained samples remain after edge trimming")

    # Amplitude mask: exclude samples where either signal has low Hilbert
    # amplitude, where phase estimates are noise-dominated.
    acth_amp = np.abs(hilbert(acth_processed))
    cortisol_amp = np.abs(hilbert(cortisol_processed))
    if amplitude_mask_quantile > 0.0 and int(np.sum(retained)) > 0:
        acth_thresh = float(np.quantile(acth_amp[retained], amplitude_mask_quantile))
        cortisol_thresh = float(np.quantile(cortisol_amp[retained], amplitude_mask_quantile))
        retained &= (acth_amp >= acth_thresh) & (cortisol_amp >= cortisol_thresh)
    if int(np.sum(retained)) < 4:
        raise ValueError("Too few retained samples remain after amplitude masking")

    mean_vector = np.mean(np.exp(1j * delta_phase[retained]))
    phase_locking_value = float(np.abs(mean_vector))
    mean_phase_lag_rad = float(np.angle(mean_vector))
    phase_circular_std_rad = (
        float(np.sqrt(-2.0 * np.log(phase_locking_value))) if 0.0 < phase_locking_value < 1.0 else 0.0
    )

    slope, intercept = np.polyfit(time_hours[retained], delta_phase_unwrapped[retained], deg=1)
    fitted = slope * time_hours[retained] + intercept
    residual_sd = float(np.std(delta_phase_unwrapped[retained] - fitted))

    mean_freq = float(
        np.nanmean(
            0.5
            * (
                acth_inst_freq[retained]
                + cortisol_inst_freq[retained]
            )
        )
    )
    mean_period_hours = float("nan")
    mean_phase_lag_hours = float("nan")
    if np.isfinite(mean_freq) and mean_freq > 0.0:
        mean_period_hours = float(1.0 / mean_freq)
        mean_phase_lag_hours = float(mean_phase_lag_rad / (2.0 * np.pi * mean_freq))

    series = pd.DataFrame(
        {
            "time_min": time_min,
            "time_hours": time_hours,
            "acth": acth,
            "cortisol": cortisol,
            "acth_processed": acth_processed,
            "cortisol_processed": cortisol_processed,
            "acth_phase_rad": acth_phase,
            "cortisol_phase_rad": cortisol_phase,
            "delta_phase_rad": delta_phase,
            "delta_phase_unwrapped_rad": delta_phase_unwrapped,
            "acth_inst_freq_cph": acth_inst_freq,
            "cortisol_inst_freq_cph": cortisol_inst_freq,
            "retained": retained.astype(int),
        }
    )
    summary = {
        "n_samples": float(len(series)),
        "retained_samples": float(int(np.sum(retained))),
        "dt_min": float(dt_min),
        "phase_locking_value": phase_locking_value,
        "mean_phase_lag_rad": mean_phase_lag_rad,
        "mean_phase_lag_hours": mean_phase_lag_hours,
        "phase_circular_std_rad": float(phase_circular_std_rad),
        "delta_phase_drift_slope_rad_per_hour": float(slope),
        "delta_phase_residual_sd_rad": residual_sd,
        "mean_inst_freq_acth_cph": float(np.nanmean(acth_inst_freq[retained])),
        "mean_inst_freq_cortisol_cph": float(np.nanmean(cortisol_inst_freq[retained])),
        "mean_ultradian_period_hours": mean_period_hours,
        "amplitude_mask_quantile": float(amplitude_mask_quantile),
    }
    return series, summary


def _build_resolved_config(settings: PhaseCoherenceSettings) -> dict[str, Any]:
    return {
        "task": "plot_habs_phase_coherence",
        "dataset": {"name": settings.dataset, "variant": settings.variant},
        "analysis": {
            "normalize": settings.normalize,
            "bandpass_min_period_hours": float(settings.bandpass_min_period_hours),
            "bandpass_max_period_hours": float(settings.bandpass_max_period_hours),
            "filter_order": int(settings.filter_order),
            "detrend": bool(settings.detrend),
            "edge_trim_hours": float(settings.edge_trim_hours),
            "amplitude_mask_quantile": float(settings.amplitude_mask_quantile),
            "phase_method": "hilbert_after_bandpass",
        },
    }


def _build_overall_summary(summary_frame: pd.DataFrame, phase_frame: pd.DataFrame) -> dict[str, float]:
    retained = phase_frame.loc[phase_frame["retained"] == 1, "delta_phase_rad"].to_numpy(dtype=float)
    pooled_vector = np.mean(np.exp(1j * retained)) if retained.size else complex(np.nan, np.nan)
    return {
        "n_ids": float(summary_frame["ID"].nunique()),
        "median_phase_locking_value": float(summary_frame["phase_locking_value"].median()),
        "min_phase_locking_value": float(summary_frame["phase_locking_value"].min()),
        "max_phase_locking_value": float(summary_frame["phase_locking_value"].max()),
        "median_abs_drift_slope_rad_per_hour": float(
            summary_frame["delta_phase_drift_slope_rad_per_hour"].abs().median()
        ),
        "pooled_phase_locking_value": float(np.abs(pooled_vector)),
        "pooled_mean_phase_lag_rad": float(np.angle(pooled_vector)),
        "pooled_retained_samples": float(retained.size),
    }


def _plot_subject_phase_panels(phase_frame: pd.DataFrame, summary_frame: pd.DataFrame, output_path: Path) -> None:
    subject_ids = sorted(summary_frame["ID"].astype(int).tolist())
    n_rows = len(subject_ids)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, max(2.8 * n_rows, 6.0)), sharex="col")
    if n_rows == 1:
        axes = np.asarray([axes])

    color_acth = "#355070"
    color_cort = "#b56576"
    color_phase = "#6d597a"

    for row_idx, subject_id in enumerate(subject_ids):
        subject = phase_frame.loc[phase_frame["ID"] == subject_id].sort_values("time_hours")
        summary = summary_frame.loc[summary_frame["ID"] == subject_id].iloc[0]
        ax_signal = axes[row_idx, 0]
        ax_phase = axes[row_idx, 1]

        retained = subject["retained"].to_numpy(dtype=bool)
        ax_signal.plot(
            subject["time_hours"],
            subject["acth_processed"],
            color=color_acth,
            lw=1.4,
            label="ACTH",
        )
        ax_signal.plot(
            subject["time_hours"],
            subject["cortisol_processed"],
            color=color_cort,
            lw=1.4,
            label="Cortisol",
        )
        if np.any(~retained):
            ax_signal.scatter(
                subject.loc[~retained, "time_hours"],
                subject.loc[~retained, "acth_processed"],
                s=10,
                color=color_acth,
                alpha=0.25,
            )
            ax_signal.scatter(
                subject.loc[~retained, "time_hours"],
                subject.loc[~retained, "cortisol_processed"],
                s=10,
                color=color_cort,
                alpha=0.25,
            )
        ax_signal.set_ylabel(f"ID {subject_id}\nFiltered")
        ax_signal.spines["top"].set_visible(False)
        ax_signal.spines["right"].set_visible(False)
        ax_signal.grid(alpha=0.15)
        if row_idx == 0:
            ax_signal.legend(loc="upper right", frameon=False)
            ax_signal.set_title("Bandpassed ACTH and cortisol")

        ax_phase.plot(
            subject["time_hours"],
            subject["delta_phase_rad"],
            color=color_phase,
            lw=1.4,
        )
        ax_phase.axhline(float(summary["mean_phase_lag_rad"]), color="#2a9d8f", lw=1.0, ls="--")
        if np.any(~retained):
            ax_phase.scatter(
                subject.loc[~retained, "time_hours"],
                subject.loc[~retained, "delta_phase_rad"],
                s=10,
                color=color_phase,
                alpha=0.25,
            )
        ax_phase.set_ylim(-np.pi, np.pi)
        ax_phase.set_yticks([-np.pi, 0.0, np.pi], labels=["-pi", "0", "pi"])
        ax_phase.spines["top"].set_visible(False)
        ax_phase.spines["right"].set_visible(False)
        ax_phase.grid(alpha=0.15)
        ax_phase.text(
            0.01,
            0.93,
            (
                f"PLV={float(summary['phase_locking_value']):.2f}, "
                f"slope={float(summary['delta_phase_drift_slope_rad_per_hour']):.2f} rad/h"
            ),
            transform=ax_phase.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        if row_idx == 0:
            ax_phase.set_title("Delta phase: cortisol - ACTH")

    axes[-1, 0].set_xlabel("Time (hours)")
    axes[-1, 1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(summary_frame: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_frame.sort_values("phase_locking_value", ascending=False).reset_index(drop=True)
    labels = ordered["ID"].astype(int).astype(str).tolist()
    x = np.arange(len(ordered), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    axes[0].bar(x, ordered["phase_locking_value"], color="#4c956c")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Phase-locking value")
    axes[0].set_xlabel("ID")
    axes[0].set_xticks(x, labels=labels)
    axes[0].set_title("Higher is more stable")

    axes[1].bar(x, ordered["delta_phase_drift_slope_rad_per_hour"].abs(), color="#bc4749")
    axes[1].set_ylabel("|Drift slope| (rad/hour)")
    axes[1].set_xlabel("ID")
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_title("Lower is more stable")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def settings_from_config(config: dict[str, Any]) -> PhaseCoherenceSettings:
    analysis = config.get("analysis", {})
    return PhaseCoherenceSettings(
        dataset=str(config["dataset"]["name"]),
        variant=str(config["dataset"]["variant"]),
        normalize=str(analysis.get("normalize", "per_id_zscore")),
        bandpass_min_period_hours=float(analysis.get("bandpass_min_period_hours", 1.0)),
        bandpass_max_period_hours=float(analysis.get("bandpass_max_period_hours", 6.0)),
        filter_order=int(analysis.get("filter_order", 2)),
        detrend=bool(analysis.get("detrend", True)),
        edge_trim_hours=float(analysis.get("edge_trim_hours", 2.0)),
        amplitude_mask_quantile=float(analysis.get("amplitude_mask_quantile", 0.10)),
    )


def run_phase_coherence(settings: PhaseCoherenceSettings, out_dir: Path) -> dict[str, Any]:
    _prepare_run_dirs(out_dir)
    logger = _setup_logging(out_dir / "logs" / "run.log")

    resolved_config = _build_resolved_config(settings)
    (out_dir / "resolved_config.yaml").write_text(dump_yaml(resolved_config))

    manifest = {
        "task": "plot_habs_phase_coherence",
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(out_dir.resolve()),
        "config_path": str((out_dir / "resolved_config.yaml").resolve()),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    spec = get_dataset_spec(settings.dataset)
    if spec.signal_names != ("ACTH", "Cortisol"):
        raise ValueError(f"Dataset {settings.dataset!r} does not expose ACTH and Cortisol")

    frame = load_dataset(settings.dataset, settings.variant)
    frame = frame.loc[:, [spec.id_col, spec.time_col, "ACTH", "Cortisol"]].copy()
    frame = frame.dropna(subset=["ACTH", "Cortisol"]).copy()
    frame = frame.rename(columns={spec.id_col: "ID", spec.time_col: "time_min"})
    frame = frame.sort_values(["ID", "time_min"]).reset_index(drop=True)

    logger.info(
        "Running phase coherence analysis on %s/%s with %d rows across %d IDs",
        settings.dataset,
        settings.variant,
        len(frame),
        frame["ID"].nunique(),
    )

    series_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int]] = []
    for subject_id, subject in frame.groupby("ID", sort=True):
        phase_series, phase_summary = analyze_phase_pair(
            subject["time_min"].to_numpy(dtype=float),
            subject["ACTH"].to_numpy(dtype=float),
            subject["Cortisol"].to_numpy(dtype=float),
            normalize=settings.normalize,
            bandpass_min_period_hours=settings.bandpass_min_period_hours,
            bandpass_max_period_hours=settings.bandpass_max_period_hours,
            filter_order=settings.filter_order,
            detrend_signals=settings.detrend,
            edge_trim_hours=settings.edge_trim_hours,
            amplitude_mask_quantile=settings.amplitude_mask_quantile,
        )
        phase_series.insert(0, "ID", int(subject_id))
        series_rows.append(phase_series)
        summary_rows.append({"ID": int(subject_id)} | phase_summary)

    phase_frame = pd.concat(series_rows, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values("ID").reset_index(drop=True)
    overall_summary = _build_overall_summary(summary_frame, phase_frame)

    phase_frame.to_csv(out_dir / "artifacts" / "phase_coherence_series.csv", index=False)
    summary_frame.to_csv(out_dir / "artifacts" / "phase_coherence_summary.csv", index=False)
    pd.DataFrame([overall_summary]).to_csv(out_dir / "artifacts" / "phase_coherence_overall.csv", index=False)
    _plot_subject_phase_panels(
        phase_frame,
        summary_frame,
        out_dir / "figures" / "habs_phase_coherence_subjects.png",
    )
    _plot_subject_phase_panels(
        phase_frame,
        summary_frame,
        out_dir / "figures" / "habs_phase_coherence_subjects.pdf",
    )
    _plot_summary(summary_frame, out_dir / "figures" / "habs_phase_coherence_summary.png")
    _plot_summary(summary_frame, out_dir / "figures" / "habs_phase_coherence_summary.pdf")

    readme_lines = [
        "# HABS Phase Coherence",
        "",
        f"- Dataset: `{settings.dataset}` ({settings.variant})",
        f"- Subjects: {int(summary_frame['ID'].nunique())}",
        f"- Normalization: `{settings.normalize}`",
        (
            f"- Bandpass: {float(settings.bandpass_min_period_hours):.2f} to "
            f"{float(settings.bandpass_max_period_hours):.2f} hour periods"
        ),
        f"- Filter order: {int(settings.filter_order)}",
        f"- Linear detrend before filtering: `{bool(settings.detrend)}`",
        f"- Edge trim before summary metrics: {float(settings.edge_trim_hours):.2f} hours per side",
        f"- Median subject PLV: {float(overall_summary['median_phase_locking_value']):.4f}",
        (
            f"- Median subject |drift slope|: "
            f"{float(overall_summary['median_abs_drift_slope_rad_per_hour']):.4f} rad/hour"
        ),
        f"- Pooled PLV: {float(overall_summary['pooled_phase_locking_value']):.4f}",
        "",
        "Interpretation:",
        "- Stable phase lag is supported by high PLV together with low drift slope over time.",
        "- Large drift or weak locking argues against a clean delay-driven phase relationship.",
        "",
        "Artifacts:",
        "- `artifacts/phase_coherence_series.csv`",
        "- `artifacts/phase_coherence_summary.csv`",
        "- `artifacts/phase_coherence_overall.csv`",
        "- `figures/habs_phase_coherence_subjects.png`",
        "- `figures/habs_phase_coherence_subjects.pdf`",
        "- `figures/habs_phase_coherence_summary.png`",
        "- `figures/habs_phase_coherence_summary.pdf`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines))

    logger.info(
        "Saved phase coherence outputs for %d IDs; median PLV=%.4f median |slope|=%.4f rad/hour",
        int(summary_frame["ID"].nunique()),
        float(overall_summary["median_phase_locking_value"]),
        float(overall_summary["median_abs_drift_slope_rad_per_hour"]),
    )
    return {
        "phase_frame": phase_frame,
        "summary": summary_frame,
        "overall": overall_summary,
    }


def run_phase_coherence_from_config(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    return run_phase_coherence(settings_from_config(config), out_dir)


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run Hilbert phase coherence analysis on HABS ACTH/cortisol data.")
    parser.add_argument("--dataset", default="habs")
    parser.add_argument("--variant", default="shifted")
    parser.add_argument("--normalize", default="per_id_zscore", choices=["per_id_zscore", "raw"])
    parser.add_argument("--bandpass-min-period-hours", type=float, default=1.0)
    parser.add_argument("--bandpass-max-period-hours", type=float, default=6.0)
    parser.add_argument("--filter-order", type=int, default=2)
    parser.add_argument("--edge-trim-hours", type=float, default=2.0)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument(
        "--out",
        default="experiments/runs/plot_habs_phase_coherence",
        help="Output run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = PhaseCoherenceSettings(
        dataset=str(args.dataset),
        variant=str(args.variant),
        normalize=str(args.normalize),
        bandpass_min_period_hours=float(args.bandpass_min_period_hours),
        bandpass_max_period_hours=float(args.bandpass_max_period_hours),
        filter_order=int(args.filter_order),
        detrend=not bool(args.no_detrend),
        edge_trim_hours=float(args.edge_trim_hours),
    )
    run_phase_coherence(settings, Path(args.out))


if __name__ == "__main__":
    main()
