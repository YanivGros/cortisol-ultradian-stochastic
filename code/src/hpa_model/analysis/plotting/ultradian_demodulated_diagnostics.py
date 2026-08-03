"""Demodulated ultradian diagnostics across packaged shifted datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from math import ceil
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import cramervonmises, norm, rayleigh
import yaml

from ...config import dump_yaml
from ...data.registry import PROJECT_ROOT, get_dataset_spec, load_dataset, load_shift_params
from ...data.two_harmonic_shift import evaluate_two_harmonic, fit_two_harmonic_params
from ...plotting import apply_paper_style, setup_nature_style
from ..flow_field.habs_lagged_cross_correlation import compute_subject_lagged_correlation
from ..flow_field.habs_phase_coherence import _infer_dt_min, _prepare_signal, analyze_phase_pair


SIGNAL_ORDER: tuple[str, ...] = ("ACTH", "Cortisol")
SIGNAL_COLORS: dict[str, str] = {
    "ACTH": "#8C8C8C",
    "Cortisol": "#1A1A1A",
}
DATASET_STYLES: dict[str, dict[str, str]] = {
    "habs": {"label": "HABS"},
    "digitize_2019": {"label": "Russell & Lightman"},
    "all_digitized": {"label": "All Digitized"},
}


@dataclass(frozen=True)
class UltradianDemodulatedDiagnosticsSettings:
    dataset_names: tuple[str, ...] = ("habs", "digitize_2019", "all_digitized", "habs_microdialysis_cortisol")
    dataset_variant: str = "shifted"
    signals: tuple[str, ...] | None = None
    envelope_window_hours: float = 6.0
    envelope_floor_quantile: float = 0.10
    bandpass_min_period_hours: float = 1.0
    bandpass_max_period_hours: float = 3.0
    filter_order: int = 2
    apply_bandpass: bool = True
    detrend: bool = True
    edge_trim_hours: float = 2.0
    amplitude_mask_quantile: float = 0.10
    max_crosscorr_lag_hours: float = 6.0
    amplitude_hist_bins: int = 30
    phase_diffusion_bins: int = 6
    return_map_bins: int = 6
    export_demodulated_series: bool = False
    dpi: int = 300
    title_prefix: str = "Demodulated ultradian diagnostics"


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model.ultradian_demodulated_diagnostics")
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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _prepare_run_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def load_settings_from_config(config_path: Path) -> UltradianDemodulatedDiagnosticsSettings:
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")

    dataset_cfg = config.get("dataset", {})
    demod_cfg = config.get("demodulation", {})
    analysis_cfg = config.get("analysis", {})
    plot_cfg = config.get("plot", {})
    signal_cfg = config.get("signals")
    signals = None if signal_cfg is None else tuple(str(signal) for signal in signal_cfg)

    return UltradianDemodulatedDiagnosticsSettings(
        dataset_names=tuple(str(name) for name in config.get("datasets", ("habs", "digitize_2019", "all_digitized", "habs_microdialysis_cortisol"))),
        dataset_variant=str(dataset_cfg.get("variant", "shifted")),
        signals=signals,
        envelope_window_hours=float(demod_cfg.get("envelope_window_hours", 6.0)),
        envelope_floor_quantile=float(demod_cfg.get("envelope_floor_quantile", 0.10)),
        bandpass_min_period_hours=float(analysis_cfg.get("bandpass_min_period_hours", 1.0)),
        bandpass_max_period_hours=float(analysis_cfg.get("bandpass_max_period_hours", 6.0)),
        filter_order=int(analysis_cfg.get("filter_order", 2)),
        apply_bandpass=bool(analysis_cfg.get("apply_bandpass", True)),
        detrend=bool(analysis_cfg.get("detrend", True)),
        edge_trim_hours=float(analysis_cfg.get("edge_trim_hours", 2.0)),
        amplitude_mask_quantile=float(analysis_cfg.get("amplitude_mask_quantile", 0.10)),
        max_crosscorr_lag_hours=float(analysis_cfg.get("max_crosscorr_lag_hours", 6.0)),
        amplitude_hist_bins=int(plot_cfg.get("amplitude_hist_bins", 30)),
        phase_diffusion_bins=int(plot_cfg.get("phase_diffusion_bins", 6)),
        return_map_bins=int(plot_cfg.get("return_map_bins", 6)),
        export_demodulated_series=bool(analysis_cfg.get("export_demodulated_series", False)),
        dpi=int(plot_cfg.get("dpi", 300)),
        title_prefix=str(plot_cfg.get("title_prefix", "Demodulated ultradian diagnostics")),
    )


def _build_resolved_config(settings: UltradianDemodulatedDiagnosticsSettings) -> dict[str, object]:
    return {
        "task": "analyze_ultradian_demodulated_diagnostics",
        "datasets": list(settings.dataset_names),
        "dataset": {"variant": settings.dataset_variant},
        "signals": None if settings.signals is None else list(settings.signals),
        "demodulation": {
            "envelope_window_hours": float(settings.envelope_window_hours),
            "envelope_floor_quantile": float(settings.envelope_floor_quantile),
            "baseline_source": "shift_params_two_harmonic_for_cortisol",
            "non_cortisol_baseline_mode": "signal_specific_two_harmonic_fit",
        },
        "analysis": {
            "bandpass_min_period_hours": float(settings.bandpass_min_period_hours),
            "bandpass_max_period_hours": float(settings.bandpass_max_period_hours),
            "filter_order": int(settings.filter_order),
            "apply_bandpass": bool(settings.apply_bandpass),
            "detrend": bool(settings.detrend),
            "edge_trim_hours": float(settings.edge_trim_hours),
            "amplitude_mask_quantile": float(settings.amplitude_mask_quantile),
            "max_crosscorr_lag_hours": float(settings.max_crosscorr_lag_hours),
            "phase_method": "hilbert_after_bandpass" if settings.apply_bandpass else "hilbert_no_bandpass",
            "export_demodulated_series": bool(settings.export_demodulated_series),
        },
        "plot": {
            "amplitude_hist_bins": int(settings.amplitude_hist_bins),
            "phase_diffusion_bins": int(settings.phase_diffusion_bins),
            "return_map_bins": int(settings.return_map_bins),
            "dpi": int(settings.dpi),
            "title_prefix": settings.title_prefix,
        },
    }


def _signal_map(dataset_name: str) -> dict[str, str]:
    spec = get_dataset_spec(dataset_name)
    return {signal.name: signal.column for signal in spec.signals}


def _dataset_label(dataset_name: str) -> str:
    return DATASET_STYLES.get(dataset_name, {}).get("label", get_dataset_spec(dataset_name).label)


def _selected_signals(dataset_name: str, settings: UltradianDemodulatedDiagnosticsSettings) -> tuple[str, ...]:
    dataset_signals = tuple(signal.name for signal in get_dataset_spec(dataset_name).signals)
    if settings.signals is None:
        return dataset_signals
    return tuple(signal for signal in settings.signals if signal in dataset_signals)


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


def _window_samples(time_min: np.ndarray, window_hours: float) -> int:
    dt_min = _infer_dt_min(time_min)
    samples = max(3, int(round((float(window_hours) * 60.0) / dt_min)))
    if samples % 2 == 0:
        samples += 1
    return samples


def estimate_envelope(
    residual: np.ndarray,
    *,
    time_min: np.ndarray,
    window_hours: float,
    floor_quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = np.asarray(residual, dtype=float)
    window = _window_samples(time_min, window_hours)
    rms = np.sqrt(
        pd.Series(np.square(residual))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    smooth = (
        pd.Series(rms)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    finite = smooth[np.isfinite(smooth)]
    if finite.size == 0:
        floor = 1e-6
    else:
        floor = float(np.quantile(finite, float(floor_quantile)))
        floor = max(floor, 1e-6)
    envelope = np.maximum(smooth, floor)
    return smooth, envelope, floor


def _analytic_signal_reflect(signal: np.ndarray, *, pad_samples: int | None = None) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if signal.size == 0:
        return np.asarray(signal, dtype=complex)
    if signal.size < 4:
        return hilbert(signal)
    if pad_samples is None:
        pad_samples = max(3, min(signal.size // 3, 32))
    pad = int(max(0, min(pad_samples, signal.size - 1)))
    if pad <= 0:
        return hilbert(signal)
    padded = np.pad(signal, pad_width=pad, mode="reflect")
    analytic = hilbert(padded)
    return analytic[pad:-pad]


def _fit_amplitude_models(amplitude: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    amplitude = np.asarray(amplitude, dtype=float)
    amplitude = amplitude[np.isfinite(amplitude) & (amplitude >= 0.0)]
    rows: list[dict[str, object]] = []
    best_model = "none"
    best_aic = float("nan")
    rayleigh_cvm = float("nan")
    delta_aic_gaussian_minus_rayleigh = float("nan")
    if amplitude.size < 3:
        return rows, {
            "best_amplitude_model": best_model,
            "best_amplitude_aic": best_aic,
            "rayleigh_cvm": rayleigh_cvm,
            "delta_aic_gaussian_minus_rayleigh": delta_aic_gaussian_minus_rayleigh,
        }

    rayleigh_loc, rayleigh_scale = rayleigh.fit(amplitude)
    rayleigh_nll = float(-np.sum(rayleigh.logpdf(amplitude, loc=rayleigh_loc, scale=max(rayleigh_scale, 1e-12))))
    rayleigh_aic = float(2.0 * 2 + 2.0 * rayleigh_nll)
    rayleigh_cvm = float(
        cramervonmises(
            amplitude,
            lambda value: rayleigh.cdf(value, loc=rayleigh_loc, scale=max(rayleigh_scale, 1e-12)),
        ).statistic
    )
    rows.append(
        {
            "fit_model": "rayleigh",
            "param_loc": float(rayleigh_loc),
            "param_scale": float(rayleigh_scale),
            "nll": rayleigh_nll,
            "aic": rayleigh_aic,
            "cvm_statistic": rayleigh_cvm,
        }
    )

    gaussian_mu, gaussian_sigma = norm.fit(amplitude)
    gaussian_sigma = max(float(gaussian_sigma), 1e-12)
    gaussian_nll = float(-np.sum(norm.logpdf(amplitude, loc=float(gaussian_mu), scale=gaussian_sigma)))
    gaussian_aic = float(2.0 * 2 + 2.0 * gaussian_nll)
    rows.append(
        {
            "fit_model": "gaussian",
            "param_loc": float(gaussian_mu),
            "param_scale": float(gaussian_sigma),
            "nll": gaussian_nll,
            "aic": gaussian_aic,
            "cvm_statistic": float("nan"),
        }
    )
    delta_aic_gaussian_minus_rayleigh = float(gaussian_aic - rayleigh_aic)

    finite_rows = [row for row in rows if np.isfinite(row["aic"])]
    if finite_rows:
        best = min(finite_rows, key=lambda row: float(row["aic"]))
        best_model = str(best["fit_model"])
        best_aic = float(best["aic"])
    return rows, {
        "best_amplitude_model": best_model,
        "best_amplitude_aic": best_aic,
        "rayleigh_cvm": rayleigh_cvm,
        "delta_aic_gaussian_minus_rayleigh": delta_aic_gaussian_minus_rayleigh,
    }


def _quantile_bin_assignments(values: np.ndarray, n_bins: int) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float))
    if series.nunique(dropna=True) <= 1:
        return pd.Series([0] * len(series), index=series.index, dtype=int)
    bins = pd.qcut(series, q=min(int(n_bins), int(series.nunique(dropna=True))), labels=False, duplicates="drop")
    return bins.astype("Int64")


def summarize_phase_diffusion(
    amplitude: np.ndarray,
    phase_velocity: np.ndarray,
    *,
    n_bins: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    amplitude = np.asarray(amplitude, dtype=float)
    phase_velocity = np.asarray(phase_velocity, dtype=float)
    finite = np.isfinite(amplitude) & np.isfinite(phase_velocity)
    amplitude = amplitude[finite]
    phase_velocity = phase_velocity[finite]
    if amplitude.size < 4:
        return pd.DataFrame(columns=["bin_index", "amplitude_mean", "amplitude_median", "phase_velocity_variance", "n_samples"]), {
            "phase_var_low_amp": float("nan"),
            "phase_var_high_amp": float("nan"),
            "phase_var_low_high_ratio": float("nan"),
        }

    bins = _quantile_bin_assignments(amplitude, n_bins)
    work = pd.DataFrame({"amplitude": amplitude, "phase_velocity": phase_velocity, "bin_index": bins})
    rows: list[dict[str, float]] = []
    for bin_index, group in work.groupby("bin_index", sort=True):
        group = group.dropna(subset=["bin_index"]).copy()
        if group.empty:
            continue
        rows.append(
            {
                "bin_index": int(bin_index),
                "amplitude_mean": float(group["amplitude"].mean()),
                "amplitude_median": float(group["amplitude"].median()),
                "phase_velocity_variance": float(group["phase_velocity"].var(ddof=0)),
                "n_samples": int(len(group)),
            }
        )

    quartiles = _quantile_bin_assignments(amplitude, 4)
    quartile_frame = pd.DataFrame({"amplitude": amplitude, "phase_velocity": phase_velocity, "quartile": quartiles})
    low = quartile_frame.loc[quartile_frame["quartile"] == quartile_frame["quartile"].min(), "phase_velocity"].to_numpy(dtype=float)
    high = quartile_frame.loc[quartile_frame["quartile"] == quartile_frame["quartile"].max(), "phase_velocity"].to_numpy(dtype=float)
    low_var = float(np.var(low, ddof=0)) if low.size else float("nan")
    high_var = float(np.var(high, ddof=0)) if high.size else float("nan")
    ratio = float(low_var / high_var) if np.isfinite(low_var) and np.isfinite(high_var) and high_var > 0.0 else float("nan")
    return pd.DataFrame(rows), {
        "phase_var_low_amp": low_var,
        "phase_var_high_amp": high_var,
        "phase_var_low_high_ratio": ratio,
    }


def summarize_return_map(
    amplitude: np.ndarray,
    *,
    n_bins: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    amplitude = np.asarray(amplitude, dtype=float)
    finite = np.isfinite(amplitude)
    amplitude = amplitude[finite]
    if amplitude.size < 5:
        return pd.DataFrame(columns=["bin_index", "amplitude_mean", "amplitude_median", "delta_a_mean", "n_samples"]), {
            "low_amp_return_slope": float("nan"),
            "return_zero_crossing": float("nan"),
        }

    current_a = amplitude[:-1]
    delta_a = amplitude[1:] - amplitude[:-1]
    finite = np.isfinite(current_a) & np.isfinite(delta_a)
    current_a = current_a[finite]
    delta_a = delta_a[finite]
    if current_a.size < 4:
        return pd.DataFrame(columns=["bin_index", "amplitude_mean", "amplitude_median", "delta_a_mean", "n_samples"]), {
            "low_amp_return_slope": float("nan"),
            "return_zero_crossing": float("nan"),
        }

    bins = _quantile_bin_assignments(current_a, n_bins)
    work = pd.DataFrame({"amplitude": current_a, "delta_a": delta_a, "bin_index": bins})
    rows: list[dict[str, float]] = []
    for bin_index, group in work.groupby("bin_index", sort=True):
        group = group.dropna(subset=["bin_index"]).copy()
        if group.empty:
            continue
        rows.append(
            {
                "bin_index": int(bin_index),
                "amplitude_mean": float(group["amplitude"].mean()),
                "amplitude_median": float(group["amplitude"].median()),
                "delta_a_mean": float(group["delta_a"].mean()),
                "n_samples": int(len(group)),
            }
        )
    bins_df = pd.DataFrame(rows).sort_values("bin_index").reset_index(drop=True)

    q30 = float(np.quantile(current_a, 0.30))
    low_mask = current_a <= q30
    low_slope = float("nan")
    if int(np.sum(low_mask)) >= 2 and np.unique(current_a[low_mask]).size >= 2:
        low_slope = float(np.polyfit(current_a[low_mask], delta_a[low_mask], deg=1)[0])

    zero_crossing = float("nan")
    if not bins_df.empty:
        amp_vals = bins_df["amplitude_mean"].to_numpy(dtype=float)
        delta_vals = bins_df["delta_a_mean"].to_numpy(dtype=float)
        for idx in range(len(bins_df) - 1):
            y0 = float(delta_vals[idx])
            y1 = float(delta_vals[idx + 1])
            if y0 == 0.0:
                zero_crossing = float(amp_vals[idx])
                break
            if y0 * y1 < 0.0:
                x0 = float(amp_vals[idx])
                x1 = float(amp_vals[idx + 1])
                zero_crossing = float(x0 - y0 * (x1 - x0) / (y1 - y0))
                break

    return bins_df, {
        "low_amp_return_slope": low_slope,
        "return_zero_crossing": zero_crossing,
    }


def summarize_demodulated_ipi(phase: np.ndarray, time_min: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(phase) & np.isfinite(time_min)
    if int(np.sum(valid)) < 2:
        return {
            "n_demodulated_ipis": 0,
            "demodulated_ipi_mean_hours": float("nan"),
            "demodulated_ipi_cv": float("nan"),
        }

    phase_work = np.asarray(phase[valid], dtype=float)
    time_hours = np.asarray(time_min[valid], dtype=float) / 60.0

    # Use phase-wrapped cycle boundaries to define demodulated cycle intervals.
    monotone_phase = np.maximum.accumulate(phase_work)
    unique_phase, unique_idx = np.unique(monotone_phase, return_index=True)
    if unique_phase.size < 2:
        return {
            "n_demodulated_ipis": 0,
            "demodulated_ipi_mean_hours": float("nan"),
            "demodulated_ipi_cv": float("nan"),
        }

    cycle_levels = np.arange(
        np.floor(unique_phase[0] / (2.0 * np.pi)) + 1.0,
        np.floor(unique_phase[-1] / (2.0 * np.pi)) + 1.0,
        dtype=float,
    )
    if cycle_levels.size < 2:
        return {
            "n_demodulated_ipis": 0,
            "demodulated_ipi_mean_hours": float("nan"),
            "demodulated_ipi_cv": float("nan"),
        }

    crossing_times_h = np.interp(cycle_levels * 2.0 * np.pi, unique_phase, time_hours[unique_idx])
    ipis_h = np.diff(crossing_times_h)
    ipis_h = ipis_h[np.isfinite(ipis_h) & (ipis_h > 0.0)]
    if ipis_h.size == 0:
        return {
            "n_demodulated_ipis": 0,
            "demodulated_ipi_mean_hours": float("nan"),
            "demodulated_ipi_cv": float("nan"),
        }

    mean_ipi_h = float(np.mean(ipis_h))
    ipi_cv = float("nan")
    if mean_ipi_h > 0.0:
        ipi_cv = float(np.std(ipis_h, ddof=0) / mean_ipi_h)
    return {
        "n_demodulated_ipis": int(ipis_h.size),
        "demodulated_ipi_mean_hours": mean_ipi_h,
        "demodulated_ipi_cv": ipi_cv,
    }


def _tau_grid_from_time(time_min: np.ndarray, max_lag_hours: float) -> tuple[float, ...]:
    dt_min = _infer_dt_min(time_min)
    max_steps = int(np.floor((float(max_lag_hours) * 60.0) / dt_min))
    return tuple(float(step * dt_min) for step in range(max_steps + 1))


def compute_cross_signal_metrics(
    time_min: np.ndarray,
    acth_demodulated: np.ndarray,
    cortisol_demodulated: np.ndarray,
    *,
    settings: UltradianDemodulatedDiagnosticsSettings,
) -> dict[str, float]:
    _, phase_summary = analyze_phase_pair(
        time_min,
        acth_demodulated,
        cortisol_demodulated,
        normalize="raw",
        apply_bandpass=settings.apply_bandpass,
        bandpass_min_period_hours=settings.bandpass_min_period_hours,
        bandpass_max_period_hours=settings.bandpass_max_period_hours,
        filter_order=settings.filter_order,
        detrend_signals=settings.detrend,
        edge_trim_hours=settings.edge_trim_hours,
        amplitude_mask_quantile=settings.amplitude_mask_quantile,
    )
    processed_series, _ = analyze_phase_pair(
        time_min,
        acth_demodulated,
        cortisol_demodulated,
        normalize="raw",
        apply_bandpass=settings.apply_bandpass,
        bandpass_min_period_hours=settings.bandpass_min_period_hours,
        bandpass_max_period_hours=settings.bandpass_max_period_hours,
        filter_order=settings.filter_order,
        detrend_signals=settings.detrend,
        edge_trim_hours=settings.edge_trim_hours,
        amplitude_mask_quantile=settings.amplitude_mask_quantile,
    )
    tau_grid = _tau_grid_from_time(time_min, settings.max_crosscorr_lag_hours)
    crosscorr_df = compute_subject_lagged_correlation(
        processed_series["time_min"].to_numpy(dtype=float),
        processed_series["acth_processed"].to_numpy(dtype=float),
        processed_series["cortisol_processed"].to_numpy(dtype=float),
        taus_min=tau_grid,
        min_pairs=4,
    )
    lag_at_max = float("nan")
    corr_at_max = float("nan")
    valid = crosscorr_df.dropna(subset=["corr"]).copy()
    if not valid.empty:
        peak = valid.loc[valid["corr"].idxmax()]
        lag_at_max = float(peak["tau_min"] / 60.0)
        corr_at_max = float(peak["corr"])
    return {
        "phase_locking_value": float(phase_summary["phase_locking_value"]),
        "mean_phase_lag_hours": float(phase_summary["mean_phase_lag_hours"]),
        "lag_at_max_crosscorr_hours": lag_at_max,
        "max_crosscorr_value": corr_at_max,
    }


def _subject_signal_diagnostics(
    *,
    dataset_name: str,
    dataset_label: str,
    subject_id: object,
    signal_name: str,
    time_min: np.ndarray,
    values: np.ndarray,
    shift_row: dict[str, float | str],
    settings: UltradianDemodulatedDiagnosticsSettings,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline, baseline_method = reconstruct_signal_baseline(
        dataset_name=dataset_name,
        signal_name=signal_name,
        time_min=time_min,
        values=values,
        shift_row=shift_row,
    )
    residual = np.asarray(values, dtype=float) - baseline
    envelope_raw, envelope, envelope_floor = estimate_envelope(
        residual,
        time_min=time_min,
        window_hours=settings.envelope_window_hours,
        floor_quantile=settings.envelope_floor_quantile,
    )
    demodulated = residual / envelope
    dt_min = _infer_dt_min(time_min)
    dt_hours = dt_min / 60.0
    processed = _prepare_signal(
        demodulated,
        normalize="raw",
        apply_detrend=settings.detrend,
        dt_hours=dt_hours,
        apply_bandpass=settings.apply_bandpass,
        min_period_hours=settings.bandpass_min_period_hours,
        max_period_hours=settings.bandpass_max_period_hours,
        filter_order=settings.filter_order,
    )
    analytic = _analytic_signal_reflect(processed)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    phase_velocity = np.gradient(phase, time_min / 60.0)

    trim_samples = int(max(0, ceil(float(settings.edge_trim_hours) / dt_hours)))
    retained = np.ones_like(time_min, dtype=bool)
    if trim_samples > 0:
        retained[:trim_samples] = False
        retained[-trim_samples:] = False
    retained &= np.isfinite(amplitude) & np.isfinite(phase) & np.isfinite(phase_velocity) & np.isfinite(demodulated)

    # Amplitude mask: exclude low-amplitude samples from phase-derived statistics
    # where Hilbert phase is noise-dominated.
    if retained.sum() > 0 and settings.amplitude_mask_quantile > 0.0:
        amp_threshold = float(np.quantile(amplitude[retained], settings.amplitude_mask_quantile))
    else:
        amp_threshold = 0.0
    phase_retained = retained & (amplitude >= amp_threshold)

    retained_amplitude = amplitude[retained]
    amplitude_cv = float("nan")
    if retained_amplitude.size and float(np.mean(retained_amplitude)) > 0.0:
        amplitude_cv = float(np.std(retained_amplitude, ddof=0) / np.mean(retained_amplitude))

    amplitude_fit_rows, amplitude_fit_summary = _fit_amplitude_models(retained_amplitude)
    demodulated_ipi_summary = summarize_demodulated_ipi(phase[phase_retained], time_min[phase_retained])
    phase_bins_df, phase_summary = summarize_phase_diffusion(
        amplitude[phase_retained],
        phase_velocity[phase_retained],
        n_bins=settings.phase_diffusion_bins,
    )
    return_bins_df, return_summary = summarize_return_map(
        retained_amplitude,
        n_bins=settings.return_map_bins,
    )

    summary = {
        "dataset": dataset_name,
        "dataset_label": dataset_label,
        "subject_id": str(subject_id),
        "signal": signal_name,
        "baseline_method": baseline_method,
        "n_samples": int(len(time_min)),
        "retained_samples": int(np.sum(retained)),
        "phase_retained_samples": int(np.sum(phase_retained)),
        "amplitude_mask_quantile": float(settings.amplitude_mask_quantile),
        "dt_min": float(dt_min),
        "baseline_mean": float(np.mean(baseline)),
        "residual_sd": float(np.std(residual, ddof=0)),
        "demodulated_sd": float(np.std(demodulated, ddof=0)),
        "envelope_floor": float(envelope_floor),
        "amplitude_cv": amplitude_cv,
        "n_demodulated_ipis": int(demodulated_ipi_summary["n_demodulated_ipis"]),
        "demodulated_ipi_mean_hours": float(demodulated_ipi_summary["demodulated_ipi_mean_hours"]),
        "demodulated_ipi_cv": float(demodulated_ipi_summary["demodulated_ipi_cv"]),
        "best_amplitude_model": str(amplitude_fit_summary["best_amplitude_model"]),
        "best_amplitude_aic": float(amplitude_fit_summary["best_amplitude_aic"]),
        "rayleigh_cvm": float(amplitude_fit_summary["rayleigh_cvm"]),
        "delta_aic_gaussian_minus_rayleigh": float(amplitude_fit_summary["delta_aic_gaussian_minus_rayleigh"]),
        "phase_var_low_amp": float(phase_summary["phase_var_low_amp"]),
        "phase_var_high_amp": float(phase_summary["phase_var_high_amp"]),
        "phase_var_low_high_ratio": float(phase_summary["phase_var_low_high_ratio"]),
        "low_amp_return_slope": float(return_summary["low_amp_return_slope"]),
        "return_zero_crossing": float(return_summary["return_zero_crossing"]),
        "phase_locking_value": float("nan"),
        "mean_phase_lag_hours": float("nan"),
        "lag_at_max_crosscorr_hours": float("nan"),
        "max_crosscorr_value": float("nan"),
    }

    for row in amplitude_fit_rows:
        row.update(
            {
                "dataset": dataset_name,
                "dataset_label": dataset_label,
                "subject_id": str(subject_id),
                "signal": signal_name,
            }
        )

    if not phase_bins_df.empty:
        phase_bins_df.insert(0, "signal", signal_name)
        phase_bins_df.insert(0, "subject_id", str(subject_id))
        phase_bins_df.insert(0, "dataset_label", dataset_label)
        phase_bins_df.insert(0, "dataset", dataset_name)
    if not return_bins_df.empty:
        return_bins_df.insert(0, "signal", signal_name)
        return_bins_df.insert(0, "subject_id", str(subject_id))
        return_bins_df.insert(0, "dataset_label", dataset_label)
        return_bins_df.insert(0, "dataset", dataset_name)

    series_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "dataset_label": dataset_label,
            "subject_id": str(subject_id),
            "signal": signal_name,
            "time_min": np.asarray(time_min, dtype=float),
            "time_h": np.asarray(time_min, dtype=float) / 60.0,
            "raw_value": np.asarray(values, dtype=float),
            "baseline": baseline,
            "residual": residual,
            "envelope_raw": envelope_raw,
            "envelope": envelope,
            "demodulated": demodulated,
            "processed": processed,
            "amplitude": amplitude,
            "phase": phase,
            "phase_velocity": phase_velocity,
            "retained": retained.astype(int),
        }
    )
    return summary, amplitude_fit_rows, phase_bins_df, return_bins_df, series_df


def collect_diagnostics(
    settings: UltradianDemodulatedDiagnosticsSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    amplitude_fit_rows: list[dict[str, object]] = []
    phase_bin_frames: list[pd.DataFrame] = []
    return_bin_frames: list[pd.DataFrame] = []
    series_frames: list[pd.DataFrame] = []

    for dataset_name in settings.dataset_names:
        dataset_label = _dataset_label(dataset_name)
        spec = get_dataset_spec(dataset_name)
        signal_map = _signal_map(dataset_name)
        frame = load_dataset(dataset_name, settings.dataset_variant).sort_values([spec.id_col, spec.time_col]).copy()
        shift_rows = _load_shift_param_rows(dataset_name, settings.dataset_variant)

        for subject_id, group in frame.groupby(spec.id_col, sort=True):
            normalized_id = _normalize_series_id(spec.id_col, subject_id)
            shift_row = shift_rows.get(normalized_id)
            if shift_row is None:
                raise KeyError(f"Missing shift params for dataset={dataset_name!r}, subject_id={subject_id!r}")

            time_min = group[spec.time_col].to_numpy(dtype=float)
            subject_signal_series: dict[str, pd.DataFrame] = {}
            subject_signal_summaries: dict[str, dict[str, object]] = {}

            for signal_name in _selected_signals(dataset_name, settings):
                values = group[signal_map[signal_name]].to_numpy(dtype=float)
                summary, signal_fit_rows, phase_bins_df, return_bins_df, series_df = _subject_signal_diagnostics(
                    dataset_name=dataset_name,
                    dataset_label=dataset_label,
                    subject_id=subject_id,
                    signal_name=signal_name,
                    time_min=time_min,
                    values=values,
                    shift_row=shift_row,
                    settings=settings,
                )
                summary_rows.append(summary)
                amplitude_fit_rows.extend(signal_fit_rows)
                if not phase_bins_df.empty:
                    phase_bin_frames.append(phase_bins_df)
                if not return_bins_df.empty:
                    return_bin_frames.append(return_bins_df)
                if settings.export_demodulated_series:
                    series_frames.append(series_df)
                subject_signal_series[signal_name] = series_df
                subject_signal_summaries[signal_name] = summary

            if {"ACTH", "Cortisol"}.issubset(subject_signal_series):
                acth_series = subject_signal_series["ACTH"]
                cortisol_series = subject_signal_series["Cortisol"]
                cross_metrics = compute_cross_signal_metrics(
                    acth_series["time_min"].to_numpy(dtype=float),
                    acth_series["demodulated"].to_numpy(dtype=float),
                    cortisol_series["demodulated"].to_numpy(dtype=float),
                    settings=settings,
                )
                for signal_name in ("ACTH", "Cortisol"):
                    subject_signal_summaries[signal_name].update(cross_metrics)

    summary_df = pd.DataFrame(summary_rows)
    amplitude_fit_df = pd.DataFrame(amplitude_fit_rows)
    phase_bins_df = pd.concat(phase_bin_frames, ignore_index=True) if phase_bin_frames else pd.DataFrame()
    return_bins_df = pd.concat(return_bin_frames, ignore_index=True) if return_bin_frames else pd.DataFrame()
    if settings.export_demodulated_series and series_frames:
        series_df = pd.concat(series_frames, ignore_index=True)
    else:
        series_df = pd.DataFrame()
    return summary_df, amplitude_fit_df, phase_bins_df, return_bins_df, series_df


def _pooled_amplitude_values(settings: UltradianDemodulatedDiagnosticsSettings) -> pd.DataFrame:
    pooled_rows: list[pd.DataFrame] = []
    for dataset_name in settings.dataset_names:
        dataset_label = _dataset_label(dataset_name)
        spec = get_dataset_spec(dataset_name)
        signal_map = _signal_map(dataset_name)
        frame = load_dataset(dataset_name, settings.dataset_variant).sort_values([spec.id_col, spec.time_col]).copy()
        shift_rows = _load_shift_param_rows(dataset_name, settings.dataset_variant)
        for subject_id, group in frame.groupby(spec.id_col, sort=True):
            normalized_id = _normalize_series_id(spec.id_col, subject_id)
            shift_row = shift_rows.get(normalized_id)
            if shift_row is None:
                continue
            time_min = group[spec.time_col].to_numpy(dtype=float)
            for signal_name in _selected_signals(dataset_name, settings):
                values = group[signal_map[signal_name]].to_numpy(dtype=float)
                _, _, _, _, series_df = _subject_signal_diagnostics(
                    dataset_name=dataset_name,
                    dataset_label=dataset_label,
                    subject_id=subject_id,
                    signal_name=signal_name,
                    time_min=time_min,
                    values=values,
                    shift_row=shift_row,
                    settings=settings,
                )
                retained = series_df.loc[series_df["retained"] == 1, ["dataset", "dataset_label", "subject_id", "signal", "amplitude"]].copy()
                pooled_rows.append(retained)
    return pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()


def _plot_amplitude_pdf_panels(amplitude_df: pd.DataFrame, settings: UltradianDemodulatedDiagnosticsSettings, output_path: Path) -> None:
    setup_nature_style()
    dataset_names = list(settings.dataset_names)
    nrows = len(SIGNAL_ORDER)
    ncols = len(dataset_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 2.6), squeeze=False)

    for row_idx, signal_name in enumerate(SIGNAL_ORDER):
        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axes[row_idx, col_idx]
            subset = amplitude_df.loc[(amplitude_df["dataset"] == dataset_name) & (amplitude_df["signal"] == signal_name)].copy()
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8.0)
                ax.set_axis_off()
                continue
            values = subset["amplitude"].to_numpy(dtype=float)
            ax.hist(
                values,
                bins=settings.amplitude_hist_bins,
                density=True,
                color=SIGNAL_COLORS[signal_name],
                alpha=0.25,
                edgecolor="white",
                linewidth=0.4,
            )
            x_grid = np.linspace(float(values.min()), float(values.max()), 256)
            rayleigh_loc, rayleigh_scale = rayleigh.fit(values)
            gaussian_mu, gaussian_sigma = norm.fit(values)
            ax.plot(x_grid, rayleigh.pdf(x_grid, loc=rayleigh_loc, scale=max(rayleigh_scale, 1e-12)), color="#1b9e77", lw=1.4, label="Rayleigh")
            ax.plot(x_grid, norm.pdf(x_grid, loc=gaussian_mu, scale=max(gaussian_sigma, 1e-12)), color="#d95f02", lw=1.2, linestyle="--", label="Gaussian")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Density")
            apply_paper_style(ax)
            ax.set_title(f"Dataset: {_dataset_label(dataset_name)} | Hormone: {signal_name}")

    handles = [
        Line2D([0], [0], color="#1b9e77", lw=1.4, label="Rayleigh"),
        Line2D([0], [0], color="#d95f02", lw=1.2, linestyle="--", label="Gaussian"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"{settings.title_prefix}: amplitude PDFs by dataset and hormone")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    fig.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_diffusion_panels(phase_bins_df: pd.DataFrame, settings: UltradianDemodulatedDiagnosticsSettings, output_path: Path) -> None:
    setup_nature_style()
    dataset_names = list(settings.dataset_names)
    nrows = len(SIGNAL_ORDER)
    ncols = len(dataset_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 2.6), squeeze=False)

    for row_idx, signal_name in enumerate(SIGNAL_ORDER):
        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axes[row_idx, col_idx]
            subset = phase_bins_df.loc[(phase_bins_df["dataset"] == dataset_name) & (phase_bins_df["signal"] == signal_name)].copy()
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8.0)
                ax.set_axis_off()
                continue
            summary = (
                subset.groupby("bin_index", sort=True)
                .agg(
                    amplitude_mean=("amplitude_mean", "mean"),
                    variance_mean=("phase_velocity_variance", "mean"),
                )
                .reset_index()
            )
            ax.plot(summary["amplitude_mean"], summary["variance_mean"], color=SIGNAL_COLORS[signal_name], marker="o", lw=1.4)
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Var(dphi/dt)")
            apply_paper_style(ax)
            ax.set_title(f"Dataset: {_dataset_label(dataset_name)} | Hormone: {signal_name}")

    fig.suptitle(f"{settings.title_prefix}: phase diffusion vs amplitude by dataset and hormone")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_return_map_panels(return_bins_df: pd.DataFrame, settings: UltradianDemodulatedDiagnosticsSettings, output_path: Path) -> None:
    setup_nature_style()
    dataset_names = list(settings.dataset_names)
    nrows = len(SIGNAL_ORDER)
    ncols = len(dataset_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 2.6), squeeze=False)

    for row_idx, signal_name in enumerate(SIGNAL_ORDER):
        for col_idx, dataset_name in enumerate(dataset_names):
            ax = axes[row_idx, col_idx]
            subset = return_bins_df.loc[(return_bins_df["dataset"] == dataset_name) & (return_bins_df["signal"] == signal_name)].copy()
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8.0)
                ax.set_axis_off()
                continue
            summary = (
                subset.groupby("bin_index", sort=True)
                .agg(
                    amplitude_mean=("amplitude_mean", "mean"),
                    delta_a_mean=("delta_a_mean", "mean"),
                )
                .reset_index()
            )
            ax.plot(summary["amplitude_mean"], summary["delta_a_mean"], color=SIGNAL_COLORS[signal_name], marker="o", lw=1.4)
            ax.axhline(0.0, color="#666666", lw=0.8, alpha=0.7)
            ax.set_xlabel("A")
            ax.set_ylabel("DeltaA")
            apply_paper_style(ax)
            ax.set_title(f"Dataset: {_dataset_label(dataset_name)} | Hormone: {signal_name}")

    fig.suptitle(f"{settings.title_prefix}: amplitude return maps by dataset and hormone")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(fig)


def _select_representative_ids(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset_name, signal_name), group in summary_df.groupby(["dataset", "signal"], sort=False):
        work = group.dropna(subset=["amplitude_cv"]).copy()
        if work.empty:
            continue
        target = float(work["amplitude_cv"].median())
        work["median_distance"] = (work["amplitude_cv"] - target).abs()
        work = work.sort_values(["median_distance", "subject_id"], ascending=[True, True])
        chosen = work.iloc[0]
        rows.append(
            {
                "dataset": dataset_name,
                "dataset_label": str(chosen["dataset_label"]),
                "signal": signal_name,
                "subject_id": str(chosen["subject_id"]),
                "amplitude_cv": float(chosen["amplitude_cv"]),
            }
        )
    return pd.DataFrame(rows)


def _collect_representative_series(
    settings: UltradianDemodulatedDiagnosticsSettings,
    representative_df: pd.DataFrame,
) -> pd.DataFrame:
    if representative_df.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for _, row in representative_df.iterrows():
        dataset_name = str(row["dataset"])
        signal_name = str(row["signal"])
        subject_id = str(row["subject_id"])
        dataset_label = _dataset_label(dataset_name)
        spec = get_dataset_spec(dataset_name)
        signal_map = _signal_map(dataset_name)
        frame = load_dataset(dataset_name, settings.dataset_variant).sort_values([spec.id_col, spec.time_col]).copy()
        shift_rows = _load_shift_param_rows(dataset_name, settings.dataset_variant)
        if spec.id_col == "ID":
            group = frame.loc[frame[spec.id_col] == int(subject_id)].copy()
            shift_key = int(subject_id)
        else:
            group = frame.loc[frame[spec.id_col].astype(str) == subject_id].copy()
            shift_key = subject_id
        shift_row = shift_rows[shift_key]
        time_min = group[spec.time_col].to_numpy(dtype=float)
        values = group[signal_map[signal_name]].to_numpy(dtype=float)
        _, _, _, _, series_df = _subject_signal_diagnostics(
            dataset_name=dataset_name,
            dataset_label=dataset_label,
            subject_id=subject_id,
            signal_name=signal_name,
            time_min=time_min,
            values=values,
            shift_row=shift_row,
            settings=settings,
        )
        frames.append(series_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_qc_grid(representative_series_df: pd.DataFrame, settings: UltradianDemodulatedDiagnosticsSettings, output_path: Path) -> None:
    if representative_series_df.empty:
        raise ValueError("No representative series available for QC plotting.")
    setup_nature_style()
    keys = list(
        dict.fromkeys(
            zip(
                representative_series_df["dataset"].astype(str),
                representative_series_df["dataset_label"].astype(str),
                representative_series_df["signal"].astype(str),
                representative_series_df["subject_id"].astype(str),
                strict=False,
            )
        )
    )
    ncols = len(keys)
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 3.0, 6.2), sharex=False, squeeze=False)

    for col_idx, (dataset_name, dataset_label, signal_name, subject_id) in enumerate(keys):
        work = representative_series_df.loc[
            (representative_series_df["dataset"] == dataset_name)
            & (representative_series_df["signal"] == signal_name)
            & (representative_series_df["subject_id"].astype(str) == subject_id)
        ].copy()
        time_h = work["time_h"].to_numpy(dtype=float)
        axes[0, col_idx].plot(time_h, work["raw_value"], color=SIGNAL_COLORS[signal_name], lw=1.2, label=signal_name)
        axes[0, col_idx].plot(time_h, work["baseline"], color="#444444", lw=1.0, linestyle="--", label="Baseline")
        axes[0, col_idx].set_ylabel("Raw")
        apply_paper_style(axes[0, col_idx])
        axes[0, col_idx].set_title(f"Dataset: {dataset_label}\nHormone: {signal_name} | ID: {subject_id}")

        axes[1, col_idx].plot(time_h, work["envelope"], color="#7a7a7a", lw=1.1)
        axes[1, col_idx].set_ylabel("Envelope")
        apply_paper_style(axes[1, col_idx])

        axes[2, col_idx].plot(time_h, work["demodulated"], color=SIGNAL_COLORS[signal_name], lw=1.1)
        axes[2, col_idx].axhline(0.0, color="#666666", lw=0.7, alpha=0.7)
        axes[2, col_idx].set_ylabel("z(t)")
        axes[2, col_idx].set_xlabel("Time (h)")
        apply_paper_style(axes[2, col_idx])

    legend_handles = [
        Line2D([0], [0], color="#444444", linestyle="--", lw=1.0, label="Two-harmonic baseline"),
        Line2D([0], [0], color="#7a7a7a", lw=1.1, label="Envelope"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"{settings.title_prefix}: representative demodulation QC")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    fig.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(fig)


def run_ultradian_demodulated_diagnostics(
    settings: UltradianDemodulatedDiagnosticsSettings,
    out_dir: Path,
) -> dict[str, object]:
    _prepare_run_dirs(out_dir)
    logger = _setup_logging(out_dir / "logs" / "run.log")
    (out_dir / "resolved_config.yaml").write_text(dump_yaml(_build_resolved_config(settings)))

    logger.info(
        "Running demodulated ultradian diagnostics for datasets=%s variant=%s",
        ",".join(settings.dataset_names),
        settings.dataset_variant,
    )
    summary_df, amplitude_fit_df, phase_bins_df, return_bins_df, series_df = collect_diagnostics(settings)
    if summary_df.empty:
        raise ValueError("No diagnostics were computed.")

    summary_path = out_dir / "artifacts" / "per_subject_summary.csv"
    amplitude_path = out_dir / "artifacts" / "amplitude_fit_summary.csv"
    phase_path = out_dir / "artifacts" / "phase_diffusion_by_amplitude_bin.csv"
    return_path = out_dir / "artifacts" / "return_map_bins.csv"
    representative_path = out_dir / "artifacts" / "representative_qc_subjects.csv"

    summary_df.to_csv(summary_path, index=False)
    amplitude_fit_df.to_csv(amplitude_path, index=False)
    phase_bins_df.to_csv(phase_path, index=False)
    return_bins_df.to_csv(return_path, index=False)

    representative_df = _select_representative_ids(summary_df)
    representative_df.to_csv(representative_path, index=False)

    artifact_paths = [
        str(summary_path.resolve()),
        str(amplitude_path.resolve()),
        str(phase_path.resolve()),
        str(return_path.resolve()),
        str(representative_path.resolve()),
    ]
    if settings.export_demodulated_series and not series_df.empty:
        series_path = out_dir / "artifacts" / "demodulated_series.csv"
        series_df.to_csv(series_path, index=False)
        artifact_paths.append(str(series_path.resolve()))

    amplitude_values_df = _pooled_amplitude_values(settings)

    amplitude_png = out_dir / "figures" / "amplitude_pdf_panels.png"
    amplitude_pdf = out_dir / "figures" / "amplitude_pdf_panels.pdf"
    phase_png = out_dir / "figures" / "phase_diffusion_vs_amplitude_panels.png"
    phase_pdf = out_dir / "figures" / "phase_diffusion_vs_amplitude_panels.pdf"
    return_png = out_dir / "figures" / "delta_a_vs_a_panels.png"
    return_pdf = out_dir / "figures" / "delta_a_vs_a_panels.pdf"
    qc_png = out_dir / "figures" / "qc_demodulation_grid.png"
    qc_pdf = out_dir / "figures" / "qc_demodulation_grid.pdf"

    _plot_amplitude_pdf_panels(amplitude_values_df, settings, amplitude_png)
    _plot_amplitude_pdf_panels(amplitude_values_df, settings, amplitude_pdf)
    _plot_phase_diffusion_panels(phase_bins_df, settings, phase_png)
    _plot_phase_diffusion_panels(phase_bins_df, settings, phase_pdf)
    _plot_return_map_panels(return_bins_df, settings, return_png)
    _plot_return_map_panels(return_bins_df, settings, return_pdf)
    representative_series_df = _collect_representative_series(settings, representative_df)
    _plot_qc_grid(representative_series_df, settings, qc_png)
    _plot_qc_grid(representative_series_df, settings, qc_pdf)

    figure_paths = [
        str(amplitude_png.resolve()),
        str(amplitude_pdf.resolve()),
        str(phase_png.resolve()),
        str(phase_pdf.resolve()),
        str(return_png.resolve()),
        str(return_pdf.resolve()),
        str(qc_png.resolve()),
        str(qc_pdf.resolve()),
    ]

    manifest = {
        "task": "analyze_ultradian_demodulated_diagnostics",
        "datasets": list(settings.dataset_names),
        "variant": settings.dataset_variant,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "run_dir": str(out_dir.resolve()),
        "figures": figure_paths,
        "artifacts": artifact_paths,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    readme_lines = [
        "# analyze_ultradian_demodulated_diagnostics",
        "",
        "## Highlights",
        f"- Datasets: `{', '.join(settings.dataset_names)}`",
        f"- Variant: `{settings.dataset_variant}`",
        f"- Circadian baseline source: shifted two-harmonic sidecars for cortisol, signal-specific two-harmonic fit for non-cortisol signals.",
        f"- Envelope window: `{settings.envelope_window_hours:.2f} h`, floor quantile: `{settings.envelope_floor_quantile:.2f}`",
        f"- Ultradian bandpass: `{settings.bandpass_min_period_hours:.2f}` to `{settings.bandpass_max_period_hours:.2f}` h",
        "- Per-subject diagnostics include amplitude CV, amplitude-model fits, phase diffusion vs amplitude, and amplitude return-map summaries.",
        "",
        "## Outputs",
        "- `artifacts/per_subject_summary.csv`",
        "- `artifacts/amplitude_fit_summary.csv`",
        "- `artifacts/phase_diffusion_by_amplitude_bin.csv`",
        "- `artifacts/return_map_bins.csv`",
        "- `artifacts/representative_qc_subjects.csv`",
        "- `figures/amplitude_pdf_panels.png`",
        "- `figures/phase_diffusion_vs_amplitude_panels.png`",
        "- `figures/delta_a_vs_a_panels.png`",
        "- `figures/qc_demodulation_grid.png`",
    ]
    if settings.export_demodulated_series:
        readme_lines.insert(-4, "- `artifacts/demodulated_series.csv`")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n")
    logger.info("Completed demodulated ultradian diagnostics into %s", out_dir)
    return manifest
