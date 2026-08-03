"""Rayleigh fit on peak amplitudes detected directly on the z-scored residual.

Simplified pipeline (no bandpass, no detrend):
  1. Subtract the per-subject two-harmonic circadian baseline.
  2. Z-score the residual within each series (normalises for cross-subject pooling).
  3. Detect peaks on the z-scored residual using a fixed sigma prominence threshold.
  4. Pool peak amplitudes across all series / datasets.
  5. Fit two Rayleigh distributions — one with loc fixed at 0 (positives only)
     and one with loc free (full sample) — and run a KS test for each.

The amplitude unit is within-subject residual standard deviations, making the
Rayleigh shape a genuine claim about pulse-size distribution rather than a
filter-envelope artefact.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kstest, rayleigh

from ...config import dump_yaml
from ...data.registry import PROJECT_ROOT, get_dataset_spec, load_dataset
from ...plotting import apply_paper_style, setup_nature_style
from ..plotting.ultradian_demodulated_diagnostics import (
    _load_shift_param_rows,
    _normalize_series_id,
    reconstruct_signal_baseline,
)


DEFAULT_DATASETS = ("habs", "digitize_2019", "all_digitized")
DEFAULT_SIGNAL = "Cortisol"
DEFAULT_VARIANT = "shifted_12h"
DEFAULT_OUT = Path("experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom03")

SIGNAL_COLORS: dict[str, str] = {
    "ACTH": "#8C8C8C",
    "Cortisol": "#1A1A1A",
}


@dataclass(frozen=True)
class DirectRayleighSettings:
    dataset_names: tuple[str, ...] = DEFAULT_DATASETS
    dataset_variant: str = DEFAULT_VARIANT
    signal: str = DEFAULT_SIGNAL
    pool_across_datasets: bool = True
    min_distance_min: float = 60.0
    prom_sigma: float = 0.3
    drop_negative: bool = True
    subtract_baseline: bool = True
    dpi: int = 300


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model.peak_amplitude_direct_rayleigh_fit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, check=True, text=True, cwd=PROJECT_ROOT,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _zscore_series(values: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 2:
        return None
    mean = float(np.mean(values[finite]))
    std = float(np.std(values[finite]))
    if not np.isfinite(std) or std <= 0.0:
        return None
    out = np.full_like(values, np.nan, dtype=float)
    out[finite] = (values[finite] - mean) / std
    return out, mean, std


def _resolve_signal_column(dataset_name: str, signal_name: str) -> str:
    spec = get_dataset_spec(dataset_name)
    for signal in spec.signals:
        if signal.name == signal_name:
            return signal.column
    raise KeyError(f"Signal {signal_name!r} not in dataset {dataset_name!r}")


def _process_series(
    dataset_name: str,
    series_id: str,
    variant: str,
    signal_name: str,
    *,
    min_distance_min: float,
    prom_sigma: float,
    subtract_baseline: bool = True,
) -> pd.DataFrame | None:
    """Return peak rows for one series, or None if the series cannot be processed."""
    spec = get_dataset_spec(dataset_name)
    df = load_dataset(dataset_name, variant).sort_values([spec.id_col, spec.time_col])
    shift_rows = _load_shift_param_rows(dataset_name, variant)
    norm_id = _normalize_series_id(spec.id_col, series_id)
    shift_row = shift_rows.get(norm_id)
    if subtract_baseline and shift_row is None:
        return None

    value_col = _resolve_signal_column(dataset_name, signal_name)
    group = df[df[spec.id_col].astype(str) == str(series_id)].copy()
    time_min = group[spec.time_col].to_numpy(dtype=float)
    raw = group[value_col].to_numpy(dtype=float)
    valid = np.isfinite(time_min) & np.isfinite(raw)
    if int(np.sum(valid)) < 10:
        return None
    time_min = time_min[valid]
    raw = raw[valid]

    if subtract_baseline:
        baseline, _ = reconstruct_signal_baseline(
            dataset_name=dataset_name,
            signal_name=signal_name,
            time_min=time_min,
            values=raw,
            shift_row=shift_row,
        )
    else:
        # Skip circadian subtraction; the prev-dip metric is a short-window
        # local difference and is largely invariant to slow circadian drift.
        baseline = np.zeros_like(raw)
    residual = raw - baseline

    z_out = _zscore_series(residual)
    if z_out is None:
        return None
    residual_z, res_mean, res_std = z_out

    # detect peaks directly on z-scored residual — no bandpass, no detrend
    dt = float(np.median(np.diff(time_min))) if time_min.size > 1 else 1.0
    min_dist_samples = max(1, int(round(min_distance_min / dt)))
    # prom_sigma is a fixed threshold in sigma units (std(residual_z) ≈ 1)
    peaks, props = find_peaks(residual_z, distance=min_dist_samples, prominence=prom_sigma)
    if peaks.size == 0:
        return None
    prominences = props["prominences"][: peaks.size]

    # Peak amplitude measured against the previous trough: for peak i, take the
    # minimum of residual_z between the prior peak (or the start of the trace
    # for i=0) and this peak. Falls back to 0 if no prior samples exist.
    prev_dip_z = np.empty(peaks.size, dtype=float)
    for i, p in enumerate(peaks):
        if i == 0:
            prev_dip_z[i] = float(residual_z[:p].min()) if p > 0 else 0.0
        else:
            prev_dip_z[i] = float(residual_z[peaks[i - 1]: p].min())
    peak_amp_prev_dip_sigma = residual_z[peaks] - prev_dip_z

    return pd.DataFrame({
        "dataset": dataset_name,
        "dataset_label": spec.label,
        "signal": signal_name,
        "series_id": str(series_id),
        "series_uid": f"{dataset_name}:{series_id}",
        "time_min": time_min[peaks],
        "peak_idx": peaks.astype(int),
        "peak_amplitude_sigma": residual_z[peaks],      # amplitude in within-subject sigma
        "peak_amplitude_raw": residual[peaks],           # amplitude in raw units (for reference)
        "residual_std_raw": res_std,                     # the normalisation factor
        "baseline_raw": baseline[peaks],
        "prominence": prominences,
        "prev_dip_sigma": prev_dip_z,
        "peak_amplitude_prev_dip_sigma": peak_amp_prev_dip_sigma,
    })


def _collect_peaks(settings: DirectRayleighSettings) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dataset_name in settings.dataset_names:
        spec = get_dataset_spec(dataset_name)
        if not any(s.name == settings.signal for s in spec.signals):
            continue
        frame = load_dataset(dataset_name, settings.dataset_variant)
        for series_id, _ in frame.groupby(spec.id_col, sort=True):
            try:
                df = _process_series(
                    dataset_name, str(series_id), settings.dataset_variant,
                    settings.signal,
                    min_distance_min=settings.min_distance_min,
                    prom_sigma=settings.prom_sigma,
                    subtract_baseline=settings.subtract_baseline,
                )
            except Exception:
                continue
            if df is not None and not df.empty:
                rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fit_rayleigh_loc0(amplitudes: np.ndarray) -> dict[str, float]:
    # Mirror scripts/build_figure1_abc.py: Rayleigh with loc=0 has support
    # (0, ∞); always fit on strictly positive amplitudes regardless of whether
    # negatives are retained for the histogram.
    values = amplitudes[np.isfinite(amplitudes)]
    positives = values[values > 0.0]
    if positives.size < 2:
        raise ValueError("Need at least two positive finite amplitudes.")
    loc, scale = rayleigh.fit(positives, floc=0.0)
    ks_stat, ks_pvalue = kstest(positives, "rayleigh", args=(loc, scale))
    return {
        "loc": float(loc),
        "scale": float(scale),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "log_likelihood": float(np.sum(rayleigh.logpdf(positives, loc=loc, scale=scale))),
        "n_total": int(values.size),
        "n_positive_used": int(positives.size),
        "n_nonpositive": int(values.size - positives.size),
    }


def _fit_rayleigh_freeloc(amplitudes: np.ndarray) -> dict[str, float]:
    # Free-loc Rayleigh ("shifted Rayleigh"): support is [loc, ∞); fit on the
    # full finite sample (negatives allowed, since loc can be < 0).
    values = amplitudes[np.isfinite(amplitudes)]
    if values.size < 2:
        raise ValueError("Need at least two finite amplitudes.")
    loc, scale = rayleigh.fit(values)
    ks_stat, ks_pvalue = kstest(values, "rayleigh", args=(loc, scale))
    return {
        "loc": float(loc),
        "scale": float(scale),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "log_likelihood": float(np.sum(rayleigh.logpdf(values, loc=loc, scale=scale))),
        "n_used": int(values.size),
    }


def _build_summary(points: pd.DataFrame, settings: DirectRayleighSettings) -> pd.DataFrame:
    rows: list[dict] = []
    groups = points.groupby("signal", sort=False) if settings.pool_across_datasets else points.groupby(["dataset", "signal"], sort=False)
    for key, group in groups:
        signal_name = key if settings.pool_across_datasets else key[1]
        dataset_key = "pooled" if settings.pool_across_datasets else key[0]
        all_amps = group["peak_amplitude_sigma"].to_numpy(dtype=float)
        all_amps = all_amps[np.isfinite(all_amps)]
        # As in scripts/build_figure1_abc.py: drop_negative controls the histogram
        # sample (and reported mean/std/CV); the Rayleigh fit always uses positives.
        display_amps = all_amps[all_amps > 0.0] if settings.drop_negative else all_amps
        fit_loc0 = _fit_rayleigh_loc0(all_amps)
        fit_free = _fit_rayleigh_freeloc(all_amps)
        cv = float(display_amps.std(ddof=1) / display_amps.mean()) if display_amps.size > 1 else 0.0
        rows.append({
            "dataset": dataset_key,
            "signal": signal_name,
            "n_peaks": int(display_amps.size),
            "n_series": int(group["series_uid"].nunique()),
            "mean_amplitude_sigma": float(display_amps.mean()) if display_amps.size else 0.0,
            "std_amplitude_sigma": float(display_amps.std(ddof=1)) if display_amps.size > 1 else 0.0,
            "cv_amplitude_sigma": cv,
            **{f"rayleigh_loc0_{k}": v for k, v in fit_loc0.items()},
            **{f"rayleigh_freeloc_{k}": v for k, v in fit_free.items()},
        })
    return pd.DataFrame(rows)


def _plot(points: pd.DataFrame, summary: pd.DataFrame, settings: DirectRayleighSettings, out_path: Path) -> None:
    setup_nature_style()
    n_panels = len(summary)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)

    for ax, row in zip(axes.ravel()[:n_panels], summary.to_dict(orient="records")):
        signal_name = str(row["signal"])
        if settings.pool_across_datasets:
            subset = points.loc[points["signal"] == signal_name]
            title = f"{signal_name} pooled across all datasets"
        else:
            subset = points.loc[(points["dataset"] == row["dataset"]) & (points["signal"] == signal_name)]
            title = f"{row['dataset']} {signal_name}"

        all_amps = subset["peak_amplitude_sigma"].to_numpy(dtype=float)
        all_amps = all_amps[np.isfinite(all_amps)]
        # Mirror scripts/build_figure1_abc.py: drop_negative filters the histogram
        # sample; the loc=0 Rayleigh PDF is scaled by the positive-mass fraction so
        # it matches a density histogram that may include negatives.
        vals = all_amps[all_amps > 0.0] if settings.drop_negative else all_amps
        pos = all_amps[all_amps > 0.0]
        color = SIGNAL_COLORS.get(signal_name, "#4c4c4c")

        ax.hist(vals, bins=20, density=True, color=color, alpha=0.30, edgecolor="white", linewidth=0.7)

        x_grid = np.linspace(float(vals.min()) - 0.1, float(vals.max()) + 0.3, 512)

        # loc=0 fit: support (0, ∞); scale by positive-mass fraction so the curve
        # matches the density histogram if it includes negatives.
        scale_pdf = (len(pos) / len(vals)) if len(vals) else 1.0
        pdf_loc0 = rayleigh.pdf(
            x_grid, loc=row["rayleigh_loc0_loc"], scale=row["rayleigh_loc0_scale"]
        ) * scale_pdf
        pdf_loc0[x_grid < 0] = 0.0
        ax.plot(x_grid, pdf_loc0, color="#C62828", linewidth=2.0, linestyle="--",
                label=f"Rayleigh (loc=0)  KS p={row['rayleigh_loc0_ks_pvalue']:.2g}")

        # free-loc fit: support [loc, ∞); use the full sample directly.
        pdf_free = rayleigh.pdf(
            x_grid, loc=row["rayleigh_freeloc_loc"], scale=row["rayleigh_freeloc_scale"]
        )
        pdf_free[x_grid < row["rayleigh_freeloc_loc"]] = 0.0
        ax.plot(x_grid, pdf_free, color=color, linewidth=2.0,
                label=f"Rayleigh (loc free)  KS p={row['rayleigh_freeloc_ks_pvalue']:.2g}")
        ax.axvline(row["rayleigh_freeloc_loc"], color="#4a4a4a", linestyle=":",
                   linewidth=1.0, alpha=0.8)

        ax.text(0.98, 0.98,
                "\n".join([
                    f"n = {int(row['n_peaks'])}",
                    f"n series = {int(row['n_series'])}",
                    f"CV = {row['cv_amplitude_sigma']:.3f}",
                    f"loc=0 scale = {row['rayleigh_loc0_scale']:.3f}",
                    f"free  loc   = {row['rayleigh_freeloc_loc']:.3f}",
                    f"free scale  = {row['rayleigh_freeloc_scale']:.3f}",
                ]),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0})

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Peak amplitude (within-subject σ)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        apply_paper_style(ax)

    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=settings.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(settings: DirectRayleighSettings, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    logger = _setup_logging(out_dir / "logs" / "run.log")

    resolved = {
        "task": "peak_amplitude_direct_rayleigh_fit",
        "datasets": list(settings.dataset_names),
        "dataset_variant": settings.dataset_variant,
        "signal": settings.signal,
        "pool_across_datasets": settings.pool_across_datasets,
        "pipeline": "baseline_subtract → zscore_residual → find_peaks (no bandpass, no detrend)",
        "peak_detection": {
            "min_distance_min": settings.min_distance_min,
            "prominence_sigma": settings.prom_sigma,
        },
        "amplitude_unit": "within_subject_residual_sigma",
        "drop_negative": settings.drop_negative,
        "fit": {
            "distribution": "rayleigh",
            "variants": {
                "loc0": {"loc": "fixed_at_0", "fit_sample": "positives_only"},
                "freeloc": {"loc": "free", "fit_sample": "all_finite"},
            },
        },
    }
    (out_dir / "resolved_config.yaml").write_text(dump_yaml(resolved))
    manifest = {
        "task": "peak_amplitude_direct_rayleigh_fit",
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(out_dir.resolve()),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Collecting peaks from %s", ", ".join(settings.dataset_names))
    points = _collect_peaks(settings)
    if points.empty:
        raise ValueError("No peaks collected.")

    summary = _build_summary(points, settings)

    peaks_path = out_dir / "artifacts" / "peak_amplitude_samples.csv"
    summary_path = out_dir / "artifacts" / "rayleigh_fit_summary.csv"
    points.to_csv(peaks_path, index=False)
    summary.to_csv(summary_path, index=False)

    png_path = out_dir / "figures" / "peak_amplitude_direct_rayleigh_fit.png"
    pdf_path = out_dir / "figures" / "peak_amplitude_direct_rayleigh_fit.pdf"
    _plot(points, summary, settings, png_path)
    _plot(points, summary, settings, pdf_path)

    manifest["figures"] = [str(png_path), str(pdf_path)]
    manifest["artifacts"] = [str(peaks_path), str(summary_path)]
    logger.info("Done. %d peaks from %d series.", int(summary["n_peaks"].sum()), int(summary["n_series"].sum()))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT)
    p.add_argument("--signal", type=str, default=DEFAULT_SIGNAL)
    p.add_argument("--dataset", dest="dataset_names", action="append", default=None)
    p.add_argument("--pool-datasets", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-distance-min", type=float, default=60.0)
    p.add_argument("--prom-sigma", type=float, default=0.3,
                   help="Peak prominence threshold in within-subject residual sigma units.")
    p.add_argument("--drop-negative", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop non-positive z-scored peaks from histogram/CV (Rayleigh fit always uses positives).")
    p.add_argument("--subtract-baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="Subtract the two-harmonic circadian baseline before z-scoring "
                        "(disable with --no-subtract-baseline to operate on raw z-scored signal).")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = DirectRayleighSettings(
        dataset_names=tuple(args.dataset_names) if args.dataset_names else DEFAULT_DATASETS,
        dataset_variant=args.variant,
        signal=args.signal,
        pool_across_datasets=args.pool_datasets,
        min_distance_min=args.min_distance_min,
        prom_sigma=args.prom_sigma,
        drop_negative=args.drop_negative,
        subtract_baseline=args.subtract_baseline,
        dpi=args.dpi,
    )
    manifest = run(settings, args.out)
    for p in manifest["figures"]:
        print(f"Saved {p}")
    for p in manifest["artifacts"]:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
