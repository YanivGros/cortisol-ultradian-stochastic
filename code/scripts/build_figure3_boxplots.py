"""Figure 3 (boxplot style): per-bin peak statistics, data vs model.

Modelled on ``experiments/scripts/plot_per_bin_stats_data_vs_model_multidataset.py``
but adapted to the *new* canonical pipeline:

* 5-bin time-of-day layout (04-08, 08-12, 12-16, 16-20, **20-04**) — morning-first,
  with a single contiguous overnight bin (20:00-04:00) placed last
* Amplitude metric = peak Z-score − previous-trough Z-score (column
  ``peak_amplitude_prev_dip_sigma`` in the prom05 peaks CSV)
* Prominence threshold 0.5σ on the z-scored residual
* Data peaks pre-extracted (read from CSV); model peaks simulated here

The figure is a 2×2 grid (Amp mean, Amp CV, IPI mean, IPI CV) with **two
boxplots per bin**: blue = data per-subject values, salmon = model
per-(subject × replicate) values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpa_model.data.two_harmonic_shift import (
    evaluate_two_harmonic, fit_two_harmonic_params,
)
from hpa_model.model.three_state_gr_delay import (
    ThreeStateGRDelayModel, TwoHarmonicNoiseDrive, TwoHarmonicDrive,
)
from hpa_model.plotting import apply_paper_style, setup_nature_style
from hpa_model.simulate.engine import simulate_trajectory_fit_arrays


BIN_EDGES  = [0.0, 240.0, 480.0, 720.0, 960.0, 1440.0]
# Morning-first bins on a clock shifted to a 04:00 origin, so the overnight
# window 20:00-04:00 forms a single contiguous (wrapping) bin placed last.
BIN_LABELS = ["04-08", "08-12", "12-16", "16-20", "20-04"]
TOD_ORIGIN_MIN = 240.0  # shift so 04:00 -> 0 and 20:00-04:00 is the last bin

DATA_COLOR  = "#2F5C85"
MODEL_COLOR = "#C85C3A"


def _bin_idx(t_min: float) -> int:
    tod = (t_min % 1440.0 - TOD_ORIGIN_MIN) % 1440.0
    return int(np.digitize(tod, np.asarray(BIN_EDGES), right=False) - 1)


def _baseline_subtract_residual_z(
    times_min: np.ndarray, values: np.ndarray,
) -> np.ndarray:
    params = fit_two_harmonic_params(times_min, values,
                                     period_min=1440.0, second_period_min=720.0)
    if params is None:
        baseline = np.full_like(values, float(np.nanmean(values)))
    else:
        baseline = evaluate_two_harmonic(times_min, params)
    residual = values - baseline
    std = float(np.nanstd(residual))
    if std <= 0:
        return np.zeros_like(residual), baseline
    return (residual - float(np.nanmean(residual))) / std, baseline


def _detect_prev_dip_amps(
    times_min: np.ndarray, residual_z: np.ndarray,
    *, prom_sigma: float, min_distance_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    if times_min.size < 2:
        return np.empty(0), np.empty(0)
    dt = float(np.median(np.diff(times_min)))
    min_dist_samples = max(1, int(round(min_distance_min / dt)))
    peaks, _ = find_peaks(residual_z, distance=min_dist_samples, prominence=prom_sigma)
    if peaks.size == 0:
        return np.empty(0), np.empty(0)
    prev_dip = np.empty(peaks.size, dtype=float)
    for i, p in enumerate(peaks):
        lo = 0 if i == 0 else int(peaks[i - 1])
        prev_dip[i] = float(residual_z[lo:p].min()) if p > lo else 0.0
    amps = residual_z[peaks] - prev_dip
    return times_min[peaks], amps


def _peaks_to_rows(peak_times, peak_amps, *, uid) -> list[dict]:
    if peak_times.size == 0:
        return []
    order = np.argsort(peak_times)
    pt = peak_times[order]; pa = peak_amps[order]
    ipis = np.concatenate((np.diff(pt), [np.nan]))
    out = []
    for t, a, ipi in zip(pt, pa, ipis):
        bi = _bin_idx(float(t))
        if not (0 <= bi < len(BIN_LABELS)):
            continue
        out.append({
            "series_uid": uid, "bin": BIN_LABELS[bi],
            "time_min": float(t), "amp": float(a),
            "ipi": float(ipi) if np.isfinite(ipi) else np.nan,
        })
    return out


def _load_data_peaks(peaks_csv: Path) -> pd.DataFrame:
    """Re-use the pre-computed prom05 prev_dip peaks (already pipeline-correct)."""
    df = pd.read_csv(peaks_csv)
    df = df[np.isfinite(df["peak_amplitude_prev_dip_sigma"])].copy()
    df["tod"] = (df["time_min"] % 1440.0 - TOD_ORIGIN_MIN) % 1440.0
    df["bin"] = pd.cut(df["tod"], bins=BIN_EDGES, labels=BIN_LABELS,
                       right=True, include_lowest=True).astype(str)
    df["amp"] = df["peak_amplitude_prev_dip_sigma"]
    df = df.sort_values(["series_uid", "time_min"])
    # IPI = time to NEXT peak, attributed to the first peak in the pair
    # (matches build_figure2_peak_stats.py convention).
    df["ipi"] = df.groupby("series_uid")["time_min"].shift(-1) - df["time_min"]
    return df[["series_uid", "bin", "time_min", "amp", "ipi"]].reset_index(drop=True)


def _simulate_model_peaks(
    fit_dir: Path, *,
    n_subjects: int, n_reps: int,
    base_seed: int, prom_sigma: float, min_distance_min: float,
    resample_dt_min: float,
) -> pd.DataFrame:
    cfg = yaml.safe_load((fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    mp = cfg["model"]["params"]
    model = ThreeStateGRDelayModel(
        a1=float(mp["a1"]), a2=float(mp["a2"]), a3=float(mp["a3"]),
        b1=float(mp["b1"]), b2=float(mp["b2"]), b3=float(mp["b3"]),
        kgr=float(mp["kgr"]), tau_min=float(mp.get("tau_min", 0.0)),
        x3_floor=float(mp.get("x3_floor", 0.01)),
        hill_coeff=float(mp.get("hill_coeff", 3.0)),
        initial_state=tuple(float(x) for x in mp["initial_state"]),
    )
    dp = {k: v for k, v in cfg["drive"]["params"].items()
          if k not in ("dataset", "series_id")}
    drive_kind = str(cfg["drive"]["kind"])
    if drive_kind == "two_harmonic_noise":
        drive = TwoHarmonicNoiseDrive(
            a24=float(dp["a24"]), phase24=float(dp["phase24"]),
            a12=float(dp["a12"]), phase12=float(dp["phase12"]),
            baseline=float(dp.get("baseline", 1.0)),
            epsilon=float(dp.get("epsilon", 0.0)),
            period_min=float(dp.get("period_min", 1440.0)),
            second_period_min=float(dp.get("second_period_min", 720.0)),
            noise_form=str(dp.get("noise_form", "multiplicative")),
        )
    else:
        drive = TwoHarmonicDrive(
            a24=float(dp["a24"]), phase24=float(dp["phase24"]),
            a12=float(dp["a12"]), phase12=float(dp["phase12"]),
            baseline=float(dp.get("baseline", 1.0)),
            period_min=float(dp.get("period_min", 1440.0)),
            second_period_min=float(dp.get("second_period_min", 720.0)),
        )

    solver = cfg["solver"]
    runtime = cfg.get("runtime", {})
    noise_form = str(runtime.get("noise_form", "multiplicative"))
    noise_locations = list(runtime.get("noise_locations", []) or [])
    noise_epsilons = dict(runtime.get("noise_epsilons", {}) or {})

    target_grid = np.arange(0.0, float(solver["duration_min"]) + 1e-9,
                            float(resample_dt_min))
    rows: list[dict] = []
    for s_idx in range(n_subjects):
        for rep in range(n_reps):
            seed = base_seed + s_idx * 10_000 + rep
            sim = simulate_trajectory_fit_arrays(
                model, drive,
                dt_min=float(solver["dt_min"]),
                warmup_min=float(solver["warmup_min"]),
                duration_min=float(solver["duration_min"]),
                seed=seed,
                noise_locations=noise_locations,
                noise_epsilons=noise_epsilons,
                noise_form=noise_form,
            )
            x3 = np.interp(target_grid, sim["time_min"], sim["x3"])
            residual_z, _ = _baseline_subtract_residual_z(target_grid, x3)
            pt, pa = _detect_prev_dip_amps(
                target_grid, residual_z,
                prom_sigma=prom_sigma, min_distance_min=min_distance_min,
            )
            rows.extend(_peaks_to_rows(pt, pa, uid=f"model:s{s_idx}:r{rep}"))
    return pd.DataFrame(rows)


def _per_uid_bin_stat(df, value_col, stat, min_peaks) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["series_uid", "bin", "value"])
    out = []
    for (uid, b), grp in df.groupby(["series_uid", "bin"], observed=True):
        v = grp[value_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if value_col == "amp":
            v = v[v > 0]
        if len(v) < min_peaks:
            continue
        if stat == "mean":
            value = float(np.mean(v))
        else:  # cv
            m = float(np.mean(v))
            value = float(np.std(v, ddof=1) / m) if m > 0 else np.nan
        out.append({"series_uid": uid, "bin": str(b), "value": value})
    return pd.DataFrame(out)


def _box_pair(ax, data_long, model_long, *, ylabel, title,
              connect_trend=True, ylim=None):
    pos = np.arange(len(BIN_LABELS))
    width = 0.36
    data_vals = [data_long.loc[data_long["bin"] == b, "value"].dropna().to_numpy()
                 for b in BIN_LABELS]
    model_vals = [model_long.loc[model_long["bin"] == b, "value"].dropna().to_numpy()
                  for b in BIN_LABELS]

    ax.boxplot(
        [v if len(v) else np.array([np.nan]) for v in data_vals],
        positions=pos - width / 2, widths=width, patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": DATA_COLOR, "alpha": 0.55, "linewidth": 0.7,
                  "edgecolor": DATA_COLOR},
        whiskerprops={"color": DATA_COLOR, "linewidth": 0.7},
        capprops={"color": DATA_COLOR, "linewidth": 0.7},
    )
    ax.boxplot(
        [v if len(v) else np.array([np.nan]) for v in model_vals],
        positions=pos + width / 2, widths=width, patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": MODEL_COLOR, "alpha": 0.55, "linewidth": 0.7,
                  "edgecolor": MODEL_COLOR},
        whiskerprops={"color": MODEL_COLOR, "linewidth": 0.7},
        capprops={"color": MODEL_COLOR, "linewidth": 0.7},
    )

    # Trend lines connecting the per-bin means (data + model) — matches Fig 2.
    if connect_trend:
        d_means = [float(np.mean(v)) if len(v) else np.nan for v in data_vals]
        m_means = [float(np.mean(v)) if len(v) else np.nan for v in model_vals]
        ax.plot(pos - width / 2, d_means, color=DATA_COLOR, lw=1.8, zorder=5,
                marker="o", markersize=4.5, markerfacecolor="white",
                markeredgecolor=DATA_COLOR, markeredgewidth=1.2)
        ax.plot(pos + width / 2, m_means, color=MODEL_COLOR, lw=1.8, zorder=5,
                marker="o", markersize=4.5, markerfacecolor="white",
                markeredgecolor=MODEL_COLOR, markeredgewidth=1.2)

    ax.set_xticks(pos)
    ax.set_xticklabels(BIN_LABELS, fontsize=10.5, rotation=30, ha="right")
    ax.set_xlabel("Time of day (h)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12.5, loc="left", pad=4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # ylim based on whisker extents (Q1-1.5·IQR, Q3+1.5·IQR clipped to data),
        # so removing fliers leaves the visible boxes well-framed.
        all_vals = [v for v in (data_vals + model_vals) if len(v)]
        if all_vals:
            lows, highs = [], []
            for v in all_vals:
                q1, q3 = np.percentile(v, [25, 75])
                iqr = q3 - q1
                lows.append(max(float(np.min(v)), float(q1 - 1.5 * iqr)))
                highs.append(min(float(np.max(v)), float(q3 + 1.5 * iqr)))
            lo = min(lows); hi = max(highs)
            margin = (hi - lo) * 0.10 if hi > lo else 1.0
            non_neg = ("CV" in ylabel) or ("Z" in ylabel) or ("min" in ylabel)
            ax.set_ylim(max(0.0, lo - margin) if non_neg else lo - margin,
                        hi + margin)
    apply_paper_style(ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v4")
    ap.add_argument("--peaks-csv", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv")
    ap.add_argument("--n-subjects", type=int, default=20,
                    help="Number of independent model 'subjects' (each gets its own seed range).")
    ap.add_argument("--n-reps", type=int, default=10,
                    help="Replicates per subject.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--resample-dt-min", type=float, default=20.0)
    ap.add_argument("--trend-line", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Draw the line (and mean markers) connecting per-bin means "
                         "in each panel. Default off (boxplots only); pass "
                         "--trend-line to re-enable.")
    ap.add_argument("--max-ipi-min", type=float, default=240.0,
                    help="Physiological IPI cutoff in minutes; intervals above "
                         "this are dropped as unresolved merges (CLAUDE.md step 5; "
                         "matches build_figure2_peak_stats.py). Applied to both data "
                         "and model. Pass a non-positive value to disable.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_figure3_boxplots")
    args = ap.parse_args()

    data_peaks = _load_data_peaks(args.peaks_csv)
    print(f"[data] {len(data_peaks)} peaks across {data_peaks['series_uid'].nunique()} subjects")

    model_peaks = _simulate_model_peaks(
        args.fit_dir,
        n_subjects=args.n_subjects, n_reps=args.n_reps,
        base_seed=args.seed,
        prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min,
        resample_dt_min=args.resample_dt_min,
    )
    print(f"[model] {len(model_peaks)} peaks across {model_peaks['series_uid'].nunique()} subj×reps")

    # Physiological IPI cutoff (CLAUDE.md step 5): NaN-out intervals above the
    # cutoff so the IPI panels drop them while amplitude panels stay intact.
    max_ipi_min = args.max_ipi_min if args.max_ipi_min and args.max_ipi_min > 0 else None
    if max_ipi_min is not None:
        for tag, dfp in (("data", data_peaks), ("model", model_peaks)):
            ipi = dfp["ipi"].to_numpy(float)
            finite = np.isfinite(ipi)
            n_total = int(finite.sum())
            over = finite & (ipi > max_ipi_min)
            n_dropped = int(over.sum())
            dfp.loc[over, "ipi"] = np.nan
            frac = (n_dropped / n_total) if n_total else 0.0
            print(f"[{tag}] IPI cutoff = {max_ipi_min:g} min: dropped "
                  f"{n_dropped}/{n_total} ({frac:.1%}) as unresolved merges.")

    fig_dir = args.out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = args.out / "artifacts"; art_dir.mkdir(parents=True, exist_ok=True)
    data_peaks.to_csv(art_dir / "data_peaks.csv", index=False)
    model_peaks.to_csv(art_dir / "model_peaks.csv", index=False)

    setup_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.4))
    # Layout matches Figure 2: top row = means, bottom row = CVs (shared ylim).
    panels = [
        ("amp", "mean", "Amplitude mean (Z-score)", "A  Amplitude mean", axes[0, 0], 1),
        ("ipi", "mean", "Inter-peak interval (IPI) mean (min)",
                         "B  Inter-peak interval (IPI) mean", axes[0, 1], 1),
        ("amp", "cv",   "Amplitude CV",             "C  Amplitude CV",   axes[1, 0], 2),
        ("ipi", "cv",   "Inter-peak interval (IPI) CV",
                         "D  Inter-peak interval (IPI) CV",   axes[1, 1], 2),
    ]
    rayleigh_cv = float(np.sqrt((4.0 - np.pi) / np.pi))

    # Pre-compute long tables so the two CV panels can share a y-limit.
    longs = {}
    for value_col, stat, ylabel, title, ax, mp in panels:
        longs[(value_col, stat)] = (
            _per_uid_bin_stat(data_peaks,  value_col, stat, mp),
            _per_uid_bin_stat(model_peaks, value_col, stat, mp),
        )
    cv_all = np.concatenate([
        np.concatenate([d["value"].to_numpy(float), m["value"].to_numpy(float)])
        for (vc, st), (d, m) in longs.items() if st == "cv"])
    cv_all = cv_all[np.isfinite(cv_all)]
    cv_ylim = (0.0, float(np.nanquantile(cv_all, 0.97)) * 1.15) if len(cv_all) else None

    for value_col, stat, ylabel, title, ax, mp in panels:
        d_long, m_long = longs[(value_col, stat)]
        _box_pair(ax, d_long, m_long, ylabel=ylabel, title=title,
                  connect_trend=args.trend_line,
                  ylim=cv_ylim if stat == "cv" else None)
        if value_col == "amp" and stat == "cv":
            ax.axhline(rayleigh_cv, color="#c0392b", linestyle="--",
                       linewidth=1.0, zorder=4,
                       label=f"Theoretical CV = {rayleigh_cv:.3f}")
            ax.legend(loc="upper right", fontsize=10, frameon=False)

    data_h = plt.Rectangle((0, 0), 1, 1, fc=DATA_COLOR, alpha=0.55, ec=DATA_COLOR)
    model_h = plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLOR, alpha=0.55, ec=MODEL_COLOR)
    n_sims = args.n_subjects * args.n_reps
    model_label = f"Model (n={n_sims} simulations)"
    fig.legend(
        [data_h, model_h],
        [f"Data (n={data_peaks['series_uid'].nunique()} subjects)",
         model_label],
        loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=2, frameon=False, fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_png = fig_dir / "figure_3.png"
    out_pdf = fig_dir / "figure_3.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"PNG: {out_png}")
    print(f"PDF: {out_pdf}")


if __name__ == "__main__":
    main()
