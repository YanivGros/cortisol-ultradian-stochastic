"""Figure 6 — cortisol pulse amplitude and inter-peak interval (IPI) vs kgr.

Sweeps the GR-feedback parameter kgr on the canonical noise-driven oscillator
(constant baseline drive u=1, lognormal drive noise eps from the fit; circadian
harmonics zeroed so the kgr effect is isolated) and, for each kgr, simulates many
noise replicates, detects cortisol peaks the same way as the manuscript pipeline
(z-scored residual, prominence 0.5 sigma, 60-min min distance), and reports the
mean peak-to-previous-trough amplitude (raw x3 units) and mean IPI (min), each
with a bootstrap 95% CI.

The fitted operating point (kgr=5) is marked.

Usage:
  PYTHONPATH=src python scripts/build_figure6_kgr_sweep.py \
      --fit-dir archive/experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v10_cort15_acth10 \
      --out experiments/runs/manuscript_figure6_kgr_sweep
"""
from __future__ import annotations

import argparse
import json
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

from hpa_model.model.three_state_gr_delay import ThreeStateGRDelayModel, TwoHarmonicNoiseDrive  # noqa: E402
from hpa_model.plotting import setup_nature_style  # noqa: E402
from hpa_model.simulate.engine import simulate_trajectory_fit_arrays  # noqa: E402

DATA_BLUE = "#2F5C85"
ACCENT_RED = "#C85C3A"


def _detect_peaks(times_min, x3, *, prom_sigma, min_distance_min):
    """Peak times, raw peak-to-prev-trough amplitudes (x3 units), via the pipeline rule."""
    resid = x3 - np.nanmean(x3)
    std = float(np.nanstd(resid))
    if std <= 0:
        return np.empty(0), np.empty(0)
    rz = (resid - np.nanmean(resid)) / std
    dt = float(np.median(np.diff(times_min)))
    dist = max(1, int(round(min_distance_min / dt)))
    peaks, _ = find_peaks(rz, distance=dist, prominence=prom_sigma)
    if peaks.size == 0:
        return np.empty(0), np.empty(0)
    amps = np.empty(peaks.size)
    for i, p in enumerate(peaks):
        lo = 0 if i == 0 else int(peaks[i - 1])
        trough = float(x3[lo:p].min()) if p > lo else float(x3[p])
        amps[i] = float(x3[p]) - trough
    return times_min[peaks], amps


def _boot_ci(vals, n_boot=2000, ci=0.95, seed=0):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, vals.size, size=(n_boot, vals.size))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(vals.mean()), float(lo), float(hi)


def sweep(fit_dir: Path, *, kgr_grid, n_reps, base_seed, prom_sigma,
          min_distance_min, resample_dt_min):
    cfg = yaml.safe_load((fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    mp = cfg["model"]["params"]
    dp = cfg["drive"]["params"]
    solver = cfg["solver"]
    runtime = cfg.get("runtime", {})
    target = np.arange(0.0, float(solver["duration_min"]) + 1e-9, float(resample_dt_min))

    rows = []
    for ki, kgr in enumerate(kgr_grid):
        model = ThreeStateGRDelayModel(
            a1=float(mp["a1"]), a2=float(mp["a2"]), a3=float(mp["a3"]),
            b1=float(mp["b1"]), b2=float(mp["b2"]), b3=float(mp["b3"]),
            kgr=float(kgr), tau_min=float(mp.get("tau_min", 0.0)),
            x3_floor=float(mp.get("x3_floor", 0.01)),
            hill_coeff=float(mp.get("hill_coeff", 3.0)),
            initial_state=tuple(float(x) for x in mp["initial_state"]),
        )
        # constant baseline drive (circadian harmonics zeroed), canonical lognormal noise
        drive = TwoHarmonicNoiseDrive(
            a24=0.0, phase24=0.0, a12=0.0, phase12=0.0,
            baseline=float(dp.get("baseline", 1.0)),
            epsilon=float(dp.get("epsilon", 0.0)),
            period_min=float(dp.get("period_min", 1440.0)),
            second_period_min=float(dp.get("second_period_min", 720.0)),
            noise_form=str(dp.get("noise_form", "lognormal")),
        )
        amps_all, ipis_all = [], []
        for rep in range(n_reps):
            # common random numbers: same noise realizations for every kgr (seed does
            # NOT depend on ki), so amplitude(kgr)/IPI(kgr) vary smoothly with kgr and
            # not with independent per-point sampling noise (variance reduction).
            sim = simulate_trajectory_fit_arrays(
                model, drive,
                dt_min=float(solver["dt_min"]),
                warmup_min=float(solver["warmup_min"]),
                duration_min=float(solver["duration_min"]),
                seed=base_seed + rep,
                noise_locations=list(runtime.get("noise_locations", []) or []),
                noise_epsilons=dict(runtime.get("noise_epsilons", {}) or {}),
                noise_form=str(runtime.get("noise_form", "multiplicative")),
            )
            x3 = np.interp(target, sim["time_min"], sim["x3"])
            pt, pa = _detect_peaks(target, x3, prom_sigma=prom_sigma,
                                   min_distance_min=min_distance_min)
            amps_all.extend(pa.tolist())
            if pt.size >= 2:
                ipis_all.extend(np.diff(np.sort(pt)).tolist())
        am, al, ah = _boot_ci(amps_all, seed=base_seed + ki)
        im, il, ih = _boot_ci(ipis_all, seed=base_seed + 7919 + ki)
        rows.append(dict(kgr=float(kgr), n_peaks=len(amps_all),
                         amp_mean=am, amp_lo=al, amp_hi=ah,
                         ipi_mean=im, ipi_lo=il, ipi_hi=ih))
        print(f"  kgr={kgr:5.2f}  peaks={len(amps_all):5d}  "
              f"amp={am:.3f} [{al:.3f},{ah:.3f}]  ipi={im:6.1f} [{il:.1f},{ih:.1f}]min")
    return pd.DataFrame(rows), float(mp["kgr"])


def _add_pct_change(df: pd.DataFrame, kgr_ref: float) -> pd.DataFrame:
    """Add %-change-from-kgr_ref columns (reference mean treated as fixed normalizer)."""
    ref = df.iloc[(df["kgr"] - kgr_ref).abs().idxmin()]
    df = df.copy()
    for m in ("amp", "ipi"):
        base = float(ref[f"{m}_mean"])
        for suf in ("mean", "lo", "hi"):
            df[f"{m}_{suf}_pct"] = 100.0 * (df[f"{m}_{suf}"] - base) / base
    return df


def build_figure(df: pd.DataFrame, kgr_fit: float, out_dir: Path):
    setup_nature_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    k = df["kgr"].to_numpy()

    axA.fill_between(k, df["amp_lo_pct"], df["amp_hi_pct"], color="#000000", alpha=0.15, lw=0)
    axA.plot(k, df["amp_mean_pct"], color="#000000", lw=2.2)
    axA.set_xlabel(r"$k_{GR}$", fontsize=13)
    axA.set_ylabel(r"Cortisol pulse amplitude" "\n" r"(% change from $k_{GR}{=}5$)",
                   fontsize=12.5)
    axA.set_title("Peak amplitude", fontsize=13)
    axA.text(-0.20, 1.04, "A", transform=axA.transAxes, fontsize=17, fontweight="bold")

    axB.fill_between(k, df["ipi_lo_pct"], df["ipi_hi_pct"], color="#000000", alpha=0.15, lw=0)
    axB.plot(k, df["ipi_mean_pct"], color="#000000", lw=2.2)
    axB.set_xlabel(r"$k_{GR}$", fontsize=13)
    axB.set_ylabel(r"Inter-peak interval" "\n" r"(% change from $k_{GR}{=}5$)",
                   fontsize=12.5)
    axB.set_title("Inter-peak interval", fontsize=13)
    axB.text(-0.20, 1.04, "B", transform=axB.transAxes, fontsize=17, fontweight="bold")

    # shared y-limits across both panels (both are % change, so directly comparable)
    lo = float(min(df["amp_lo_pct"].min(), df["ipi_lo_pct"].min()))
    hi = float(max(df["amp_hi_pct"].max(), df["ipi_hi_pct"].max()))
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    k_min, k_max = float(k.min()), float(k.max())
    for ax in (axA, axB):
        ax.set_ylim(ylim)
        ax.axhline(0.0, color="0.6", ls=":", lw=0.9)
        ax.axvline(kgr_fit, color="0.35", ls="--", lw=1.2)
        ax.text(kgr_fit, ylim[0], f" $k_{{GR}}$={kgr_fit:g}",
                color="0.35", fontsize=10, va="bottom", ha="left")
        ax.set_xlim(k_min, k_max)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    fdir = out_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"figure_6.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fdir / "figure_6.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "archive/experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v10_cort15_acth10")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/runs/manuscript_figure6_kgr_sweep")
    ap.add_argument("--kgr-min", type=float, default=1.0)
    ap.add_argument("--kgr-max", type=float, default=10.0)
    ap.add_argument("--n-points", type=int, default=19)
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--resample-dt-min", type=float, default=5.0)
    args = ap.parse_args()

    kgr_grid = np.linspace(args.kgr_min, args.kgr_max, args.n_points)
    print(f"[sweep] kgr {args.kgr_min}-{args.kgr_max} ({args.n_points} pts), "
          f"{args.n_reps} reps each, fit-dir={args.fit_dir.name}")
    df, kgr_fit = sweep(args.fit_dir, kgr_grid=kgr_grid, n_reps=args.n_reps,
                        base_seed=args.seed, prom_sigma=args.prom_sigma,
                        min_distance_min=args.min_distance_min,
                        resample_dt_min=args.resample_dt_min)
    df = _add_pct_change(df, kgr_fit)
    art = args.out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    df.to_csv(art / "kgr_sweep_metrics.csv", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "task": "build_figure6_kgr_sweep",
        "fit_dir": str(args.fit_dir),
        "drive": "constant baseline (circadian harmonics zeroed) + canonical lognormal noise",
        "kgr_grid": [args.kgr_min, args.kgr_max, args.n_points],
        "n_reps": args.n_reps, "prom_sigma": args.prom_sigma,
        "min_distance_min": args.min_distance_min, "fit_kgr": kgr_fit,
    }, indent=2))
    png = build_figure(df, kgr_fit, args.out)
    print(f"PNG: {png}")


if __name__ == "__main__":
    main()
