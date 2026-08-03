"""Companion to Figure 6: example cortisol trajectories at a few kgr values, with
the detected peaks and previous-troughs marked — i.e. exactly what the amplitude
(peak − prev trough) and IPI (peak-to-peak interval) measurements in figure_6 are
computed from. Single noise replicate per kgr, same model/noise/detection as the
sweep (constant baseline drive + lognormal noise, peaks on z-residual prom 0.5sigma,
60-min min distance).

Usage:
  PYTHONPATH=src python scripts/plot_figure6_example_trajectories.py \
      --fit-dir archive/experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v10_cort15_acth10 \
      --kgr 2 5 10 --out experiments/runs/manuscript_figure6_kgr_sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpa_model.model.three_state_gr_delay import ThreeStateGRDelayModel, TwoHarmonicNoiseDrive  # noqa: E402
from hpa_model.plotting import setup_nature_style  # noqa: E402
from hpa_model.simulate.engine import simulate_trajectory_fit_arrays  # noqa: E402

LINE = "#2F5C85"
PEAK = "#C85C3A"
TROUGH = "#3E8E5A"


def _detect(times_min, x3, *, prom_sigma, min_distance_min):
    """Return (peak_idx, prev_trough_idx) using the manuscript peak rule."""
    resid = x3 - np.nanmean(x3)
    std = float(np.nanstd(resid))
    if std <= 0:
        return np.empty(0, int), np.empty(0, int)
    rz = (resid - np.nanmean(resid)) / std
    dt = float(np.median(np.diff(times_min)))
    dist = max(1, int(round(min_distance_min / dt)))
    peaks, _ = find_peaks(rz, distance=dist, prominence=prom_sigma)
    troughs = np.empty(peaks.size, int)
    for i, p in enumerate(peaks):
        lo = 0 if i == 0 else int(peaks[i - 1])
        troughs[i] = (lo + int(np.argmin(x3[lo:p]))) if p > lo else p
    return peaks, troughs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "archive/experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v10_cort15_acth10")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/runs/manuscript_figure6_kgr_sweep")
    ap.add_argument("--kgr", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--resample-dt-min", type=float, default=5.0)
    args = ap.parse_args()

    cfg = yaml.safe_load((args.fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    mp, dp, solver = cfg["model"]["params"], cfg["drive"]["params"], cfg["solver"]
    runtime = cfg.get("runtime", {})
    target = np.arange(0.0, float(solver["duration_min"]) + 1e-9, float(args.resample_dt_min))

    setup_nature_style()
    n = len(args.kgr)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 1.7 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, kgr in zip(axes, args.kgr):
        model = ThreeStateGRDelayModel(
            a1=float(mp["a1"]), a2=float(mp["a2"]), a3=float(mp["a3"]),
            b1=float(mp["b1"]), b2=float(mp["b2"]), b3=float(mp["b3"]),
            kgr=float(kgr), tau_min=float(mp.get("tau_min", 0.0)),
            x3_floor=float(mp.get("x3_floor", 0.01)),
            hill_coeff=float(mp.get("hill_coeff", 3.0)),
            initial_state=tuple(float(x) for x in mp["initial_state"]),
        )
        drive = TwoHarmonicNoiseDrive(
            a24=0.0, phase24=0.0, a12=0.0, phase12=0.0,
            baseline=float(dp.get("baseline", 1.0)), epsilon=float(dp.get("epsilon", 0.0)),
            period_min=float(dp.get("period_min", 1440.0)),
            second_period_min=float(dp.get("second_period_min", 720.0)),
            noise_form=str(dp.get("noise_form", "lognormal")),
        )
        sim = simulate_trajectory_fit_arrays(
            model, drive, dt_min=float(solver["dt_min"]),
            warmup_min=float(solver["warmup_min"]), duration_min=float(solver["duration_min"]),
            seed=args.seed,
            noise_locations=list(runtime.get("noise_locations", []) or []),
            noise_epsilons=dict(runtime.get("noise_epsilons", {}) or {}),
            noise_form=str(runtime.get("noise_form", "multiplicative")),
        )
        x3 = np.interp(target, sim["time_min"], sim["x3"])
        hr = target / 60.0
        pk, tr = _detect(target, x3, prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min)

        ax.plot(hr, x3, color=LINE, lw=1.1)
        # amplitude segments: prev-trough -> peak
        for p, t in zip(pk, tr):
            ax.plot([hr[p], hr[p]], [x3[t], x3[p]], color="0.6", lw=0.8, zorder=1)
        ax.plot(hr[pk], x3[pk], "v", color=PEAK, ms=5, label="peak")
        ax.plot(hr[tr], x3[tr], "o", color=TROUGH, ms=3.5, mfc="none", label="prev trough")

        amp = float(np.mean(x3[pk] - x3[tr])) if pk.size else np.nan
        ipi = float(np.mean(np.diff(target[pk]))) if pk.size >= 2 else np.nan
        ax.set_ylabel("Cortisol (a.u.)")
        ax.set_title(rf"$k_{{GR}}={kgr:g}$   ·   {pk.size} peaks   ·   "
                     rf"mean amp={amp:.2f}   ·   mean IPI={ipi:.0f} min",
                     fontsize=8, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (hours)")
    axes[-1].set_xlim(0, target[-1] / 60.0)
    fig.suptitle("Figure 6 example trajectories: peak / prev-trough detection per $k_{GR}$ "
                 "(single noise replicate)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fdir = args.out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"figure_6_example_trajectories.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {fdir}/figure_6_example_trajectories.png")


if __name__ == "__main__":
    main()
