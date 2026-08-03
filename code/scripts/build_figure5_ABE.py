"""Figure 5, three-panel layout: A (dynamics), B (raster), E (resonance).

Restructured from the five-panel build_figure5_entrainment.py: drops the phase
histogram (old C) and the phase-locking-vs-period curve (old D); keeps A, B, and
the resonance panel E, with E moved into C's slot (so the figure is A | B E).

Unlike the original D/E (probed with a weak sine in a separate feedback-engaged
regime), E here is the resonance of the SAME cue + model + noise used in A/B:
the additive (or multiplicative) pulse cue is swept over periods and the mean
peak-to-trough cortisol amplitude is measured. The hump marks the oscillator's
natural ultradian period.

Usage:
  PYTHONPATH=src python scripts/build_figure5_ABE.py \
      --fit-dir experiments/runs/fit_drive_noise_nocirc_baseline1 \
      --cue-mode additive --stim-amplitude 15 --stim-period-min 90 \
      --out experiments/runs/fig5_ABE
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# reuse the five-panel builder's drive + sim helpers
_spec = importlib.util.spec_from_file_location(
    "b5", PROJECT_ROOT / "scripts/build_figure5_entrainment.py")
b5 = importlib.util.module_from_spec(_spec)
sys.modules["b5"] = b5
_spec.loader.exec_module(b5)

from hpa_model.plotting import setup_nature_style  # noqa: E402

COL_NOISE = b5.COL_NOISE
COL_ENTRAIN = b5.COL_ENTRAIN
COL_STIM = b5.COL_STIM


def _per_rep_peak_to_trough(trajs, *, dt_min, min_dist_min):
    """Mean peak-to-trough cortisol amplitude per realization."""
    dist = max(1, int(round(min_dist_min / dt_min)))
    amps = []
    for _t, x in trajs:
        pkk, _ = find_peaks(x, distance=dist)
        trr, _ = find_peaks(-x, distance=dist)
        if len(pkk) and len(trr):
            amps.append(float(x[pkk].mean() - x[trr].mean()))
    return np.asarray(amps, dtype=float)


def _resonance_sweep(model, common, *, periods, amp, sim_kw):
    """Sweep the cue period; return DataFrame with peak-to-trough amplitude & R."""
    rows = []
    dist = max(1, int(round(sim_kw["min_dist_min"] / sim_kw["dt_min"])))
    for i, per in enumerate(periods):
        c = dict(common); c["stim_period_min"] = float(per)
        kw = dict(sim_kw); kw["base_seed"] = sim_kw["base_seed"] + 1000 + i * sim_kw["n_reps"]
        drive = b5.CircadianPulseDrive(stim_amplitude=float(amp), **c)
        trajs, pk = b5._simulate(drive, model, **kw)
        p2t = []
        for _t, x in trajs:
            pkk, _ = find_peaks(x, distance=dist)
            trr, _ = find_peaks(-x, distance=dist)
            if len(pkk) and len(trr):
                p2t.append(float(x[pkk].mean() - x[trr].mean()))
        R = b5._rayleigh_R(pk[0], float(per))
        rows.append(dict(period_min=per, period_hr=per / 60.0,
                         peak_to_trough=float(np.mean(p2t)) if p2t else float("nan"),
                         rayleigh_R=R))
        print(f"   period={per:6.0f} min  peak-to-trough={rows[-1]['peak_to_trough']:.3f}  R={R:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/fit_drive_noise_nocirc_baseline1")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/runs/fig5_ABE")
    ap.add_argument("--cue-mode", choices=["multiplicative", "additive"], default="additive")
    ap.add_argument("--stim-amplitude", type=float, default=15.0)
    ap.add_argument("--stim-period-min", type=float, default=90.0)
    ap.add_argument("--pulse-width-min", type=float, default=10.0)
    ap.add_argument("--dt-min", type=float, default=1.0)
    ap.add_argument("--warmup-min", type=float, default=1440.0)
    ap.add_argument("--duration-min", type=float, default=1440.0)
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--sweep-n-reps", type=int, default=40)
    ap.add_argument("--sweep-periods-min", nargs="+", type=float,
                    default=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 135, 150, 180, 220, 270])
    ap.add_argument("--prom-factor", type=float, default=0.1)
    ap.add_argument("--min-distance-min", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_nature_style()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "artifacts").mkdir(parents=True, exist_ok=True)

    model, circ, drive_eps, noise_form = b5._load_model_and_params(args.fit_dir)
    nat_period = b5._natural_period_min(model, level=circ["baseline"])
    print(f"[model] baseline={circ['baseline']} eps={drive_eps:.3f} {noise_form} "
          f"cue={args.cue_mode} natural_period={nat_period:.1f}min")

    common = dict(stim_period_min=args.stim_period_min, pulse_width_min=args.pulse_width_min,
                  drive_epsilon=drive_eps, drive_noise_form=noise_form,
                  cue_mode=args.cue_mode, **circ)
    sim_kw = dict(dt_min=args.dt_min, warmup_min=args.warmup_min,
                  duration_min=args.duration_min, n_reps=args.n_reps,
                  base_seed=args.seed, prom_factor=args.prom_factor,
                  min_dist_min=args.min_distance_min)

    print(f"[sim] noise-only + entrained ({args.n_reps} reps each)...")
    noise_trajs, noise_pk = b5._simulate(b5.CircadianPulseDrive(stim_amplitude=0.0, **common),
                                         model, **sim_kw)
    entr_trajs, entr_pk = b5._simulate(b5.CircadianPulseDrive(stim_amplitude=args.stim_amplitude, **common),
                                       model, **sim_kw)
    R_entr = b5._rayleigh_R(entr_pk[0], args.stim_period_min)
    print(f"[lock] entrained R={R_entr:.3f}")

    print(f"[resonance] sweeping cue period ({args.sweep_n_reps} reps each)...")
    sweep_kw = dict(sim_kw); sweep_kw["n_reps"] = args.sweep_n_reps
    sweep_df = _resonance_sweep(model, common, periods=args.sweep_periods_min,
                                amp=args.stim_amplitude, sim_kw=sweep_kw)
    sweep_df.to_csv(args.out / "artifacts" / "resonance_sweep.csv", index=False)

    _assemble(args, model, circ, nat_period, noise_trajs, entr_trajs, entr_pk, R_entr, sweep_df)
    print(f"[done] {args.out}")


def _assemble(args, model, circ, nat_period, noise_trajs, entr_trajs, entr_pk, R_entr, sweep_df):
    t_max_hr = args.duration_min / 60.0
    fig = plt.figure(figsize=(12.0, 6.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.5, 1.15],
                           hspace=0.55, wspace=0.36,
                           left=0.07, right=0.97, top=0.93, bottom=0.11)

    # Panel A: dynamics (noise-only top, entrained bottom) ------------------------
    gsA = gs[0, :].subgridspec(2, 1, hspace=0.5)
    axA1 = fig.add_subplot(gsA[0]); axA2 = fig.add_subplot(gsA[1])
    t0, x0 = noise_trajs[0]
    axA1.plot(t0 / 60.0, x0, color=COL_NOISE, lw=1.1)
    axA1.set_title("Noise only (no stimulus)", color=COL_NOISE, fontsize=13)
    axA1.set_ylabel("Cortisol (a.u.)", fontsize=12)
    axA1.tick_params(labelbottom=False)
    t1, x1 = entr_trajs[0]
    b5._draw_pulses(axA2, args.stim_period_min, args.pulse_width_min, t_max_hr,
                    COL_STIM, alpha=0.30, label="stimulus pulse")
    axA2.plot(t1 / 60.0, x1, color=COL_ENTRAIN, lw=1.1)
    axA2.set_title(f"+ pulsatile stimulus every {args.stim_period_min:.0f} min",
                   color=COL_ENTRAIN, fontsize=13)
    axA2.set_ylabel("Cortisol (a.u.)", fontsize=12)
    axA2.set_xlabel("Time (hours)", fontsize=12)
    axA2.legend(loc="upper right", frameon=False, fontsize=10)
    for ax in (axA1, axA2):
        ax.set_xlim(0, t_max_hr)
    axA1.text(-0.06, 1.08, "A", transform=axA1.transAxes, fontweight="bold", fontsize=17)

    # Panel B: peak raster -------------------------------------------------------
    axB = fig.add_subplot(gs[1, 0])
    n_reps = args.n_reps
    b5._draw_pulses(axB, args.stim_period_min, args.pulse_width_min, t_max_hr, COL_STIM, alpha=0.22)
    et, _, erep = entr_pk
    axB.scatter(et / 60.0, erep, s=4, color=COL_ENTRAIN, alpha=0.8, lw=0)
    axB.set_xlim(0, t_max_hr)
    axB.set_ylim(-1, n_reps + 1)
    axB.set_yticks([0, n_reps]); axB.set_ylabel("realization", fontsize=12)
    axB.set_xlabel("Time (hours)", fontsize=12)
    axB.set_title(f"Peak raster (entrained, R={R_entr:.2f})", fontsize=13)
    axB.text(-0.18, 1.06, "B", transform=axB.transAxes, fontweight="bold", fontsize=17)

    # Panel C: resonance (peak-to-trough vs cue period) --------------------------
    axE = fig.add_subplot(gs[1, 1])
    x = sweep_df["period_hr"].to_numpy()
    y = sweep_df["peak_to_trough"].to_numpy()
    peak_idx = int(np.nanargmax(y))
    res_hr = float(x[peak_idx]); res_min = float(sweep_df["period_min"].to_numpy()[peak_idx])
    axE.plot(x, y, "o-", color=COL_ENTRAIN, lw=1.6, ms=4)
    axE.fill_between(x, 0, y, color=COL_ENTRAIN, alpha=0.12)
    axE.set_xlabel("Stimulus period (hours)", fontsize=12)
    axE.set_ylabel("Peak-to-trough\namplitude (a.u.)", fontsize=12)
    axE.set_ylim(bottom=0)
    axE.set_title("Resonance: response peaks\nat the natural period", fontsize=13)
    axE.text(-0.24, 1.06, "C", transform=axE.transAxes, fontweight="bold", fontsize=17)

    # Panel D: cortisol pulse amplitude, noise-only vs entrained ------------------
    axD = fig.add_subplot(gs[1, 2])
    amp_noise = _per_rep_peak_to_trough(noise_trajs, dt_min=args.dt_min,
                                        min_dist_min=args.min_distance_min)
    amp_entr = _per_rep_peak_to_trough(entr_trajs, dt_min=args.dt_min,
                                       min_dist_min=args.min_distance_min)
    groups = [amp_noise, amp_entr]
    colors = [COL_NOISE, COL_ENTRAIN]
    labels = ["Noise\nonly", "+ pulsatile\nstimulus"]
    vdata = [g for g in groups if len(g) > 1]
    vpos = [pos for pos, g in zip([0, 1], groups) if len(g) > 1]
    vcols = [c for c, g in zip(colors, groups) if len(g) > 1]
    if vdata:
        vp = axD.violinplot(vdata, positions=vpos, widths=0.7,
                            showmeans=False, showextrema=False, showmedians=True)
        for body, c in zip(vp["bodies"], vcols):
            body.set_facecolor(c); body.set_alpha(0.45); body.set_edgecolor(c)
            body.set_linewidth(1.0)
        if "cmedians" in vp:
            vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(1.6)
    rng = np.random.default_rng(7)
    for pos, g, c in zip([0, 1], groups, colors):
        if len(g):
            axD.scatter(pos + rng.uniform(-0.13, 0.13, len(g)), g,
                        s=12, color=c, alpha=0.6, edgecolors="none", zorder=3)
    axD.set_xticks([0, 1]); axD.set_xticklabels(labels, fontsize=11)
    axD.set_ylabel("Peak-to-trough\namplitude (a.u.)", fontsize=12)
    _dmax = max((float(np.max(g)) for g in groups if len(g)), default=1.0)
    axD.set_ylim(0, _dmax * 1.30)  # headroom for violin KDE tails + the % / p label
    axD.set_title("Cortisol amplitude:\nnoise-only vs entrained", fontsize=13)
    axD.text(-0.26, 1.06, "D", transform=axD.transAxes, fontweight="bold", fontsize=17)
    if len(amp_noise) and len(amp_entr):
        try:
            _, p = mannwhitneyu(amp_noise, amp_entr, alternative="two-sided")
        except ValueError:
            p = float("nan")
        pct = 100.0 * (np.median(amp_entr) - np.median(amp_noise)) / np.median(amp_noise)
        if p < 0.001:
            p_str = "p < 0.001"
        elif p < 0.01:
            p_str = f"p = {p:.1e}"
        else:
            p_str = f"p = {p:.3f}"
        axD.text(0.5, 0.97, f"{pct:+.0f}%  ({p_str})", transform=axD.transAxes,
                 ha="center", va="top", fontsize=10, color="0.25")

    for ax in (axB, axE, axD):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax in (axA1, axA2):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    pd.DataFrame({
        "condition": (["noise_only"] * len(amp_noise) + ["entrained"] * len(amp_entr)),
        "peak_to_trough": np.concatenate([amp_noise, amp_entr]),
    }).to_csv(args.out / "artifacts" / "amplitude_compare.csv", index=False)

    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"figure_5_ABE.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
