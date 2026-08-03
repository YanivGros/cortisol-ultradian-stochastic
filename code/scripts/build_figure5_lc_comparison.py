"""Fig 5 — limit cycle (with/without noise) vs the noise-driven damped spiral.

Three conditions, all under the SAME fitted two-harmonic circadian drive and the
identical peak-extraction pipeline (z-scored residual, prominence 0.5σ, 60-min min
separation, 240-min IPI cutoff):

  1. **Walker (2010) delay limit cycle, deterministic** (ε=0) — metronomic timing.
  2. **Walker (2010) delay limit cycle + lognormal drive noise** at ε=ε* (params
     re-fit with noise active so it still matches the data's mean amplitude + IPI).
     Larger noise than the spiral is needed to reach the data's timing variability.
  3. **Noise-driven damped spiral** — Karin (2020) HPA model, k_GR=5, τ=0, ε=1.5.

Data (pooled cortisol peaks) is the dashed reference. The amplitude-multimodality
(Hartigan dip) argument is dropped; the figure focuses on IPI (timing): the
deterministic limit cycle's IPI CV collapses, while both noisy models reproduce it.

Reuses helpers from scripts/compare_limit_cycle_vs_noise.py (clc) and
scripts/noisy_limit_cycle_sweep.py (nls).

Usage:
  PYTHONPATH=src python scripts/build_figure5_lc_comparison.py \
      --fit-dir experiments/runs/eps15_acth20_cort15 \
      --peaks-csv experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv \
      --walker-noise-eps 3.0 \
      --out experiments/runs/limit_cycle_vs_noise_acth20_fig5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compare_limit_cycle_vs_noise as clc  # noqa: E402
import noisy_limit_cycle_sweep as nls  # noqa: E402
from hpa_model.plotting import apply_paper_style, setup_nature_style  # noqa: E402

# Colours: one per model condition, reused between the trace panels and the bars.
DATA_COLOR = clc.DATA_COLOR            # blue (reference)
DET_LC_COLOR = "#6A51A3"               # purple   — deterministic limit cycle
NOISY_LC_COLOR = "#2B8CBE"             # teal     — limit cycle + noise
SPIRAL_COLOR = clc.NOISE_COLOR         # orange   — noise-driven damped spiral
WALKER_BOUNDS = dict(p1=(2.0, 80.0), p2=(2.0, 150.0), p4=(0.005, 1.0),
                     p5=(0.0, 1.0), p6=(0.3, 10.0), tau_min=(5.0, 30.0))


def _align_window(t_min, x, acro_data):
    """Slice a seamless 24 h window whose circadian acrophase sits at acro_data,
    then z-score (display only; mirrors clc._trajectory_traces._align)."""
    t_min = np.asarray(t_min, float); x = np.asarray(x, float)
    dt = float(np.median(np.diff(t_min)))
    n_day = int(round(1440.0 / dt))
    acro = clc._circadian_acrophase_min(t_min, x)
    shift = float((acro - acro_data) % 1440.0)
    i0 = int(np.searchsorted(t_min, shift, side="left"))
    i0 = max(0, min(i0, len(t_min) - n_day))
    sl = slice(i0, i0 + n_day)
    tw = t_min[sl] - t_min[i0]; xw = x[sl]
    z = (xw - xw.mean()) / xw.std() if xw.std() > 1e-9 else np.zeros_like(xw)
    return tw / 60.0, z


def _spiral_peaks(cp, *, n_reps, dt_min, prom_sigma, min_distance_min, seed):
    """Pool peaks from n_reps noise-driven (kgr=5, τ=0, ε from cp) realizations."""
    model = clc._our_model(5.0, 0.0)
    drive = clc._two_harmonic(cp, with_noise=True)
    amps, ipis = [], []
    for rep in range(n_reps):
        t, x = clc._sim_window(model, drive, dt_min=dt_min, warmup_min=1440.0,
                               duration_min=1440.0, seed=seed + rep)
        pt, a = clc._prev_dip_amps_z(t, x, prom_sigma=prom_sigma,
                                     min_distance_min=min_distance_min)
        amps.extend(a.tolist())
        if pt.size >= 2:
            ipis.extend([v for v in np.diff(pt) if 0 < v <= 240.0])
    return np.asarray(amps, float), np.asarray(ipis, float)


def _walker_noisy_trace(cp, params, eps, *, dt_min, seed):
    drive_fn = clc._two_harmonic(cp, with_noise=False).base_value
    p3 = clc.p3_from_half_lives(clc.CORT_HALF_LIFE_MIN, clc.ACTH_HALF_LIFE_MIN)
    sim = clc.simulate_walker(
        p1=params["p1"], p2=params["p2"], p4=params["p4"], p5=params["p5"],
        p6=params["p6"], tau_min=params["tau_min"], p3=p3,
        cort_half_life_min=clc.CORT_HALF_LIFE_MIN, dt_min=dt_min,
        warmup_min=2880.0, duration_min=2880.0, drive_fn=drive_fn,
        epsilon=float(eps), noise_form=cp["noise_form"], seed=seed)
    return sim["time_min"], sim["o"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/eps15_acth20_cort15")
    ap.add_argument("--peaks-csv", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/limit_cycle_vs_noise_acth20_fig5")
    ap.add_argument("--walker-noise-eps", type=float, required=True,
                    help="ε* for the noisy Walker LC (chosen so its IPI CV ≈ data).")
    ap.add_argument("--n-reps", type=int, default=200,
                    help="Stochastic realizations for the noisy conditions.")
    ap.add_argument("--lc-n-days", type=int, default=60,
                    help="Days of deterministic Walker LC pooled for its distribution.")
    ap.add_argument("--walker-maxiter", type=int, default=40)
    ap.add_argument("--walker-noise-maxiter", type=int, default=12)
    ap.add_argument("--walker-noise-fitreps", type=int, default=16)
    ap.add_argument("--dt-min", type=float, default=1.0)
    ap.add_argument("--walker-dt-min", type=float, default=0.25)
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--variant", type=str, default="shifted_12h")
    ap.add_argument("--data-example-dataset", type=str, default="all_digitized")
    ap.add_argument("--data-example-id", type=str, default="6")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noisy-traj-seed", type=int, default=17,
                    help="Seed for the noisy-LC example trace (panel B); chosen to "
                         "show a representative pulsatile realization.")
    ap.add_argument("--refit", action="store_true", help="Ignore cached DE fits.")
    args = ap.parse_args()

    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "artifacts").mkdir(parents=True, exist_ok=True)
    eps_star = float(args.walker_noise_eps)

    cp = clc._circadian_params(args.fit_dir)
    eps_spiral = float(cp["epsilon"])
    amp_data, amp_mean, ipi_data = clc._data_targets(args.peaks_csv)
    ipi_mean = float(np.mean(ipi_data))
    print(f"[data] amp_mean={amp_mean:.3f} ipi_mean={ipi_mean:.1f} "
          f"ampCV={clc._cv(amp_data):.3f} ipiCV={clc._ipi_cv(ipi_data):.3f}")
    print(f"[circadian] a24={cp['a24']:.3f} a12={cp['a12']:.3f} ε_spiral={eps_spiral:.2f} "
          f"ε*_walker={eps_star:.2f}")

    # ── fits (cached) ────────────────────────────────────────────────────────────
    fit_cache = args.out / "artifacts" / "fit_cache.json"
    cached = {}
    if fit_cache.exists() and not args.refit:
        try:
            cached = json.loads(fit_cache.read_text())
        except Exception:  # noqa: BLE001
            cached = {}

    if cached.get("walker_det"):
        w_det = cached["walker_det"]
        print(f"[walker-det] cached τ={w_det['tau_min']:.1f}min")
    else:
        print("[walker-det] fitting deterministic Walker LC...")
        w_det = clc._fit_walker(cp, data_amp=amp_mean, data_ipi=ipi_mean,
                                dt_min=args.walker_dt_min, prom_sigma=args.prom_sigma,
                                min_distance_min=args.min_distance_min,
                                bounds=WALKER_BOUNDS, maxiter=args.walker_maxiter)

    if cached.get("walker_noisy") and abs(cached.get("walker_noisy_eps", -1) - eps_star) < 1e-9:
        w_noisy = cached["walker_noisy"]
        print(f"[walker-noisy] cached τ={w_noisy['tau_min']:.1f}min (ε={eps_star})")
    else:
        print(f"[walker-noisy] re-fitting Walker LC with noise ε={eps_star}...")
        w_noisy = nls._refit_walker_noisy(
            cp, eps_star, data_amp=amp_mean, data_ipi=ipi_mean, bounds=WALKER_BOUNDS,
            dt_min=args.walker_dt_min, warmup_min=2880.0,
            fit_reps=args.walker_noise_fitreps, prom_sigma=args.prom_sigma,
            min_distance_min=args.min_distance_min, maxiter=args.walker_noise_maxiter,
            seed0=7)
    fit_cache.write_text(json.dumps(
        {"walker_det": w_det, "walker_noisy": w_noisy, "walker_noisy_eps": eps_star},
        indent=2))

    # ── peaks for the three conditions ───────────────────────────────────────────
    amp_det, ipi_det, _ = clc._walker_peaks(
        cp, w_det, dt_min=args.walker_dt_min, n_days=args.lc_n_days,
        prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min)
    amp_nz, ipi_nz = nls._noisy_walker_peaks(
        cp, w_noisy, eps_star, reps=args.n_reps, dt_min=args.walker_dt_min,
        warmup_min=2880.0, prom_sigma=args.prom_sigma,
        min_distance_min=args.min_distance_min, seed0=500)
    amp_sp, ipi_sp = _spiral_peaks(
        cp, n_reps=args.n_reps, dt_min=args.dt_min, prom_sigma=args.prom_sigma,
        min_distance_min=args.min_distance_min, seed=args.seed)

    conds = [
        dict(key="det_lc", color=DET_LC_COLOR,
             label=f"Limit cycle\n(deterministic, ε=0)", amp=amp_det, ipi=ipi_det),
        dict(key="noisy_lc", color=NOISY_LC_COLOR,
             label=f"Limit cycle\n+ noise (ε={eps_star:.1f})", amp=amp_nz, ipi=ipi_nz),
        dict(key="spiral", color=SPIRAL_COLOR,
             label=f"Damped spiral\n+ noise (ε={eps_spiral:.1f})", amp=amp_sp, ipi=ipi_sp),
    ]
    for c in conds:
        print(f"[{c['key']:>9}] n={c['amp'].size:5d} ampMean={np.mean(c['amp']):.3f} "
              f"ampCV={clc._cv(c['amp']):.3f} ipiMean={np.mean(c['ipi'][np.isfinite(c['ipi'])]):.1f} "
              f"ipiCV={clc._ipi_cv(c['ipi']):.3f}")

    # ── figure: row 1 = 3 example traces; row 2 = amplitude / IPI / CV ───────────
    setup_nature_style()
    td, rd, acro_data = clc._data_example_trace(
        args.variant, args.data_example_id, dataset=args.data_example_dataset)
    det_t, det_x = clc._walker_window(cp, w_det, dt_min=args.walker_dt_min, n_days=2)
    nz_t, nz_x = _walker_noisy_trace(cp, w_noisy, eps_star, dt_min=args.walker_dt_min,
                                     seed=args.noisy_traj_seed)
    sp_t, sp_x = clc._sim_window(clc._our_model(5.0, 0.0),
                                 clc._two_harmonic(cp, with_noise=True),
                                 dt_min=args.dt_min, warmup_min=1440.0,
                                 duration_min=2880.0, seed=args.seed)
    traces = [
        (conds[0], *_align_window(det_t, det_x, acro_data)),
        (conds[1], *_align_window(nz_t, nz_x, acro_data)),
        (conds[2], *_align_window(sp_t, sp_x, acro_data)),
    ]

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.95],
                           hspace=0.42, wspace=0.32,
                           left=0.07, right=0.985, top=0.92, bottom=0.10)

    # Row 1 — example z-scored cortisol over 24 h (one model per panel)
    for col, (c, th, z) in enumerate(traces):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(th, z, color=c["color"], lw=1.3)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6))
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("Time of day (h)", fontsize=10.5)
        if col == 0:
            ax.set_ylabel("Cortisol (z-score)", fontsize=10.5)
        ax.set_title(("ABC"[col]) + "  " + c["label"].replace("\n", " "),
                     fontsize=10.5, loc="left", color=c["color"])
        apply_paper_style(ax)

    # Row 2 — amplitude mean, IPI mean, CV (amp + IPI). Data is its own bin/bar.
    def boot(arr, fn, seed):
        arr = arr[np.isfinite(arr)]
        return clc._boot_ci(arr, fn, seed=seed)
    mean_fn = lambda v: float(np.mean(v))
    cv_amp = clc._cv
    cv_ipi = clc._ipi_cv
    # metric panels include the data as the first category ("bin").
    data_cond = dict(key="data", color=DATA_COLOR, label="Data",
                     amp=amp_data, ipi=ipi_data)
    mconds = [data_cond] + conds
    mlabels = ["data", "det LC", "noisy LC", "spiral"]
    xm = np.arange(len(mconds))

    # D: amplitude mean
    axD = fig.add_subplot(gs[1, 0])
    for j, c in enumerate(mconds):
        v, lo, hi = boot(c["amp"], mean_fn, j)
        axD.bar(j, v, width=0.7, color=c["color"], alpha=0.85, edgecolor="white", lw=0.6)
        axD.errorbar(j, v, yerr=[[max(v-lo, 0)], [max(hi-v, 0)]], fmt="none",
                     ecolor="0.25", elinewidth=1.0, capsize=3)
    axD.set_xticks(xm); axD.set_xticklabels(mlabels, fontsize=8.5)
    axD.set_ylabel("Amplitude mean (z)", fontsize=10.5)
    axD.set_title("D  Pulse amplitude", fontsize=10.5, loc="left")
    apply_paper_style(axD)

    # E: IPI mean
    axE = fig.add_subplot(gs[1, 1])
    for j, c in enumerate(mconds):
        ip = c["ipi"][np.isfinite(c["ipi"]) & (c["ipi"] > 0)]
        v, lo, hi = boot(ip, mean_fn, j)
        axE.bar(j, v, width=0.7, color=c["color"], alpha=0.85, edgecolor="white", lw=0.6)
        axE.errorbar(j, v, yerr=[[max(v-lo, 0)], [max(hi-v, 0)]], fmt="none",
                     ecolor="0.25", elinewidth=1.0, capsize=3)
    axE.set_xticks(xm); axE.set_xticklabels(mlabels, fontsize=8.5)
    axE.set_ylabel("IPI mean (min)", fontsize=10.5)
    axE.set_title("E  Inter-peak interval", fontsize=10.5, loc="left")
    apply_paper_style(axE)

    # F: CV — grouped amplitude-CV and IPI-CV, one bar per condition incl. data
    axF = fig.add_subplot(gs[1, 2])
    groups = [("amplitude CV", cv_amp, "amp"), ("IPI CV", cv_ipi, "ipi")]
    w = 0.19
    for gi, (glab, fn, field) in enumerate(groups):
        for j, c in enumerate(mconds):
            arr = c[field]
            arr = arr[np.isfinite(arr) & (arr > 0)] if field == "ipi" else arr
            axF.bar(gi + (j - 1.5) * w, fn(arr), width=w, color=c["color"],
                    alpha=0.85, edgecolor="white", lw=0.6)
    axF.set_xticks([0, 1]); axF.set_xticklabels(["amplitude CV", "IPI CV"], fontsize=9.5)
    axF.set_ylabel("coefficient of variation", fontsize=10.5)
    axF.set_title("F  Variability", fontsize=10.5, loc="left")
    axF.set_ylim(0, max(0.7, axF.get_ylim()[1]))
    from matplotlib.patches import Patch
    axF.legend(handles=[Patch(facecolor=c["color"], label=lab)
                        for c, lab in zip(mconds, mlabels)],
               frameon=False, fontsize=7.5, loc="upper left", ncol=2,
               handlelength=1.1, columnspacing=0.7, handletextpad=0.4)
    apply_paper_style(axF)

    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"limit_cycle_vs_noise_fig5.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── summary ──────────────────────────────────────────────────────────────────
    def stats(c):
        ip = c["ipi"][np.isfinite(c["ipi"]) & (c["ipi"] > 0)]
        return dict(n_peaks=int(c["amp"].size), amp_mean=float(np.mean(c["amp"])),
                    amp_cv=float(clc._cv(c["amp"])), ipi_mean=float(np.mean(ip)),
                    ipi_cv=float(clc._ipi_cv(c["ipi"])))
    summary = dict(
        circadian=cp, eps_spiral=eps_spiral, eps_star_walker=eps_star,
        walker_det={k: w_det[k] for k in clc.WALKER_KEYS} | {"loss": w_det.get("loss")},
        walker_noisy={k: w_noisy[k] for k in clc.WALKER_KEYS} | {"loss": w_noisy.get("loss")},
        data=dict(amp_mean=float(amp_mean), amp_cv=float(clc._cv(amp_data)),
                  ipi_mean=ipi_mean, ipi_cv=float(clc._ipi_cv(ipi_data))),
        conditions={c["key"]: stats(c) for c in conds})
    (args.out / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "manifest.json").write_text(json.dumps(
        {"task": "build_figure5_lc_comparison", "created_at": datetime.now(UTC).isoformat(),
         "fit_dir": str(args.fit_dir), "eps_spiral": eps_spiral,
         "eps_star_walker": eps_star}, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
