"""Symmetric-noise control for the Fig. 5 limit-cycle comparison.

The manuscript Fig. 5 contrasts the *noise-driven* spiral (kgr=5, lognormal drive
noise epsilon=1.5) against two *deterministic* delay limit cycles (our HPA model at
strong feedback kgr<1; the human-scaled Walker 2010 model). That comparison is
asymmetric: only the spiral carries noise. This script injects the **same**
lognormal drive noise into both limit cycles and **re-fits their parameters with
noise active** (so each limit cycle gets its best shot at matching the data's mean
amplitude + mean IPI), then sweeps the noise amplitude epsilon from 0 -> 1.5.

For each epsilon and model it reports the inter-peak-interval CV, amplitude CV,
Rayleigh KS p, and the Hartigan dip-test p (amplitude multimodality), all through
the identical peak-extraction pipeline used everywhere else. A reference curve for
the noise-driven spiral (kgr=5, tau=0; no refit) is included.

Reuses the canonical helpers from scripts/compare_limit_cycle_vs_noise.py.

Usage:
  PYTHONPATH=src python scripts/noisy_limit_cycle_sweep.py \
      --fit-dir experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v15_cort15_rayleighcv \
      --peaks-csv experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv \
      --out experiments/runs/noisy_limit_cycle_vs_noise_v15 \
      --eps-grid 0,0.5,1.0,1.5 --reps 100 --fit-reps 16
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
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compare_limit_cycle_vs_noise as clc  # noqa: E402
from hpa_model.model.three_state_gr_delay import TwoHarmonicNoiseDrive  # noqa: E402

DATA_COLOR = clc.DATA_COLOR
NOISE_COLOR = clc.NOISE_COLOR
OUR_LC_COLOR = clc.OUR_LC_COLOR
WALKER_COLOR = clc.WALKER_COLOR


# ── noisy peak generators (mirror the noise-driven block of the main script) ──────

def _our_noisy_drive(cp, eps):
    return TwoHarmonicNoiseDrive(
        a24=cp["a24"], phase24=cp["phase24"], a12=cp["a12"], phase12=cp["phase12"],
        baseline=cp["baseline"], epsilon=float(eps),
        period_min=cp["period_min"], second_period_min=cp["second_period_min"],
        noise_form=cp["noise_form"])


def _noisy_our_peaks(cp, kgr, tau, eps, *, reps, dt_min, warmup_min,
                     prom_sigma, min_distance_min, seed0):
    model = clc._our_model(kgr, tau)
    drive = _our_noisy_drive(cp, eps)
    amps, ipis = [], []
    for rep in range(reps):
        t, x = clc._sim_window(model, drive, dt_min=dt_min, warmup_min=warmup_min,
                               duration_min=1440.0, seed=seed0 + rep)
        pt, a = clc._prev_dip_amps_z(t, x, prom_sigma=prom_sigma,
                                     min_distance_min=min_distance_min)
        amps.extend(a.tolist())
        if pt.size >= 2:
            ipis.extend([v for v in np.diff(pt) if 0 < v <= 240.0])
    return np.asarray(amps, float), np.asarray(ipis, float)


def _noisy_walker_peaks(cp, params, eps, *, reps, dt_min, warmup_min,
                        prom_sigma, min_distance_min, seed0):
    drive = clc._two_harmonic(cp, with_noise=False)
    drive_fn = drive.base_value
    p3 = clc.p3_from_half_lives(clc.CORT_HALF_LIFE_MIN, clc.ACTH_HALF_LIFE_MIN)
    amps, ipis = [], []
    for rep in range(reps):
        sim = clc.simulate_walker(
            p1=params["p1"], p2=params["p2"], p4=params["p4"], p5=params["p5"],
            p6=params["p6"], tau_min=params["tau_min"], p3=p3,
            cort_half_life_min=clc.CORT_HALF_LIFE_MIN, dt_min=dt_min,
            warmup_min=warmup_min, duration_min=1440.0, drive_fn=drive_fn,
            epsilon=float(eps), noise_form=cp["noise_form"], seed=seed0 + rep)
        pt, a = clc._prev_dip_amps_z(sim["time_min"], sim["o"], prom_sigma=prom_sigma,
                                     min_distance_min=min_distance_min)
        amps.extend(a.tolist())
        if pt.size >= 2:
            ipis.extend([v for v in np.diff(pt) if 0 < v <= 240.0])
    return np.asarray(amps, float), np.asarray(ipis, float)


# ── refit-with-noise (common random numbers via a fixed seed0 across evals) ───────

def _refit_our_noisy(cp, eps, *, data_amp, data_ipi, kgr_bounds, tau_bounds,
                     dt_min, warmup_min, fit_reps, prom_sigma, min_distance_min,
                     maxiter, seed0):
    def loss(theta):
        amps, ipis = _noisy_our_peaks(cp, float(theta[0]), float(theta[1]), eps,
                                      reps=fit_reps, dt_min=dt_min, warmup_min=warmup_min,
                                      prom_sigma=prom_sigma, min_distance_min=min_distance_min,
                                      seed0=seed0)
        if amps.size < 4 or ipis.size < 2:
            return 1e3
        amp, ipi = float(np.mean(amps)), float(np.mean(ipis))
        return ((amp - data_amp) / data_amp) ** 2 + ((ipi - data_ipi) / data_ipi) ** 2
    res = differential_evolution(loss, bounds=[tuple(kgr_bounds), tuple(tau_bounds)],
                                 maxiter=maxiter, popsize=8, tol=1e-2, seed=0,
                                 polish=False, workers=1, init="sobol")
    return dict(kgr=float(res.x[0]), tau_min=float(res.x[1]), loss=float(res.fun))


def _refit_walker_noisy(cp, eps, *, data_amp, data_ipi, bounds, dt_min, warmup_min,
                        fit_reps, prom_sigma, min_distance_min, maxiter, seed0):
    def loss(theta):
        params = dict(zip(clc.WALKER_KEYS, (float(v) for v in theta)))
        amps, ipis = _noisy_walker_peaks(cp, params, eps, reps=fit_reps, dt_min=dt_min,
                                         warmup_min=warmup_min, prom_sigma=prom_sigma,
                                         min_distance_min=min_distance_min, seed0=seed0)
        if amps.size < 4 or ipis.size < 2:
            return 1e3
        amp, ipi = float(np.mean(amps)), float(np.mean(ipis))
        return ((amp - data_amp) / data_amp) ** 2 + ((ipi - data_ipi) / data_ipi) ** 2
    res = differential_evolution(loss, bounds=[tuple(bounds[k]) for k in clc.WALKER_KEYS],
                                 maxiter=maxiter, popsize=10, tol=1e-2, seed=0,
                                 polish=False, workers=1, init="sobol")
    params = dict(zip(clc.WALKER_KEYS, (float(v) for v in res.x)))
    params["loss"] = float(res.fun)
    return params


# ── stats ─────────────────────────────────────────────────────────────────────────

def _stats(amps, ipis, *, data_n, dip_boot):
    n_match = int(min(data_n, amps.size)) if amps.size else 0
    dip = clc._dip_matched(amps, n_match=max(5, n_match), n_boot=dip_boot) \
        if amps.size >= 5 else (float("nan"), float("nan"))
    return dict(
        n_peaks=int(amps.size),
        amp_mean=float(np.mean(amps)) if amps.size else float("nan"),
        amp_cv=clc._cv(amps),
        ipi_mean=float(np.mean(ipis)) if ipis.size else float("nan"),
        ipi_cv=clc._ipi_cv(ipis),
        rayleigh_ks_p=clc._rayleigh_ks_p(amps),
        dip_p=dip[0], dip_frac_multimodal=dip[1])


def _plot_sweep(df, *, data_ipi_cv, data_dip_p, out):
    """Two-panel SI figure: IPI CV and amplitude dip-test p vs noise amplitude."""
    clc.setup_nature_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    models = [("our_lc", OUR_LC_COLOR, "Karin model, limit cycle (refit w/ noise)"),
              ("walker", WALKER_COLOR, "Walker 2010 limit cycle (refit w/ noise)"),
              ("spiral_kgr5", NOISE_COLOR, "Noise-driven spiral ($k_{GR}$=5)")]
    handles = []
    for key, color, lab in models:
        sub = df[df["model"] == key].sort_values("eps")
        h, = axL.plot(sub["eps"], sub["ipi_cv"], "-o", color=color, lw=1.8, ms=5, label=lab)
        axR.plot(sub["eps"], sub["dip_p"], "-o", color=color, lw=1.8, ms=5, label=lab)
        handles.append(h)
    axL.axhspan(data_ipi_cv * 0.9, data_ipi_cv * 1.1, color=DATA_COLOR, alpha=0.18, zorder=0)
    hd = axL.axhline(data_ipi_cv, color=DATA_COLOR, ls="--", lw=1.3,
                     label=f"Data ({data_ipi_cv:.2f} / {data_dip_p:.2f})")
    axL.set_xlabel("Drive-noise amplitude $\\varepsilon$", fontsize=12)
    axL.set_ylabel("Inter-peak-interval CV", fontsize=12)
    axL.set_ylim(0, max(0.42, data_ipi_cv * 1.2))
    axL.set_title("A  Timing variability vs noise", fontsize=12, loc="left")
    axR.axhline(0.05, color="0.5", ls=":", lw=1.0)
    axR.axhline(data_dip_p, color=DATA_COLOR, ls="--", lw=1.3)
    axR.set_xlabel("Drive-noise amplitude $\\varepsilon$", fontsize=12)
    axR.set_ylabel("Hartigan dip-test $p$ (unimodal $\\rightarrow$ high)", fontsize=12)
    axR.set_ylim(-0.03, 1.05)
    axR.set_title("B  Amplitude multimodality vs noise", fontsize=12, loc="left")
    for ax in (axL, axR):
        clc.apply_paper_style(ax)
    fig.legend(handles=handles + [hd], frameon=False, fontsize=9.0, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"noisy_limit_cycle_sweep.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v15_cort15_rayleighcv")
    ap.add_argument("--peaks-csv", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/noisy_limit_cycle_vs_noise_v15")
    ap.add_argument("--eps-grid", type=str, default="0,0.5,1.0,1.5")
    ap.add_argument("--reps", type=int, default=100, help="realizations per eps for final stats")
    ap.add_argument("--fit-reps", type=int, default=16, help="realizations per DE objective eval")
    ap.add_argument("--our-maxiter", type=int, default=12)
    ap.add_argument("--walker-maxiter", type=int, default=10)
    ap.add_argument("--our-kgr-bounds", type=float, nargs=2, default=[0.3, 1.2],
                    help="constrained to the self-oscillating (limit-cycle) regime")
    ap.add_argument("--our-tau-bounds", type=float, nargs=2, default=[1.0, 50.0])
    ap.add_argument("--dt-min", type=float, default=1.0)
    ap.add_argument("--walker-dt-min", type=float, default=0.5)
    ap.add_argument("--warmup-min", type=float, default=1440.0)
    ap.add_argument("--dip-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the figure from the saved CSV/summary (no simulation)")
    args = ap.parse_args()

    (args.out / "artifacts").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    if args.replot:
        df = pd.read_csv(args.out / "artifacts" / "noisy_lc_sweep.csv")
        summ = json.loads((args.out / "artifacts" / "summary.json").read_text())
        _plot_sweep(df, data_ipi_cv=summ["data"]["ipi_cv"],
                    data_dip_p=summ["data"]["dip_p"], out=args.out)
        print(f"[replot] regenerated figures from {args.out / 'artifacts' / 'noisy_lc_sweep.csv'}")
        return

    cp = clc._circadian_params(args.fit_dir)
    eps_grid = [float(s) for s in str(args.eps_grid).split(",") if s.strip() != ""]

    amp_data, data_amp_mean, ipi_data = clc._data_targets(args.peaks_csv)
    data_ipi_mean = float(np.mean(ipi_data))
    data_n = int(amp_data.size)
    data_ipi_cv = clc._ipi_cv(ipi_data)
    data_dip = clc._dip_matched(amp_data, n_match=data_n, n_boot=args.dip_boot)
    print(f"[data] n={data_n}  amp mean={data_amp_mean:.3f}  IPI mean={data_ipi_mean:.1f}min  "
          f"IPI CV={data_ipi_cv:.3f}  dip p={data_dip[0]:.3g}")
    print(f"[circadian] a24={cp['a24']:.3f} a12={cp['a12']:.3f} baseline={cp['baseline']:.2f} "
          f"noise_form={cp['noise_form']}  manuscript eps={cp['epsilon']:.2f}")

    walker_bounds = dict(p1=(2.0, 80.0), p2=(2.0, 150.0), p4=(0.005, 1.0),
                         p5=(0.0, 1.0), p6=(0.3, 10.0), tau_min=(5.0, 30.0))

    rows = []
    raw_eps15 = {}
    for eps in eps_grid:
        print(f"\n===== epsilon = {eps:.2f} =====")
        # --- noise-driven spiral reference (kgr=5, tau=0; no refit) ---
        if eps == 0.0:
            amps_sp, ipis_sp, _ = clc._our_lc_peaks(cp, 5.0, 0.0, dt_min=args.dt_min,
                                                    n_days=60, prom_sigma=0.5, min_distance_min=60.0)
        else:
            amps_sp, ipis_sp = _noisy_our_peaks(cp, 5.0, 0.0, eps, reps=args.reps,
                                                dt_min=args.dt_min, warmup_min=args.warmup_min,
                                                prom_sigma=0.5, min_distance_min=60.0,
                                                seed0=args.seed)
        st_sp = _stats(amps_sp, ipis_sp, data_n=data_n, dip_boot=args.dip_boot)
        print(f"  [spiral kgr=5] n={st_sp['n_peaks']} ampCV={st_sp['amp_cv']:.3f} "
              f"IPI_CV={st_sp['ipi_cv']:.3f} dip_p={st_sp['dip_p']:.3g}")
        rows.append(dict(model="spiral_kgr5", eps=eps, refit=False, **st_sp))

        # --- our model, limit-cycle regime, refit WITH noise ---
        if eps == 0.0:
            of = clc._fit_our_lc(cp, data_amp=data_amp_mean, data_ipi=data_ipi_mean,
                                 dt_min=args.dt_min, prom_sigma=0.5, min_distance_min=60.0,
                                 kgr_bounds=args.our_kgr_bounds, tau_bounds=args.our_tau_bounds,
                                 maxiter=args.our_maxiter)
            amps_o, ipis_o, _ = clc._our_lc_peaks(cp, of["kgr"], of["tau_min"], dt_min=args.dt_min,
                                                  n_days=60, prom_sigma=0.5, min_distance_min=60.0)
        else:
            of = _refit_our_noisy(cp, eps, data_amp=data_amp_mean, data_ipi=data_ipi_mean,
                                  kgr_bounds=args.our_kgr_bounds, tau_bounds=args.our_tau_bounds,
                                  dt_min=args.dt_min, warmup_min=args.warmup_min,
                                  fit_reps=args.fit_reps, prom_sigma=0.5, min_distance_min=60.0,
                                  maxiter=args.our_maxiter, seed0=args.seed)
            amps_o, ipis_o = _noisy_our_peaks(cp, of["kgr"], of["tau_min"], eps, reps=args.reps,
                                              dt_min=args.dt_min, warmup_min=args.warmup_min,
                                              prom_sigma=0.5, min_distance_min=60.0, seed0=args.seed)
        st_o = _stats(amps_o, ipis_o, data_n=data_n, dip_boot=args.dip_boot)
        print(f"  [our-LC kgr={of['kgr']:.2f} tau={of['tau_min']:.0f}] n={st_o['n_peaks']} "
              f"ampCV={st_o['amp_cv']:.3f} IPI_CV={st_o['ipi_cv']:.3f} dip_p={st_o['dip_p']:.3g} "
              f"fitloss={of['loss']:.3f}")
        rows.append(dict(model="our_lc", eps=eps, refit=(eps > 0),
                         kgr=of["kgr"], tau_min=of["tau_min"], fit_loss=of["loss"], **st_o))

        # --- Walker limit cycle, refit WITH noise ---
        if eps == 0.0:
            wf = clc._fit_walker(cp, data_amp=data_amp_mean, data_ipi=data_ipi_mean,
                                 dt_min=args.walker_dt_min, prom_sigma=0.5, min_distance_min=60.0,
                                 bounds=walker_bounds, maxiter=args.walker_maxiter)
            amps_w, ipis_w, _ = clc._walker_peaks(cp, wf, dt_min=args.walker_dt_min, n_days=60,
                                                  prom_sigma=0.5, min_distance_min=60.0)
        else:
            wf = _refit_walker_noisy(cp, eps, data_amp=data_amp_mean, data_ipi=data_ipi_mean,
                                     bounds=walker_bounds, dt_min=args.walker_dt_min,
                                     warmup_min=args.warmup_min, fit_reps=args.fit_reps,
                                     prom_sigma=0.5, min_distance_min=60.0,
                                     maxiter=args.walker_maxiter, seed0=args.seed)
            amps_w, ipis_w = _noisy_walker_peaks(cp, wf, eps, reps=args.reps,
                                                 dt_min=args.walker_dt_min, warmup_min=args.warmup_min,
                                                 prom_sigma=0.5, min_distance_min=60.0, seed0=args.seed)
        st_w = _stats(amps_w, ipis_w, data_n=data_n, dip_boot=args.dip_boot)
        print(f"  [walker tau={wf['tau_min']:.0f}] n={st_w['n_peaks']} ampCV={st_w['amp_cv']:.3f} "
              f"IPI_CV={st_w['ipi_cv']:.3f} dip_p={st_w['dip_p']:.3g} fitloss={wf['loss']:.3f}")
        rows.append(dict(model="walker", eps=eps, refit=(eps > 0),
                         tau_min=wf["tau_min"], fit_loss=wf["loss"], **st_w))

        if abs(eps - cp["epsilon"]) < 1e-6 or abs(eps - 1.5) < 1e-6:
            raw_eps15 = dict(spiral=amps_sp.tolist(), our_lc=amps_o.tolist(),
                             walker=amps_w.tolist(), eps=eps)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "artifacts" / "noisy_lc_sweep.csv", index=False)
    summary = dict(
        circadian=cp, eps_grid=eps_grid, reps=args.reps, fit_reps=args.fit_reps,
        our_kgr_bounds=list(args.our_kgr_bounds),
        data=dict(n=data_n, amp_mean=data_amp_mean, ipi_mean=data_ipi_mean,
                  ipi_cv=data_ipi_cv, dip_p=data_dip[0]),
        rows=rows)
    (args.out / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_sweep(df, data_ipi_cv=data_ipi_cv, data_dip_p=data_dip[0], out=args.out)
    print(f"\n[done] wrote {args.out / 'artifacts' / 'noisy_lc_sweep.csv'} and figures/")


if __name__ == "__main__":
    main()
