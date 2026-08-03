"""Local sensitivity of cortisol IPI and pulse amplitude to a1,a2,a3,b1,b2,b3,kgr.

Around the fitted operating point, each parameter is swept independently over a
multiplicative grid (factor x fitted value; the a=b kinetic symmetry is broken on
purpose so clearance a_i and production b_i separate). For every factor we run
many noise replicates on the canonical noise-driven oscillator (constant baseline
drive u=1, lognormal drive noise eps from the fit, circadian harmonics zeroed),
detect cortisol peaks with the manuscript rule (z-scored residual, prominence
0.5 sigma, 60-min min distance), and record mean inter-peak interval (min) and
mean peak-to-previous-trough amplitude (raw x3 units), each with a bootstrap 95% CI.

Outputs:
  * overlay sweep curves: % change in IPI / amplitude vs parameter multiple
  * elasticity "tornado": local d ln(metric)/d ln(param) at the operating point,
    ranking each parameter's influence on IPI and on amplitude.

Usage:
  PYTHONPATH=src python scripts/build_sensitivity_ipi_amplitude.py \
      --fit-dir archive/experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v10_cort15_acth10 \
      --out experiments/runs/sensitivity_ipi_amplitude
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

PARAMS = ["a1", "a2", "a3", "b1", "b2", "b3", "kgr"]
PARAM_LABELS = {
    "a1": r"$a_1$ (CRH clearance)", "a2": r"$a_2$ (ACTH clearance)",
    "a3": r"$a_3$ (cortisol clearance)", "b1": r"$b_1$ (CRH production)",
    "b2": r"$b_2$ (ACTH production)", "b3": r"$b_3$ (cortisol production)",
    "kgr": r"$k_{GR}$ (GR feedback)",
}
PARAM_COLORS = {
    "a1": "#8C2D04", "a2": "#D94801", "a3": "#FD8D3C",
    "b1": "#08519C", "b2": "#3182BD", "b3": "#6BAED6",
    "kgr": "#54278F",
}


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


def _run_point(base, drive_kw, solver, runtime, *, n_reps, base_seed, prom_sigma,
               min_distance_min, target):
    """Mean IPI / amplitude (+CI) at one parameter point. CRN across all points."""
    model = ThreeStateGRDelayModel(
        a1=base["a1"], a2=base["a2"], a3=base["a3"],
        b1=base["b1"], b2=base["b2"], b3=base["b3"],
        kgr=base["kgr"], tau_min=base["tau_min"],
        x3_floor=base["x3_floor"], hill_coeff=base["hill_coeff"],
        initial_state=base["initial_state"],
    )
    drive = TwoHarmonicNoiseDrive(**drive_kw)
    amps_all, ipis_all = [], []
    for rep in range(n_reps):
        sim = simulate_trajectory_fit_arrays(
            model, drive,
            dt_min=float(solver["dt_min"]), warmup_min=float(solver["warmup_min"]),
            duration_min=float(solver["duration_min"]), seed=base_seed + rep,
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
    am, al, ah = _boot_ci(amps_all, seed=base_seed + 11)
    im, il, ih = _boot_ci(ipis_all, seed=base_seed + 7919)
    return dict(n_peaks=len(amps_all), amp_mean=am, amp_lo=al, amp_hi=ah,
                ipi_mean=im, ipi_lo=il, ipi_hi=ih)


def sweep(fit_dir, *, params, factors, n_reps, base_seed, prom_sigma,
          min_distance_min, resample_dt_min, abs_ranges=None, n_points=None):
    cfg = yaml.safe_load((fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    mp = cfg["model"]["params"]
    dp = cfg["drive"]["params"]
    solver, runtime = cfg["solver"], cfg.get("runtime", {})
    target = np.arange(0.0, float(solver["duration_min"]) + 1e-9, float(resample_dt_min))

    op = dict(  # operating point (fitted)
        a1=float(mp["a1"]), a2=float(mp["a2"]), a3=float(mp["a3"]),
        b1=float(mp["b1"]), b2=float(mp["b2"]), b3=float(mp["b3"]),
        kgr=float(mp["kgr"]), tau_min=float(mp.get("tau_min", 0.0)),
        x3_floor=float(mp.get("x3_floor", 0.01)), hill_coeff=float(mp.get("hill_coeff", 3.0)),
        initial_state=tuple(float(x) for x in mp["initial_state"]),
    )
    drive_kw = dict(
        a24=0.0, phase24=0.0, a12=0.0, phase12=0.0,
        baseline=float(dp.get("baseline", 1.0)), epsilon=float(dp.get("epsilon", 0.0)),
        period_min=float(dp.get("period_min", 1440.0)),
        second_period_min=float(dp.get("second_period_min", 720.0)),
        noise_form=str(dp.get("noise_form", "lognormal")),
    )
    common = dict(drive_kw=drive_kw, solver=solver, runtime=runtime, n_reps=n_reps,
                  base_seed=base_seed, prom_sigma=prom_sigma,
                  min_distance_min=min_distance_min, target=target)

    abs_ranges = abs_ranges or {}
    rows = []
    for p in params:
        if p in abs_ranges:
            # explicit absolute value grid; always include the fitted value op[p]
            lo, hi = abs_ranges[p]
            npts = n_points if n_points else len(factors)
            vals = np.unique(np.r_[np.linspace(lo, hi, npts), op[p]])
        else:
            vals = op[p] * np.asarray(factors, float)
        for v in vals:
            base = dict(op)
            base[p] = float(v)
            f = float(v) / op[p]
            res = _run_point(base, **common)
            rows.append(dict(param=p, factor=f, value=float(v), **res))
            print(f"  {p:4s} x{f:4.2f} (={v:6.3f})  peaks={res['n_peaks']:5d}  "
                  f"ipi={res['ipi_mean']:6.1f}  amp={res['amp_mean']:.3f}")
    return pd.DataFrame(rows), op


def _elasticity(sub, metric, window=(0.7, 1.43)):
    """Local d ln(metric)/d ln(factor) via OLS slope over the central factor window."""
    m = sub.sort_values("factor")
    sel = (m["factor"] >= window[0]) & (m["factor"] <= window[1])
    f = m.loc[sel, "factor"].to_numpy(float)
    y = m.loc[sel, f"{metric}_mean"].to_numpy(float)
    ok = np.isfinite(f) & np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(f[ok]), np.log(y[ok]), 1)[0])


def _pct(sub, metric):
    """% change of metric relative to factor==1 (nearest)."""
    m = sub.sort_values("factor").copy()
    i0 = (m["factor"] - 1.0).abs().idxmin()
    base = float(m.loc[i0, f"{metric}_mean"])
    out = {}
    for suf in ("mean", "lo", "hi"):
        out[suf] = 100.0 * (m[f"{metric}_{suf}"].to_numpy(float) - base) / base
    out["factor"] = m["factor"].to_numpy(float)
    return out


def build_curves_figure(df, params, out_dir):
    setup_nature_style()
    fig, (axI, axA) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for p in params:
        sub = df[df["param"] == p]
        ci = _pct(sub, "ipi"); ca = _pct(sub, "amp")
        c = PARAM_COLORS[p]
        axI.plot(ci["factor"], ci["mean"], color=c, lw=2.0, marker="o", ms=3, label=PARAM_LABELS[p])
        axA.plot(ca["factor"], ca["mean"], color=c, lw=2.0, marker="o", ms=3, label=PARAM_LABELS[p])
    for ax, ttl in ((axI, "Inter-peak interval"), (axA, "Cortisol pulse amplitude")):
        ax.axhline(0, color="0.6", ls=":", lw=0.9)
        ax.axvline(1.0, color="0.35", ls="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.set_xlabel("parameter / fitted value", fontsize=12)
        ax.set_ylabel(f"{ttl}\n(% change from fit)", fontsize=11.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axI.text(-0.16, 1.04, "A", transform=axI.transAxes, fontsize=16, fontweight="bold")
    axA.text(-0.16, 1.04, "B", transform=axA.transAxes, fontsize=16, fontweight="bold")
    axA.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Local sensitivity of cortisol IPI and amplitude to model parameters",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fdir = out_dir / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"sensitivity_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fdir / "sensitivity_curves.png"


def build_grid_figure(df, params, out_dir):
    """2x2 grid: row 0 = clearance (a1,a2,a3), row 1 = production (b1,b2,b3);
    col 0 = IPI %change, col 1 = amplitude %change. kgr is omitted (it is its own
    figure). The three kinetic tiers are overlaid per panel and distinguished by
    line style only (no colour), so the figure does not reuse the model/data
    colour code. y-axis shared within each column for direct comparison.
    """
    setup_nature_style()
    # columns = kinetic tier (CRH/ACTH/cortisol); rows = clearance / production.
    # Each panel overlays that parameter's IPI and amplitude response, distinguished
    # by line style only (no colour), so the model/data colour code is not reused.
    TIERS = [("1", "CRH"), ("2", "ACTH"), ("3", "Cortisol")]
    ROWS = [("a", "clearance"), ("b", "production")]
    SERIES = [("ipi", "Inter-peak interval", "-"), ("amp", "Pulse amplitude", "--")]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.4),
                             sharex=True, sharey=True, squeeze=False)
    for r, (pre, rword) in enumerate(ROWS):
        for c, (suf, tname) in enumerate(TIERS):
            ax = axes[r][c]
            sub = df[df["param"] == f"{pre}{suf}"]
            for metric, mlabel, ls in SERIES:
                if sub.empty:
                    continue
                pc = _pct(sub, metric)
                ax.fill_between(pc["factor"], pc["lo"], pc["hi"], color="0.55", alpha=0.10, lw=0)
                ax.plot(pc["factor"], pc["mean"], color="#000000", lw=1.8, ls=ls, label=mlabel)
            ax.axhline(0, color="0.6", ls=":", lw=0.9)
            ax.axvline(1.0, color="0.35", ls="--", lw=1.0)
            ax.set_xscale("log")
            ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0])
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            # each panel self-labelled as "<tier> <clearance|production>"
            # (e.g. "CRH clearance", "ACTH production")
            ax.set_title(f"{tname} {rword}", fontsize=11.5)
            if r == len(ROWS) - 1:
                ax.set_xlabel("fold change in parameter", fontsize=11)
        axes[r][0].set_ylabel("percent change", fontsize=11.5)
    axes[0][0].legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.tight_layout()
    fdir = out_dir / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"sensitivity_grid.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fdir / "sensitivity_grid.png"


def build_tornado_figure(elas_df, out_dir):
    setup_nature_style()
    # single consistent row order for both panels (by combined influence), so the
    # shared y-labels line up unambiguously with the bars in each panel.
    d = elas_df.copy()
    d["combined"] = d["elas_ipi"].abs() + d["elas_amp"].abs()
    d = d.sort_values("combined")  # smallest at bottom -> largest at top
    ypos = range(len(d))
    colors = [PARAM_COLORS[p] for p in d["param"]]
    fig, (axI, axA) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, metric, ttl in ((axI, "ipi", "IPI"), (axA, "amp", "amplitude")):
        ax.barh(list(ypos), d[f"elas_{metric}"], color=colors)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([PARAM_LABELS[p] for p in d["param"]], fontsize=9)
        ax.axvline(0, color="0.4", lw=0.9)
        ax.set_xlabel(f"elasticity  d ln({ttl})/d ln(param)", fontsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Parameter influence (local elasticity at the fitted operating point)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fdir = out_dir / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fdir / f"sensitivity_tornado.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fdir / "sensitivity_tornado.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "archive/experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v10_cort15_acth10")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/runs/sensitivity_ipi_amplitude")
    ap.add_argument("--params", type=str, default=",".join(PARAMS))
    ap.add_argument("--factor-min", type=float, default=0.5)
    ap.add_argument("--factor-max", type=float, default=2.0)
    ap.add_argument("--n-points", type=int, default=13)
    ap.add_argument("--n-reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--resample-dt-min", type=float, default=5.0)
    ap.add_argument("--kgr-min", type=float, default=2.0,
                    help="absolute lower bound for the kgr sweep (overrides factor grid for kgr)")
    ap.add_argument("--kgr-max", type=float, default=10.0,
                    help="absolute upper bound for the kgr sweep")
    args = ap.parse_args()

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    factors = np.geomspace(args.factor_min, args.factor_max, args.n_points)
    # kgr swept over an absolute range (2-10 by default) rather than the shared
    # multiplicative factor grid; the fitted value is always kept in the grid.
    abs_ranges = {"kgr": (args.kgr_min, args.kgr_max)} if "kgr" in params else {}
    print(f"[sensitivity] params={params}  factors {args.factor_min}-{args.factor_max} "
          f"({args.n_points} pts)  kgr abs [{args.kgr_min},{args.kgr_max}]  "
          f"{args.n_reps} reps  fit-dir={args.fit_dir.name}")
    df, op = sweep(args.fit_dir, params=params, factors=factors, n_reps=args.n_reps,
                   base_seed=args.seed, prom_sigma=args.prom_sigma,
                   min_distance_min=args.min_distance_min, resample_dt_min=args.resample_dt_min,
                   abs_ranges=abs_ranges, n_points=args.n_points)

    elas_rows = []
    for p in params:
        sub = df[df["param"] == p]
        elas_rows.append(dict(param=p, elas_ipi=_elasticity(sub, "ipi"),
                              elas_amp=_elasticity(sub, "amp")))
    elas_df = pd.DataFrame(elas_rows)

    art = args.out / "artifacts"; art.mkdir(parents=True, exist_ok=True)
    df.to_csv(art / "sensitivity_sweep.csv", index=False)
    elas_df.to_csv(art / "sensitivity_elasticity.csv", index=False)
    print("\n  elasticity (d ln metric / d ln param) at operating point:")
    for _, r in elas_df.sort_values("elas_ipi", key=lambda s: s.abs(), ascending=False).iterrows():
        print(f"    {r['param']:4s}  IPI {r['elas_ipi']:+.2f}   amp {r['elas_amp']:+.2f}")

    (args.out / "manifest.json").write_text(json.dumps({
        "task": "build_sensitivity_ipi_amplitude",
        "fit_dir": str(args.fit_dir),
        "operating_point": {k: op[k] for k in PARAMS},
        "drive": "constant baseline (circadian zeroed) + canonical lognormal drive noise",
        "params": params, "factor_grid": [args.factor_min, args.factor_max, args.n_points],
        "n_reps": args.n_reps, "prom_sigma": args.prom_sigma,
        "min_distance_min": args.min_distance_min,
        "elasticity": {r["param"]: {"ipi": r["elas_ipi"], "amp": r["elas_amp"]}
                       for _, r in elas_df.iterrows()},
    }, indent=2))
    png1 = build_curves_figure(df, params, args.out)
    png2 = build_tornado_figure(elas_df, args.out)
    png3 = build_grid_figure(df, params, args.out)
    print(f"\nPNG: {png1}\nPNG: {png2}\nPNG: {png3}")


if __name__ == "__main__":
    main()
