"""Combined Walker limit-cycle vs data figure (Fig 3 + Fig 4 layout).

A single figure comparing the Walker (2010) delay limit cycle — deterministic and
with matched lognormal drive noise — against cortisol data, in the visual idiom of
the manuscript's Fig 3 (example trajectories) stacked over Fig 4 (per-time-of-day
binned pulse statistics).

  TOP row — three example panels of z-scored cortisol (solid) and ACTH (dashed)
  over 24 h, ordered data -> model-with-noise -> model-without-noise:
    A  one representative HABS subject (real data),
    B  noisy Walker LC (ε=ε*) — pulsatile,
    C  deterministic Walker LC (ε=0) — metronomic.

  BOTTOM 2×2 — peak AMPLITUDE and IPI, mean and CV, binned into the 5 time-of-day
  bins, with three grouped boxes per bin (pooled HABS data / det LC / noisy LC):
    D  amplitude mean,   E  IPI mean,
    F  amplitude CV (+ Rayleigh CV reference),   G  IPI CV.

The deterministic LC's CV boxes collapse toward zero (metronomic timing); the data
and the noisy LC show matched spread. Pooled HABS peaks (the full --peaks-csv) are
the data source in the bottom panels; the single --data-example-id subject is only
the top-row data trace.

Reuses helpers from scripts/compare_limit_cycle_vs_noise.py (clc),
scripts/build_figure5_lc_comparison.py (f5), and scripts/noisy_limit_cycle_sweep.py
(nls). By default it REUSES the cached Walker fits from the limit-cycle fit run
(--fit-cache); pass --refit to re-run the DE fits.

Usage:
  PYTHONPATH=src python scripts/build_walker_lc_combined_figure.py \
      --fit-dir experiments/runs/eps15_acth20_cort15 \
      --peaks-csv experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv \
      --fit-cache experiments/runs/limit_cycle_vs_noise_acth20_fig5/artifacts/fit_cache.json \
      --walker-noise-eps 1.5 --lc-n-days 120 --n-reps 200 --data-example-id 1 \
      --out experiments/runs/walker_lc_combined_figure
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
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_figure5_lc_comparison as f5  # noqa: E402  (provides WALKER_BOUNDS for --refit)
import compare_limit_cycle_vs_noise as clc  # noqa: E402
import noisy_limit_cycle_sweep as nls  # noqa: E402
from hpa_model.plotting import apply_paper_style, setup_nature_style  # noqa: E402

# Colours: one per source, reused between the trace panels and the binned boxes.
# The two Walker variants share a hue (purple); the noisy LC is the darker shade.
DATA_COLOR = clc.DATA_COLOR        # blue         — data
NOISY_LC_COLOR = "#54278F"         # dark purple  — limit cycle + noise
DET_LC_COLOR = "#9E9AC8"           # light purple — deterministic limit cycle
ACTH_COLOR = "#8C8C8C"             # gray         — ACTH trace (hormone-identity convention)

# Time-of-day bins — identical convention to manuscript Fig 4 (04:00 origin, the
# 20:00–04:00 overnight window forms a single wrapping bin placed last).
TOD_ORIGIN_MIN = 240.0
BIN_EDGES = [0.0, 240.0, 480.0, 720.0, 960.0, 1440.0]
BIN_LABELS = ["04-08", "08-12", "12-16", "16-20", "20-04"]

RAYLEIGH_CV = float(np.sqrt((4.0 - np.pi) / np.pi))  # ≈ 0.523

# Model trace panels are sampled on the data's 20-min grid for display, so the model
# is drawn at the same resolution as the data (not its 0.25-min integration step).
TRACE_SAMPLE_DT_MIN = 20.0


# ── per-realisation peak rows for the noisy Walker LC ────────────────────────────
# nls._noisy_walker_peaks discards peak times + uids, so it cannot feed the per-bin
# boxplots. This mirrors it but emits one row per peak (uid = realisation), exactly
# the schema clc._pool_daily_peaks / clc._data_rows produce.

def _noisy_walker_rows(cp, params, eps, *, reps, dt_min, warmup_min,
                       prom_sigma, min_distance_min, seed0):
    drive_fn = clc._two_harmonic(cp, with_noise=False).base_value
    p3 = clc.p3_from_half_lives(clc.CORT_HALF_LIFE_MIN, clc.ACTH_HALF_LIFE_MIN)
    rows = []
    for rep in range(reps):
        sim = clc.simulate_walker(
            p1=params["p1"], p2=params["p2"], p4=params["p4"], p5=params["p5"],
            p6=params["p6"], tau_min=params["tau_min"], p3=p3,
            cort_half_life_min=clc.CORT_HALF_LIFE_MIN, dt_min=dt_min,
            warmup_min=warmup_min, duration_min=1440.0, drive_fn=drive_fn,
            epsilon=float(eps), noise_form=cp["noise_form"], seed=seed0 + rep)
        pt, a = clc._prev_dip_amps_z(sim["time_min"], sim["o"], prom_sigma=prom_sigma,
                                     min_distance_min=min_distance_min)
        ipi = np.full(pt.size, np.nan)
        if pt.size >= 2:
            ipi[:-1] = np.diff(pt)
        for i in range(pt.size):
            rows.append({"uid": f"rep{rep}", "tod_min": float(pt[i]),
                         "amp": float(a[i]), "ipi": float(ipi[i])})
    return pd.DataFrame(rows, columns=["uid", "tod_min", "amp", "ipi"])


# ── binning + per-bin statistics (Fig-4 convention) ──────────────────────────────

def _add_bin(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["uid", "tod_min", "amp", "ipi", "bin"])
    tod = (df["tod_min"].to_numpy(float) % 1440.0 - TOD_ORIGIN_MIN) % 1440.0
    idx = np.clip(np.digitize(tod, BIN_EDGES, right=False) - 1, 0, len(BIN_LABELS) - 1)
    out = df.copy()
    out["bin"] = [BIN_LABELS[i] for i in idx]
    return out


def _cap_ipi(df, max_ipi_min):
    out = df.copy()
    bad = ~np.isfinite(out["ipi"]) | (out["ipi"] <= 0) | (out["ipi"] > float(max_ipi_min))
    out.loc[bad, "ipi"] = np.nan
    return out


def _per_bin_stat(df, value_col, stat, min_peaks):
    """Per (uid, bin) mean or CV of value_col, dropping cells with < min_peaks
    finite (and, for amplitude, positive) values. Mirrors Fig 4's _per_uid_bin_stat
    / clc._per_series_bin_stat."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["uid", "bin", "value"])
    out = []
    for (uid, b), g in df.groupby(["uid", "bin"], observed=True):
        v = g[value_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if value_col == "amp":
            v = v[v > 0]
        if v.size < min_peaks:
            continue
        if stat == "mean":
            out.append({"uid": uid, "bin": str(b), "value": float(np.mean(v))})
        else:
            m = float(np.mean(v))
            if m > 0 and v.size > 1:
                out.append({"uid": uid, "bin": str(b),
                            "value": float(np.std(v, ddof=1) / m)})
    return pd.DataFrame(out)


def _box_trio(ax, longs, colors, *, ylabel, title, ylim=None):
    """Three grouped boxes per time-of-day bin (data / det-LC / noisy-LC).
    Adapted from clc._box_groups with this module's Fig-4 bin labels."""
    pos = np.arange(len(BIN_LABELS))
    n = len(longs)
    gw = 0.84
    bw = gw / n * 0.82
    for k, (lf, c) in enumerate(zip(longs, colors)):
        off = (k - (n - 1) / 2) * (gw / n)
        by_bin = [lf.loc[lf["bin"] == b, "value"].dropna().to_numpy(float)
                  if not lf.empty else np.array([]) for b in BIN_LABELS]
        ax.boxplot([v if len(v) else np.array([np.nan]) for v in by_bin],
                   positions=pos + off, widths=bw, patch_artist=True, showfliers=False,
                   medianprops={"color": "white", "linewidth": 1.2},
                   boxprops={"facecolor": c, "alpha": 0.55, "linewidth": 0.6, "edgecolor": c},
                   whiskerprops={"linewidth": 0.6, "color": c},
                   capprops={"linewidth": 0.6, "color": c})
    ax.set_xticks(pos)
    ax.set_xticklabels(BIN_LABELS, fontsize=9.5, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(BIN_LABELS) - 0.4)
    ax.set_xlabel("Time of day (h)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11.5, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)
    apply_paper_style(ax)


def _align_pair(t_min, o, a, acro_data):
    """Slice a 24 h window whose cortisol (o) circadian acrophase sits at acro_data,
    SAMPLE both hormones on the data's 20-min grid (so the model trace is drawn at the
    same resolution as the data, not its 0.25-min integration step), then z-score each
    (display only; mirrors the data's per-subject z-scoring)."""
    t_min = np.asarray(t_min, float)
    o = np.asarray(o, float); a = np.asarray(a, float)
    dt = float(np.median(np.diff(t_min)))
    n_day = int(round(1440.0 / dt))
    acro = clc._circadian_acrophase_min(t_min, o)
    shift = float((acro - acro_data) % 1440.0)
    i0 = int(np.searchsorted(t_min, shift, side="left"))
    i0 = max(0, min(i0, len(t_min) - n_day))
    sl = slice(i0, i0 + n_day)
    tw = t_min[sl] - t_min[i0]                       # minutes from window start
    grid = np.arange(0.0, 1440.0, TRACE_SAMPLE_DT_MIN)
    o_s, a_s = np.interp(grid, tw, o[sl]), np.interp(grid, tw, a[sl])

    def _z(v):
        return (v - v.mean()) / v.std() if v.std() > 1e-9 else np.zeros_like(v)

    return grid / 60.0, _z(o_s), _z(a_s)


def _walker_det_pair(cp, params, *, dt_min, n_days=2):
    """Deterministic Walker window returning (time_min, cortisol, ACTH)."""
    drive_fn = clc._two_harmonic(cp, with_noise=False).base_value
    p3 = clc.p3_from_half_lives(clc.CORT_HALF_LIFE_MIN, clc.ACTH_HALF_LIFE_MIN)
    sim = clc.simulate_walker(
        p1=params["p1"], p2=params["p2"], p4=params["p4"], p5=params["p5"],
        p6=params["p6"], tau_min=params["tau_min"], p3=p3,
        cort_half_life_min=clc.CORT_HALF_LIFE_MIN, dt_min=dt_min,
        warmup_min=clc.LC_WARMUP_MIN, duration_min=1440.0 * n_days, drive_fn=drive_fn)
    return sim["time_min"], sim["o"], sim["a"]


def _walker_noisy_pair(cp, params, eps, *, dt_min, seed):
    """Noisy Walker window returning (time_min, cortisol, ACTH) for one realisation."""
    drive_fn = clc._two_harmonic(cp, with_noise=False).base_value
    p3 = clc.p3_from_half_lives(clc.CORT_HALF_LIFE_MIN, clc.ACTH_HALF_LIFE_MIN)
    sim = clc.simulate_walker(
        p1=params["p1"], p2=params["p2"], p4=params["p4"], p5=params["p5"],
        p6=params["p6"], tau_min=params["tau_min"], p3=p3,
        cort_half_life_min=clc.CORT_HALF_LIFE_MIN, dt_min=dt_min,
        warmup_min=2880.0, duration_min=2880.0, drive_fn=drive_fn,
        epsilon=float(eps), noise_form=cp["noise_form"], seed=seed)
    return sim["time_min"], sim["o"], sim["a"]


def _data_example_raw(variant, series_id, dataset):
    """HABS subject cortisol AND ACTH, each z-scored RAW (circadian envelope retained,
    NOT detrended), plus the cortisol circadian acrophase from a 24h+12h fit (used only
    to align the model windows). Both hormones are processed like the model traces, so
    all top-row panels are comparable. Returns (t_cort_h, z_cort, t_acth_h, z_acth, acro);
    z_acth/t_acth are None if the dataset has no usable ACTH column."""
    spec = clc.get_dataset_spec(dataset)
    df = clc.load_dataset(dataset, variant)
    sub = df[df[spec.id_col].astype(str) == str(series_id)].sort_values(spec.time_col)
    cort_col = next((c for c in ("Cortisol", "cortisol", "value") if c in sub.columns), None)
    if cort_col is None:
        raise KeyError(f"no cortisol column in dataset '{dataset}': {list(sub.columns)}")
    acth_col = next((c for c in ("ACTH", "acth") if c in sub.columns), None)

    def _series(col):
        s = sub[[spec.time_col, col]].dropna()
        t = s[spec.time_col].to_numpy(float); x = s[col].to_numpy(float)
        z = (x - x.mean()) / x.std() if x.std() > 1e-9 else np.zeros_like(x)
        return t / 60.0, z

    th_c, z_c = _series(cort_col)
    sc = sub[[spec.time_col, cort_col]].dropna()
    p = clc.fit_two_harmonic_params(sc[spec.time_col].to_numpy(float),
                                    sc[cort_col].to_numpy(float),
                                    period_min=1440.0, second_period_min=720.0)
    grid = np.arange(0.0, 1440.0, 1.0)
    acro = (float(grid[int(np.argmax(clc.evaluate_two_harmonic(grid, p)))])
            if p is not None else 0.0)
    if acth_col is not None and sub[acth_col].notna().any():
        th_a, z_a = _series(acth_col)
    else:
        th_a, z_a = None, None
    return th_c, z_c, th_a, z_a, acro


def _pooled_stats(df):
    """Pooled (across all uids/bins) amp/ipi mean and CV for a peak-row table."""
    amp = df["amp"].to_numpy(float)
    amp = amp[np.isfinite(amp) & (amp > 0)]
    ipi = df["ipi"].to_numpy(float)
    ipi = ipi[np.isfinite(ipi) & (ipi > 0)]
    return dict(n_peaks=int(amp.size),
                amp_mean=float(np.mean(amp)) if amp.size else float("nan"),
                amp_cv=float(clc._cv(amp)),
                ipi_mean=float(np.mean(ipi)) if ipi.size else float("nan"),
                ipi_cv=float(clc._ipi_cv(ipi)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/eps15_acth20_cort15")
    ap.add_argument("--peaks-csv", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv")
    ap.add_argument("--fit-cache", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/limit_cycle_vs_noise_acth20_fig5/artifacts/fit_cache.json",
                    help="Cached {walker_det, walker_noisy, walker_noisy_eps} reused by default.")
    ap.add_argument("--walker-noise-eps", type=float, default=1.5,
                    help="ε* for the noisy Walker LC (must match cached walker_noisy_eps unless --refit).")
    ap.add_argument("--noisy-traj-seed", type=int, default=9,
                    help="Seed for the noisy-LC example trace (panel B); chosen as a "
                         "representative data-like realization.")
    ap.add_argument("--n-reps", type=int, default=200,
                    help="Stochastic realisations pooled for the noisy-LC peak distribution.")
    ap.add_argument("--lc-n-days", type=int, default=120,
                    help="Days of deterministic Walker LC pooled (each day = one pseudo-subject).")
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--max-ipi-min", type=float, default=240.0)
    ap.add_argument("--walker-dt-min", type=float, default=0.25)
    ap.add_argument("--variant", type=str, default="shifted_12h")
    ap.add_argument("--data-example-dataset", type=str, default="habs")
    ap.add_argument("--data-example-id", type=str, default="1")
    ap.add_argument("--walker-maxiter", type=int, default=40, help="(only with --refit)")
    ap.add_argument("--walker-noise-maxiter", type=int, default=12, help="(only with --refit)")
    ap.add_argument("--walker-noise-fitreps", type=int, default=16, help="(only with --refit)")
    ap.add_argument("--refit", action="store_true",
                    help="Ignore --fit-cache and re-run the slow DE fits (writes a new cache into --out).")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/walker_lc_combined_figure")
    args = ap.parse_args()

    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "artifacts").mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    eps_star = float(args.walker_noise_eps)

    # ── circadian drive + data targets ───────────────────────────────────────────
    cp = clc._circadian_params(args.fit_dir)
    amp_data, amp_mean, ipi_data = clc._data_targets(args.peaks_csv)
    ipi_mean = float(np.mean(ipi_data))
    print(f"[data]      amp_mean={amp_mean:.3f} ipi_mean={ipi_mean:.1f} "
          f"ampCV={clc._cv(amp_data):.3f} ipiCV={clc._ipi_cv(ipi_data):.3f}")
    print(f"[circadian] a24={cp['a24']:.3f} a12={cp['a12']:.3f} ε_drive(fit)={cp['epsilon']:.2f} "
          f"ε*_walker={eps_star:.2f}")

    # ── Walker fits (reuse cache by default) ─────────────────────────────────────
    cached = {}
    if args.fit_cache.exists() and not args.refit:
        try:
            cached = json.loads(args.fit_cache.read_text())
        except Exception:  # noqa: BLE001
            cached = {}

    if not args.refit and cached.get("walker_det") and cached.get("walker_noisy"):
        if abs(float(cached.get("walker_noisy_eps", -1)) - eps_star) > 1e-9:
            raise SystemExit(
                f"Cached walker_noisy was fit at ε={cached.get('walker_noisy_eps')} but "
                f"--walker-noise-eps={eps_star}. A noisy fit is only valid at its fitted ε — "
                f"either set --walker-noise-eps {cached.get('walker_noisy_eps')} or pass --refit.")
        w_det, w_noisy = cached["walker_det"], cached["walker_noisy"]
        print(f"[walker]    reused cached fits  det τ={w_det['tau_min']:.1f}min  "
              f"noisy τ={w_noisy['tau_min']:.1f}min (ε={eps_star})  [{args.fit_cache}]")
    else:
        print("[walker-det]   fitting deterministic Walker LC...")
        w_det = clc._fit_walker(cp, data_amp=amp_mean, data_ipi=ipi_mean,
                                dt_min=args.walker_dt_min, prom_sigma=args.prom_sigma,
                                min_distance_min=args.min_distance_min,
                                bounds=f5.WALKER_BOUNDS, maxiter=args.walker_maxiter)
        print(f"[walker-noisy] re-fitting Walker LC with noise ε={eps_star}...")
        w_noisy = nls._refit_walker_noisy(
            cp, eps_star, data_amp=amp_mean, data_ipi=ipi_mean, bounds=f5.WALKER_BOUNDS,
            dt_min=args.walker_dt_min, warmup_min=2880.0,
            fit_reps=args.walker_noise_fitreps, prom_sigma=args.prom_sigma,
            min_distance_min=args.min_distance_min, maxiter=args.walker_noise_maxiter, seed0=7)
        (args.out / "artifacts" / "fit_cache.json").write_text(json.dumps(
            {"walker_det": w_det, "walker_noisy": w_noisy, "walker_noisy_eps": eps_star},
            indent=2))

    # ── peak-row tables for the three sources ────────────────────────────────────
    data_tbl = _cap_ipi(_add_bin(clc._data_rows(args.peaks_csv)), args.max_ipi_min)

    _ad, _id, df_det = clc._walker_peaks(
        cp, w_det, dt_min=args.walker_dt_min, n_days=args.lc_n_days,
        prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min)
    det_tbl = _cap_ipi(_add_bin(df_det), args.max_ipi_min)

    df_noisy = _noisy_walker_rows(
        cp, w_noisy, eps_star, reps=args.n_reps, dt_min=args.walker_dt_min,
        warmup_min=2880.0, prom_sigma=args.prom_sigma,
        min_distance_min=args.min_distance_min, seed0=500)
    noisy_tbl = _cap_ipi(_add_bin(df_noisy), args.max_ipi_min)

    # Box-group order: data, LC with noise, LC without noise.
    tables = [data_tbl, noisy_tbl, det_tbl]
    colors = [DATA_COLOR, NOISY_LC_COLOR, DET_LC_COLOR]
    labels = ["Data (HABS, pooled)",
              f"Walker LC + noise (ε={eps_star:.1f})",
              "Walker LC (deterministic, ε=0)"]
    for lab, tbl in zip(labels, tables):
        s = _pooled_stats(tbl)
        print(f"[{lab[:22]:>22}] n={s['n_peaks']:5d} ampMean={s['amp_mean']:.3f} "
              f"ampCV={s['amp_cv']:.3f} ipiMean={s['ipi_mean']:.1f} ipiCV={s['ipi_cv']:.3f}")
        # warn on empty bins for the mean panels (det LC needs enough days)
        empty = [b for b in BIN_LABELS if tbl[tbl["bin"] == b].empty]
        if empty:
            print(f"    ! empty TOD bins for '{lab[:22]}': {empty} (raise --lc-n-days)")

    def longs(col, stat, mp):
        return [_per_bin_stat(t, col, stat, mp) for t in tables]

    amp_cv_longs = longs("amp", "cv", 2)
    ipi_cv_longs = longs("ipi", "cv", 2)
    cv_vals = pd.concat([lf["value"] for lf in (amp_cv_longs + ipi_cv_longs)
                         if not lf.empty], ignore_index=True) if any(
        not lf.empty for lf in amp_cv_longs + ipi_cv_longs) else pd.Series([1.0])
    cv_ylim = (0.0, float(np.nanquantile(cv_vals.to_numpy(float), 0.97)) * 1.15)

    # ── figure ───────────────────────────────────────────────────────────────────
    setup_nature_style()
    fig = plt.figure(figsize=(11.0, 11.0))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.85, 1.0, 1.0],
                           hspace=0.55, wspace=0.28,
                           left=0.085, right=0.985, top=0.94, bottom=0.06)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.28)

    # Top row — example z-scored cortisol (solid) + ACTH (dashed) over 24 h,
    # ordered data -> LC + noise -> deterministic LC.
    dc_th, dc_zc, da_th, da_za, acro_data = _data_example_raw(
        args.variant, args.data_example_id, args.data_example_dataset)
    det_t, det_o, det_a = _walker_det_pair(cp, w_det, dt_min=args.walker_dt_min, n_days=2)
    nz_t, nz_o, nz_a = _walker_noisy_pair(cp, w_noisy, eps_star,
                                          dt_min=args.walker_dt_min, seed=args.noisy_traj_seed)
    det_th, det_zo, det_za = _align_pair(det_t, det_o, det_a, acro_data)
    nz_th, nz_zo, nz_za = _align_pair(nz_t, nz_o, nz_a, acro_data)
    # (t_cort, z_cort, t_acth, z_acth, colour, title)
    traces = [
        (dc_th, dc_zc, da_th, da_za, DATA_COLOR,
         f"Representative HABS subject {args.data_example_id} (data)"),
        (nz_th, nz_zo, nz_th, nz_za, NOISY_LC_COLOR,
         f"Walker LC + noise (ε={eps_star:.1f})"),
        (det_th, det_zo, det_th, det_za, DET_LC_COLOR,
         "Walker LC (deterministic, ε=0)"),
    ]
    _allz = np.concatenate(
        [zc for _, zc, _, _, _, _ in traces]
        + [za for _, _, _, za, _, _ in traces if za is not None])
    _zlo, _zhi = float(np.nanmin(_allz)), float(np.nanmax(_allz))
    _pad = 0.07 * (_zhi - _zlo)
    ylim = (_zlo - _pad, _zhi + _pad)
    for col, (tc, zc, ta, za, c, ttl) in enumerate(traces):
        ax = fig.add_subplot(gs_top[0, col])
        ax.plot(tc, zc, color=c, lw=1.4, zorder=3)
        if za is not None:
            ax.plot(ta, za, color=ACTH_COLOR, lw=1.1, ls="--", alpha=0.9, zorder=2)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6))
        ax.set_ylim(*ylim)
        ax.set_xlabel("Time of day (h)", fontsize=10.5)
        if col == 0:
            ax.set_ylabel("Hormone (z-score)", fontsize=10.5)
            ax.legend(handles=[
                Line2D([0], [0], color="#1A1A1A", lw=1.4, label="Cortisol"),
                Line2D([0], [0], color=ACTH_COLOR, lw=1.1, ls="--", label="ACTH")],
                loc="upper right", fontsize=8.0, frameon=False,
                handlelength=1.8, borderaxespad=0.2)
        ax.set_title(("ABC"[col]) + "  " + ttl, fontsize=10.0, loc="left", color=c)
        apply_paper_style(ax)

    # Bottom 2×2 — per-time-of-day binned pulse statistics.
    axDmean = fig.add_subplot(gs[1, 0])
    axEmean = fig.add_subplot(gs[1, 1])
    axFcv = fig.add_subplot(gs[2, 0])
    axGcv = fig.add_subplot(gs[2, 1])
    _box_trio(axDmean, longs("amp", "mean", 1), colors,
              ylabel="Amplitude mean (Z)", title="D  Amplitude mean")
    _box_trio(axEmean, longs("ipi", "mean", 1), colors,
              ylabel="IPI mean (min)", title="E  Inter-peak interval mean")
    _box_trio(axFcv, amp_cv_longs, colors,
              ylabel="Amplitude CV", title="F  Amplitude CV", ylim=cv_ylim)
    _box_trio(axGcv, ipi_cv_longs, colors,
              ylabel="IPI CV", title="G  Inter-peak interval CV", ylim=cv_ylim)

    fig.legend(handles=[Patch(facecolor=c, alpha=0.55, edgecolor=c, label=lab)
                        for c, lab in zip(colors, labels)],
               loc="upper center", bbox_to_anchor=(0.5, 0.70), ncol=3,
               frameon=False, fontsize=10)

    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"walker_lc_combined_figure.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── outputs ──────────────────────────────────────────────────────────────────
    data_tbl.to_csv(args.out / "artifacts" / "data_peaks.csv", index=False)
    det_tbl.to_csv(args.out / "artifacts" / "det_lc_peaks.csv", index=False)
    noisy_tbl.to_csv(args.out / "artifacts" / "noisy_lc_peaks.csv", index=False)
    summary = dict(
        circadian=cp, eps_star_walker=eps_star,
        walker_det={k: w_det[k] for k in clc.WALKER_KEYS} | {"loss": w_det.get("loss")},
        walker_noisy={k: w_noisy[k] for k in clc.WALKER_KEYS} | {"loss": w_noisy.get("loss")},
        sources={
            "data": _pooled_stats(data_tbl),
            "det_lc": _pooled_stats(det_tbl),
            "noisy_lc": _pooled_stats(noisy_tbl),
        })
    (args.out / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "manifest.json").write_text(json.dumps(
        {"task": "build_walker_lc_combined_figure",
         "created_at": datetime.now(UTC).isoformat(),
         "fit_dir": str(args.fit_dir), "peaks_csv": str(args.peaks_csv),
         "fit_cache": str(args.fit_cache), "refit": bool(args.refit),
         "eps_star_walker": eps_star, "lc_n_days": args.lc_n_days, "n_reps": args.n_reps,
         "noisy_traj_seed": args.noisy_traj_seed,
         "data_example": {"dataset": args.data_example_dataset, "id": args.data_example_id,
                          "variant": args.variant}},
        indent=2))
    _write_readme(args, eps_star)
    print(f"[done] {args.out}")


def _rel(p):
    """Path relative to the repo root if possible, else the path as given."""
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _write_readme(args, eps_star):
    txt = f"""# Walker limit-cycle vs data — combined figure (Fig 3 + Fig 4 layout)

Single figure: TOP row = three example panels of z-scored cortisol (solid) and ACTH
(dashed) over 24 h, ordered data -> Walker LC + noise -> deterministic Walker LC; BOTTOM
2×2 = peak amplitude and IPI, mean and CV, binned into the 5 time-of-day bins
({", ".join(BIN_LABELS)}) with three grouped boxes per bin (pooled HABS data / noisy LC /
det LC). The two Walker variants share a purple hue (noisy = darker, deterministic =
lighter). The deterministic LC's CV boxes collapse toward zero (metronomic timing); the
data and noisy LC show matched spread.

Built by `scripts/build_walker_lc_combined_figure.py`. Walker fits are reused from the
limit-cycle fit run by default (`--fit-cache`); pass `--refit` to re-run the DE fits.
Script-built figures have no Hydra `resolved_config` (the effective args live in
`manifest.json`), consistent with the other manuscript-figure builders.

## Invocation
```
PYTHONPATH=src python scripts/build_walker_lc_combined_figure.py \\
    --fit-dir {_rel(args.fit_dir)} \\
    --peaks-csv {_rel(args.peaks_csv)} \\
    --fit-cache {_rel(args.fit_cache)} \\
    --walker-noise-eps {eps_star} --lc-n-days {args.lc_n_days} --n-reps {args.n_reps} \\
    --data-example-id {args.data_example_id} \\
    --out {_rel(args.out)}
```

## Outputs
- `figures/walker_lc_combined_figure.{{png,pdf}}` — the figure.
- `artifacts/{{data,det_lc,noisy_lc}}_peaks.csv` — binned peak rows per source.
- `artifacts/summary.json` — circadian params, Walker fits, pooled per-source stats.
- `manifest.json` — effective args / inputs / seeds.
"""
    (args.out / "README.md").write_text(txt)


if __name__ == "__main__":
    main()
