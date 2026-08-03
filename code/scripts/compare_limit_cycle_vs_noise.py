"""Delay limit-cycle models vs the noise-driven oscillator vs data.

Manuscript Results figure: show that **two** deterministic delay-based
limit-cycle models fail to reproduce the stochastic signature of cortisol
ultradian pulses, whereas the noise-driven damped oscillator (the canonical
model) succeeds. All conditions share the same circadian input.

Four conditions are compared:
  1. **Data** — pooled HABS+digitised cortisol peaks (manuscript pipeline).
  2. **Noise-driven** — canonical model (kgr=5, tau=0) with lognormal drive noise
     (epsilon from the v15 fit). This is our proposed mechanism.
  3. **Our model, delay limit cycle** — the SAME three-state HPA model at its
     biological feedback (kgr=5 fixed), deterministic, with the drive *baseline*
     and ACTH->cortisol *delay* free. At kgr=5 the system is a stable focus for
     all physiological delays, so it produces NO sustained ultradian pulses.
  4. **Walker (2010) delay limit cycle** — the Walker/Terry/Lightman pituitary
     -adrenal delay model, human-scaled from our half-lives (p3=CORT/ACTH=0.75,
     CORT half-life 15 min), with the GR-submodule params (p1,p2,p4,p5,p6) and
     delay fit to the data. It self-oscillates, but as a deterministic limit
     cycle its pulses are metronomic (near-zero IPI CV) and amplitude-multimodal
     (mode-locked to the circadian envelope).

The discriminators (panel B/C): amplitude-distribution multimodality (Hartigan
dip test) and the inter-peak-interval CV (data ~0.46 vs a limit cycle ~0).

Usage:
  PYTHONPATH=src python scripts/compare_limit_cycle_vs_noise.py \
      --fit-dir experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v15_cort15_rayleighcv \
      --peaks-csv experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv \
      --out experiments/runs/limit_cycle_vs_noise_v15
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import diptest
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from scipy.stats import kstest, rayleigh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpa_model.data.registry import get_dataset_spec, load_dataset  # noqa: E402
from hpa_model.data.two_harmonic_shift import (  # noqa: E402
    evaluate_two_harmonic, fit_two_harmonic_params,
)
from hpa_model.model.three_state_gr_delay import (  # noqa: E402
    ThreeStateGRDelayModel, TwoHarmonicDrive, TwoHarmonicNoiseDrive,
    rate_from_half_life,
)
from hpa_model.model.walker2010 import (  # noqa: E402
    WALKER_P2_RODENT, WALKER_P4_RODENT, WALKER_P5_RODENT, WALKER_P6_RODENT,
    p3_from_half_lives, simulate_walker,
)
from hpa_model.plotting import apply_paper_style, setup_nature_style  # noqa: E402
from hpa_model.simulate.engine import simulate_trajectory_fit_arrays  # noqa: E402

DATA_COLOR = "#2F5C85"      # blue
NOISE_COLOR = "#C85C3A"     # salmon
OUR_LC_COLOR = "#7E5CA8"    # purple — our model pushed to a delay limit cycle
WALKER_COLOR = "#3E8E5A"    # green — Walker (2010) delay limit cycle

# Time-of-day bins — same as manuscript Figs 2 & 4.
BIN_EDGES = [0.0, 240.0, 480.0, 720.0, 960.0, 1440.0]
BIN_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-24"]

# Long warmup so circadian-forced deterministic limit cycles settle onto their
# periodic steady state before we score peaks.
LC_WARMUP_MIN = 5760.0

# Human half-lives (our canonical model) used to human-scale the Walker model.
ACTH_HALF_LIFE_MIN = 20.0
CORT_HALF_LIFE_MIN = 15.0


# ── peak amplitude (within-window z-scored peak − previous trough) ──────────────

def _prev_dip_amps_z(times_min, x3, *, prom_sigma, min_distance_min, detrend="two_harmonic"):
    """Z-scored residual peak − previous-trough amplitudes (manuscript metric)."""
    if detrend == "two_harmonic":
        params = fit_two_harmonic_params(times_min, x3, period_min=1440.0,
                                         second_period_min=720.0)
        base = (evaluate_two_harmonic(times_min, params) if params is not None
                else np.full_like(x3, float(np.nanmean(x3))))
    else:
        base = np.full_like(x3, float(np.nanmean(x3)))
    resid = x3 - base
    std = float(np.nanstd(resid))
    if std <= 0:
        return np.empty(0), np.empty(0)
    rz = (resid - float(np.nanmean(resid))) / std
    dt = float(np.median(np.diff(times_min)))
    dist = max(1, int(round(min_distance_min / dt)))
    peaks, _ = find_peaks(rz, distance=dist, prominence=prom_sigma)
    if peaks.size == 0:
        return np.empty(0), np.empty(0)
    amps = np.empty(peaks.size)
    for i, p in enumerate(peaks):
        lo = 0 if i == 0 else int(peaks[i - 1])
        trough = float(rz[lo:p].min()) if p > lo else float(rz[p])
        amps[i] = float(rz[p]) - trough
    return times_min[peaks], amps


def _cv(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() > 0 else float("nan")


def _ipi_cv(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() > 0 else float("nan")


def _rayleigh_ks_p(v):
    """KS p-value for a Rayleigh fit (loc=0). High p = Rayleigh-consistent."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return float("nan")
    loc, sc = rayleigh.fit(v, floc=0.0)
    return float(kstest(v, "rayleigh", args=(loc, sc)).pvalue)


def _dip_matched(v, *, n_match, n_boot=300, seed=0):
    """Hartigan's dip test for unimodality at a common sample size (equal power).
    Returns (median p-value, fraction of subsamples rejecting unimodality)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return float("nan"), float("nan")
    if v.size < n_match:
        _, p = diptest.diptest(v)
        return float(p), float(p < 0.05)
    rng = np.random.default_rng(seed)
    ps = np.empty(n_boot)
    for i in range(n_boot):
        sample = v[rng.choice(v.size, size=n_match, replace=False)]
        ps[i] = diptest.diptest(sample)[1]
    return float(np.median(ps)), float(np.mean(ps < 0.05))


# ── our three-state model + shared circadian drive ──────────────────────────────

def _our_model(kgr, tau_min):
    a1 = rate_from_half_life(5.0); a2 = rate_from_half_life(ACTH_HALF_LIFE_MIN)
    a3 = rate_from_half_life(CORT_HALF_LIFE_MIN)
    return ThreeStateGRDelayModel(a1=a1, a2=a2, a3=a3, b1=a1, b2=a2, b3=a3,
                                  kgr=float(kgr), tau_min=float(tau_min),
                                  hill_coeff=3.0, initial_state=(1.0, 1.0, 1.0))


def _circadian_params(fit_dir):
    dp = yaml.safe_load((fit_dir / "artifacts" / "fitted_config.yaml").read_text())["drive"]["params"]
    return dict(a24=float(dp["a24"]), phase24=float(dp["phase24"]),
                a12=float(dp["a12"]), phase12=float(dp["phase12"]),
                baseline=float(dp.get("baseline", 1.0)),
                period_min=float(dp.get("period_min", 1440.0)),
                second_period_min=float(dp.get("second_period_min", 720.0)),
                epsilon=float(dp.get("epsilon", 1.0)),
                noise_form=str(dp.get("noise_form", "lognormal")))


def _two_harmonic(cp, *, baseline=None, with_noise=False):
    b = cp["baseline"] if baseline is None else float(baseline)
    if with_noise:
        return TwoHarmonicNoiseDrive(
            a24=cp["a24"], phase24=cp["phase24"], a12=cp["a12"], phase12=cp["phase12"],
            baseline=b, epsilon=cp["epsilon"],
            period_min=cp["period_min"], second_period_min=cp["second_period_min"],
            noise_form=cp["noise_form"])
    return TwoHarmonicDrive(
        a24=cp["a24"], phase24=cp["phase24"], a12=cp["a12"], phase12=cp["phase12"],
        baseline=b, period_min=cp["period_min"], second_period_min=cp["second_period_min"])


def _sim_window(model, drive, *, dt_min, warmup_min, duration_min, seed):
    sim = simulate_trajectory_fit_arrays(
        model, drive, dt_min=dt_min, warmup_min=warmup_min,
        duration_min=duration_min, seed=seed, noise_locations=[], noise_epsilons={})
    return sim["time_min"], sim["x3"]


# ── pooled daily peaks for a deterministic circadian-forced trajectory ───────────

def _pool_daily_peaks(t, x, *, n_days, prom_sigma, min_distance_min):
    """Pool peak amplitudes + IPIs over many 24 h windows of a deterministic,
    circadian-forced trajectory. Each day is z-scored within itself (2-harmonic
    detrend) exactly like a data subject. Because the ultradian period is
    incommensurate with 24 h, successive days sample different circadian phases,
    so the pooled distribution is the true stationary one."""
    amps, ipis, rows = [], [], []
    for d in range(n_days):
        m = (t >= d * 1440.0) & (t < (d + 1) * 1440.0)
        if m.sum() < 10:
            continue
        tod = t[m] - d * 1440.0
        pt, a = _prev_dip_amps_z(tod, x[m], prom_sigma=prom_sigma,
                                 min_distance_min=min_distance_min)
        amps.extend(a.tolist())
        ipi = np.full(pt.size, np.nan)
        if pt.size >= 2:
            ipi[:-1] = np.diff(pt)
            ipis.extend(np.diff(pt).tolist())
        for i in range(pt.size):
            rows.append({"uid": f"day{d}", "tod_min": float(pt[i]),
                         "amp": float(a[i]), "ipi": float(ipi[i])})
    return np.asarray(amps, float), np.asarray(ipis, float), pd.DataFrame(rows)


# ── our-model delay limit cycle (free kgr + tau) ─────────────────────────────────
# Our model only self-oscillates at strong feedback (low kgr); at the fitted
# biological kgr=5 it is a stable focus (no limit cycle — see _our_lc_kgr5_check).
# Per the user's guidance, we let kgr go low to *produce* the limit cycle and then
# show that limit cycle fails the data's stochastic signature.

def _our_lc_window(cp, kgr, tau_min, *, dt_min, n_days):
    model = _our_model(kgr, tau_min)
    drive = _two_harmonic(cp, with_noise=False)
    return _sim_window(model, drive, dt_min=dt_min, warmup_min=LC_WARMUP_MIN,
                       duration_min=1440.0 * n_days, seed=0)


# Below this raw detrended-residual fraction (residual std / mean cortisol) the
# deterministic trace is judged to have no genuine ultradian pulses — within-day
# z-scoring otherwise inflates tiny numerical/harmonic residual into spurious
# "peaks". The data and Walker LC are well above this; our-model LC is ~2%.
MIN_RESID_FRAC = 0.05


def _resid_frac(t, x, *, n_days):
    """Mean per-day raw residual std / mean cortisol after 2-harmonic detrend."""
    fracs = []
    for d in range(n_days):
        m = (t >= d * 1440.0) & (t < (d + 1) * 1440.0)
        if m.sum() < 10:
            continue
        tod = t[m] - d * 1440.0; xx = x[m]
        p = fit_two_harmonic_params(tod, xx, period_min=1440.0, second_period_min=720.0)
        base = evaluate_two_harmonic(tod, p) if p is not None else np.full_like(xx, xx.mean())
        mean = float(np.mean(xx))
        if mean > 1e-9:
            fracs.append(float(np.std(xx - base) / mean))
    return float(np.mean(fracs)) if fracs else 0.0


def _our_lc_peaks(cp, kgr, tau_min, *, dt_min, n_days, prom_sigma, min_distance_min):
    t, x = _our_lc_window(cp, kgr, tau_min, dt_min=dt_min, n_days=n_days)
    rf = _resid_frac(t, x, n_days=n_days)
    if rf < MIN_RESID_FRAC:  # no genuine pulses — stable focus tracking the drive
        return np.empty(0), np.empty(0), pd.DataFrame(columns=["uid", "tod_min", "amp", "ipi"])
    return _pool_daily_peaks(t, x, n_days=n_days, prom_sigma=prom_sigma,
                             min_distance_min=min_distance_min)


def _our_lc_kgr5_check(cp, *, dt_min, prom_sigma, min_distance_min,
                       taus=(0.0, 10.0, 19.0, 30.0)):
    """At the fitted biological feedback (kgr=5) the model is a stable focus for all
    physiological delays — report the max detrended-residual fraction and genuine
    peak count to document the absence of a limit cycle there."""
    best = {"max_resid_frac": 0.0, "max_npeaks": 0}
    for tau in taus:
        t, x = _our_lc_window(cp, 5.0, tau, dt_min=dt_min, n_days=3)
        best["max_resid_frac"] = max(best["max_resid_frac"], _resid_frac(t, x, n_days=3))
        amps, _i, _r = _our_lc_peaks(cp, 5.0, tau, dt_min=dt_min, n_days=3,
                                     prom_sigma=prom_sigma, min_distance_min=min_distance_min)
        best["max_npeaks"] = max(best["max_npeaks"], int(amps.size))
    return best


def _fit_our_lc(cp, *, data_amp, data_ipi, dt_min, prom_sigma, min_distance_min,
                kgr_bounds, tau_bounds, maxiter, fit_n_days=6):
    """Fit (kgr, tau) of our model to the data's mean amplitude + IPI. kgr is free
    to go low (strong feedback) so the model can self-oscillate into a limit
    cycle; we then show that limit cycle fails the data's stochastic signature."""
    def loss(theta):
        kgr, tau = float(theta[0]), float(theta[1])
        amps, ipis, _ = _our_lc_peaks(cp, kgr, tau, dt_min=dt_min, n_days=fit_n_days,
                                      prom_sigma=prom_sigma, min_distance_min=min_distance_min)
        if amps.size < 4 or ipis.size < 2:
            return 1e3  # no limit cycle here
        amp, ipi = float(np.mean(amps)), float(np.mean(ipis))
        return ((amp - data_amp) / data_amp) ** 2 + ((ipi - data_ipi) / data_ipi) ** 2
    res = differential_evolution(loss, bounds=[tuple(kgr_bounds), tuple(tau_bounds)],
                                 maxiter=maxiter, popsize=10, tol=1e-3, seed=0,
                                 polish=True, workers=1, init="sobol")
    kgr, tau = float(res.x[0]), float(res.x[1])
    return dict(kgr=kgr, tau_min=tau, loss=float(res.fun))


# ── Walker (2010) delay limit cycle (human-scaled; free p1,p2,p4,p5,p6,tau) ──────

WALKER_KEYS = ("p1", "p2", "p4", "p5", "p6", "tau_min")


def _walker_window(cp, params, *, dt_min, n_days):
    drive = _two_harmonic(cp, with_noise=False)
    drive_fn = drive.base_value  # mean ~1 circadian envelope (shared input)
    p3 = p3_from_half_lives(CORT_HALF_LIFE_MIN, ACTH_HALF_LIFE_MIN)
    sim = simulate_walker(
        p1=params["p1"], p2=params["p2"], p4=params["p4"], p5=params["p5"],
        p6=params["p6"], tau_min=params["tau_min"], p3=p3,
        cort_half_life_min=CORT_HALF_LIFE_MIN, dt_min=dt_min, warmup_min=LC_WARMUP_MIN,
        duration_min=1440.0 * n_days, drive_fn=drive_fn)
    return sim["time_min"], sim["o"]


def _walker_peaks(cp, params, *, dt_min, n_days, prom_sigma, min_distance_min):
    t, x = _walker_window(cp, params, dt_min=dt_min, n_days=n_days)
    return _pool_daily_peaks(t, x, n_days=n_days, prom_sigma=prom_sigma,
                             min_distance_min=min_distance_min)


def _fit_walker(cp, *, data_amp, data_ipi, dt_min, prom_sigma, min_distance_min,
                bounds, maxiter, fit_n_days=8):
    """Fit (p1,p2,p4,p5,p6,tau) — p3 fixed at the human ratio — to the data's mean
    amplitude + IPI, giving the Walker limit cycle its best chance to match."""
    def loss(theta):
        params = dict(zip(WALKER_KEYS, (float(v) for v in theta)))
        amps, ipis, _ = _walker_peaks(cp, params, dt_min=dt_min, n_days=fit_n_days,
                                      prom_sigma=prom_sigma, min_distance_min=min_distance_min)
        if amps.size < 4 or ipis.size < 2:
            return 1e3
        amp, ipi = float(np.mean(amps)), float(np.mean(ipis))
        return ((amp - data_amp) / data_amp) ** 2 + ((ipi - data_ipi) / data_ipi) ** 2
    res = differential_evolution(loss, bounds=[tuple(bounds[k]) for k in WALKER_KEYS],
                                 maxiter=maxiter, popsize=12, tol=1e-3, seed=0,
                                 polish=True, workers=1, init="sobol")
    params = dict(zip(WALKER_KEYS, (float(v) for v in res.x)))
    params["loss"] = float(res.fun)
    return params


# ── data ────────────────────────────────────────────────────────────────────────

def _data_targets(peaks_csv):
    df = pd.read_csv(peaks_csv)
    amp = df["peak_amplitude_prev_dip_sigma"].to_numpy(float)
    amp = amp[np.isfinite(amp) & (amp > 0)]
    ipis = []
    for _uid, g in df.sort_values("time_min").groupby("series_uid"):
        d = np.diff(g["time_min"].to_numpy(float))
        ipis.extend(d[(d > 0) & (d <= 240.0)].tolist())
    return amp, float(np.mean(amp)), np.asarray(ipis, float)


def _data_rows(peaks_csv):
    df = pd.read_csv(peaks_csv)
    df = df[np.isfinite(df["peak_amplitude_prev_dip_sigma"])].copy()
    df = df.rename(columns={"series_uid": "uid",
                            "peak_amplitude_prev_dip_sigma": "amp"})
    df = df.sort_values(["uid", "time_min"])
    df["ipi"] = df.groupby("uid")["time_min"].shift(-1) - df["time_min"]
    df.loc[(df["ipi"] <= 0) | (df["ipi"] > 240.0), "ipi"] = np.nan
    df["tod_min"] = df["time_min"] % 1440.0
    return df[["uid", "tod_min", "amp", "ipi"]]


def _boot_ci(v, fn, *, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.array([fn(v[i]) for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(fn(v)), float(lo), float(hi)


def _circadian_acrophase_min(t_min, x):
    """Time-of-day (minutes) at which the fitted 24h+12h circadian level peaks.
    Robust to large ultradian pulses — the least-squares two-harmonic fit picks
    out the slow circadian component only."""
    p = fit_two_harmonic_params(np.asarray(t_min, float), np.asarray(x, float),
                                period_min=1440.0, second_period_min=720.0)
    grid = np.arange(0.0, 1440.0, 1.0)
    base = evaluate_two_harmonic(grid, p) if p is not None else np.zeros_like(grid)
    return float(grid[int(np.argmax(base))])


def _data_example_trace(variant, series_id, dataset="digitize_2019"):
    spec = get_dataset_spec(dataset)
    df = load_dataset(dataset, variant)
    sub = df[df[spec.id_col].astype(str) == str(series_id)]
    # cortisol column name varies by dataset (habs: "Cortisol", digitized: "value"/"cortisol")
    cort_col = next((c for c in ("Cortisol", "cortisol", "value") if c in sub.columns), None)
    if cort_col is None:
        raise KeyError(f"no cortisol column in dataset '{dataset}': {list(sub.columns)}")
    sub = sub[[spec.time_col, cort_col]].dropna().sort_values(spec.time_col)
    t = sub[spec.time_col].to_numpy(float); x = sub[cort_col].to_numpy(float)
    params = fit_two_harmonic_params(t, x, period_min=1440.0, second_period_min=720.0)
    base = evaluate_two_harmonic(t, params) if params is not None else np.full_like(x, x.mean())
    rz = (x - base - np.mean(x - base)) / np.std(x - base)
    grid = np.arange(0.0, 1440.0, 1.0)
    acro_min = (float(grid[int(np.argmax(evaluate_two_harmonic(grid, params)))])
                if params is not None else 0.0)
    return t / 60.0, rz, acro_min


# ── main ──────────────────────────────────────────────────────────────────────

def _fit_cache_key(args):
    """Args that change the fitted (kgr,tau) / Walker params; used to invalidate
    the on-disk fit cache so a stale cache is never silently reused."""
    return dict(our_kgr_bounds=list(args.our_kgr_bounds),
                our_tau_bounds=list(args.our_tau_bounds),
                our_maxiter=args.our_maxiter, walker_maxiter=args.walker_maxiter,
                prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min,
                dt_min=args.dt_min, walker_dt_min=args.walker_dt_min,
                fit_dir=str(args.fit_dir), peaks_csv=str(args.peaks_csv))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v15_cort15_rayleighcv")
    ap.add_argument("--peaks-csv", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/runs/limit_cycle_vs_noise_v15")
    # our-model delay-LC fit: free kgr (low → limit cycle) + tau
    ap.add_argument("--our-kgr-bounds", type=float, nargs=2, default=[0.3, 4.0],
                    help="GR feedback range for the our-model LC fit; low kgr (strong "
                         "feedback) is needed for a delay limit cycle (fitted biological kgr=5).")
    ap.add_argument("--our-tau-bounds", type=float, nargs=2, default=[1.0, 50.0],
                    help="ACTH->cortisol delay range for the our-model LC fit.")
    # Walker (2010) human-scaled LC fit bounds
    ap.add_argument("--walker-maxiter", type=int, default=40)
    ap.add_argument("--our-maxiter", type=int, default=25)
    ap.add_argument("--lc-n-days", type=int, default=60,
                    help="Days of deterministic limit cycle pooled for its distribution.")
    ap.add_argument("--n-reps", type=int, default=200, help="Noise-model realizations.")
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--dt-min", type=float, default=1.0)
    ap.add_argument("--walker-dt-min", type=float, default=0.25)
    ap.add_argument("--variant", type=str, default="shifted_12h")
    ap.add_argument("--data-example-dataset", type=str, default="all_digitized",
                    help="Dataset for the Panel A example cortisol trace "
                         "(all_digitized = Young et al. 2004 digitized cortisol).")
    ap.add_argument("--data-example-id", type=str, default="6")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Override the drive-noise amplitude ε for the noise-driven "
                         "condition (default: use the fitted ε from --fit-dir). "
                         "Changes the noise simulations, so requires a full run "
                         "(not --replot).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refit", action="store_true",
                    help="Ignore any cached fit and re-run the (slow) DE fits.")
    ap.add_argument("--replot", action="store_true",
                    help="Redraw the figures instantly from the cached plot payload "
                         "(artifacts/plot_cache.pkl) without re-simulating or re-fitting. "
                         "Use for title/style tweaks; run once without --replot first.")
    ap.add_argument("--traj-only", action="store_true",
                    help="Render ONLY the example trajectory comparison (data vs Karin noise-only "
                         "vs the two delay limit cycles) as a standalone figure. No DE fitting and "
                         "no heavy sims — reuses cached delay-LC/Walker params (--lc-fit-cache), "
                         "which are ε-independent. Honors --fit-dir / --epsilon for the noise trace.")
    ap.add_argument("--lc-fit-cache", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/limit_cycle_vs_noise_v15/artifacts/fit_cache.json",
                    help="fit_cache.json giving our_fit (kgr,tau) + walker_fit for --traj-only "
                         "(LC params don't depend on ε, so the canonical v15 cache is reused).")
    args = ap.parse_args()

    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "artifacts").mkdir(parents=True, exist_ok=True)

    # ── fast path: redraw figures from the cached plot payload ───────────────────
    plot_cache_path = args.out / "artifacts" / "plot_cache.pkl"
    if args.replot:
        if args.epsilon is not None:
            ap.error("--epsilon changes the noise simulations and cannot be applied "
                     "with --replot; run a full (non-replot) run to a new --out dir.")
        if not plot_cache_path.exists():
            ap.error(f"--replot requested but no plot cache at {plot_cache_path}; "
                     "run once without --replot to build it.")
        with open(plot_cache_path, "rb") as fh:
            payload = pickle.load(fh)
        cp = payload["cp"]
        conditions = payload["conditions"]
        n_match = payload["n_match"]
        our_fit = payload["our_fit"]
        w_fit = payload["w_fit"]
        _apply_labels(conditions, cp, our_fit, w_fit)  # refresh labels from code
        noise_model = _our_model(5.0, 0.0)
        noise_drive = _two_harmonic(cp, with_noise=True)
        print(f"[replot] loaded cached plot data from {plot_cache_path}; "
              "redrawing figures only (no simulate/fit).")
        _figure_bottomline(args, cp, conditions, n_match, our_fit, w_fit,
                           noise_model, noise_drive)
        _figure_metrics(args, conditions)
        _figure_per_bin(args, conditions)
        print(f"[replot done] {args.out}")
        return

    # ── lightweight path: trajectory comparison only (no fitting, no heavy sims) ──
    if args.traj_only:
        cp = _circadian_params(args.fit_dir)
        if args.epsilon is not None:
            print(f"[override] drive-noise ε {cp['epsilon']:.3f} → {args.epsilon:.3f}")
            cp["epsilon"] = float(args.epsilon)
        if not args.lc_fit_cache.exists():
            ap.error(f"--traj-only needs delay-LC/Walker params; no cache at {args.lc_fit_cache}.")
        blob = json.loads(args.lc_fit_cache.read_text())
        our_fit, w_fit = blob["our_fit"], blob["walker_fit"]
        noise_model = _our_model(5.0, 0.0)
        noise_drive = _two_harmonic(cp, with_noise=True)
        print(f"[traj-only] ε={cp['epsilon']:.3f}; delay-LC/Walker params from "
              f"{args.lc_fit_cache} (no fitting).")
        _figure_trajectory_only(args, cp, our_fit, w_fit, noise_model, noise_drive)
        print(f"[traj-only done] {args.out}")
        return

    cp = _circadian_params(args.fit_dir)
    if args.epsilon is not None:
        print(f"[override] drive-noise ε {cp['epsilon']:.3f} → {args.epsilon:.3f}")
        cp["epsilon"] = float(args.epsilon)
    amp_data, data_amp_mean, ipi_data = _data_targets(args.peaks_csv)
    data_ipi_mean = float(np.mean(ipi_data))
    print(f"[data] mean amp={data_amp_mean:.3f}  mean IPI={data_ipi_mean:.1f}min  "
          f"n_amp={amp_data.size}  amp CV={_cv(amp_data):.3f}  IPI CV={_ipi_cv(ipi_data):.3f}")
    print(f"[circadian] a24={cp['a24']:.3f} a12={cp['a12']:.3f} baseline={cp['baseline']} "
          f"epsilon(noise)={cp['epsilon']:.3f}")

    # ── fit cache (skip the slow DE fits when args + inputs are unchanged) ───────
    fit_cache_path = args.out / "artifacts" / "fit_cache.json"
    fit_key = _fit_cache_key(args)
    cached_fits = None
    if not args.refit and fit_cache_path.exists():
        try:
            blob = json.loads(fit_cache_path.read_text())
            if blob.get("key") == fit_key:
                cached_fits = blob
                print(f"[cache] loaded fitted params from {fit_cache_path}; "
                      "skipping DE fits (use --refit to recompute).")
            else:
                print("[cache] fit cache present but args changed → refitting.")
        except Exception as e:  # noqa: BLE001
            print(f"[cache] could not read fit cache ({e}); refitting.")

    # ── 1) Noise-driven (kgr=5, tau=0, lognormal drive noise from v15) ──────────
    noise_model = _our_model(5.0, 0.0)
    noise_drive = _two_harmonic(cp, with_noise=True)
    amp_noise, ipi_noise, noise_rows = [], [], []
    for rep in range(args.n_reps):
        t, x = _sim_window(noise_model, noise_drive, dt_min=args.dt_min,
                           warmup_min=1440.0, duration_min=1440.0, seed=args.seed + rep)
        pt, a = _prev_dip_amps_z(t, x, prom_sigma=args.prom_sigma,
                                 min_distance_min=args.min_distance_min)
        amp_noise.extend(a.tolist())
        ipi = np.full(pt.size, np.nan)
        if pt.size >= 2:
            ipi[:-1] = np.diff(pt)
            ipi_noise.extend([v for v in np.diff(pt) if 0 < v <= 240.0])
        for i in range(pt.size):
            noise_rows.append({"uid": f"noise_r{rep}", "tod_min": float(pt[i]),
                               "amp": float(a[i]), "ipi": float(ipi[i])})
    amp_noise = np.asarray(amp_noise, float)
    ipi_noise = np.asarray(ipi_noise, float)
    noise_rows = pd.DataFrame(noise_rows)
    print(f"[noise-driven] n={amp_noise.size}  amp CV={_cv(amp_noise):.3f}  "
          f"IPI CV={_ipi_cv(ipi_noise):.3f}")

    # ── 2) Our model, delay limit cycle: free kgr (→ low) + tau ─────────────────
    kgr5 = _our_lc_kgr5_check(cp, dt_min=args.dt_min, prom_sigma=args.prom_sigma,
                              min_distance_min=args.min_distance_min)
    print(f"[our-model] biological feedback check: at kgr=5, max resid frac="
          f"{kgr5['max_resid_frac']:.2e}, max n_peaks={kgr5['max_npeaks']} "
          f"→ {'NO limit cycle (stable focus)' if kgr5['max_npeaks'] < 4 else 'pulses present'}")
    if cached_fits is not None:
        our_fit = cached_fits["our_fit"]
        print(f"[our-model LC] cached fit kgr={our_fit['kgr']:.3f} "
              f"tau={our_fit['tau_min']:.1f}min")
    else:
        our_fit = _fit_our_lc(cp, data_amp=data_amp_mean, data_ipi=data_ipi_mean,
                              dt_min=args.dt_min, prom_sigma=args.prom_sigma,
                              min_distance_min=args.min_distance_min,
                              kgr_bounds=args.our_kgr_bounds,
                              tau_bounds=args.our_tau_bounds, maxiter=args.our_maxiter)
    amp_ourlc, ipi_ourlc, ourlc_rows = _our_lc_peaks(
        cp, our_fit["kgr"], our_fit["tau_min"], dt_min=args.dt_min,
        n_days=args.lc_n_days, prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min)
    print(f"[our-model LC] fit kgr={our_fit['kgr']:.3f} tau={our_fit['tau_min']:.1f}min "
          f"→ n_peaks={amp_ourlc.size}  amp CV={_cv(amp_ourlc):.3f}  IPI CV={_ipi_cv(ipi_ourlc):.3f}")

    # ── 3) Walker (2010) human-scaled delay limit cycle: fit p1,p2,p4,p5,p6,tau ──
    walker_bounds = dict(p1=(2.0, 80.0), p2=(2.0, 150.0), p4=(0.005, 1.0),
                         p5=(0.0, 1.0), p6=(0.3, 10.0), tau_min=(5.0, 30.0))
    if cached_fits is not None:
        w_fit = cached_fits["walker_fit"]
        print(f"[walker] cached fit tau={w_fit['tau_min']:.1f}min")
    else:
        print("[walker] fitting human-scaled Walker (2010) LC (p1,p2,p4,p5,p6,tau; p3=0.75)...")
        w_fit = _fit_walker(cp, data_amp=data_amp_mean, data_ipi=data_ipi_mean,
                            dt_min=args.walker_dt_min, prom_sigma=args.prom_sigma,
                            min_distance_min=args.min_distance_min, bounds=walker_bounds,
                            maxiter=args.walker_maxiter)
        fit_cache_path.write_text(json.dumps(
            {"key": fit_key, "our_fit": our_fit, "walker_fit": w_fit}, indent=2))
        print(f"[cache] wrote fitted params → {fit_cache_path}")
    amp_walker, ipi_walker, walker_rows = _walker_peaks(
        cp, w_fit, dt_min=args.walker_dt_min, n_days=args.lc_n_days,
        prom_sigma=args.prom_sigma, min_distance_min=args.min_distance_min)
    print(f"[walker] fit p1={w_fit['p1']:.2f} p2={w_fit['p2']:.2f} p4={w_fit['p4']:.3f} "
          f"p5={w_fit['p5']:.3f} p6={w_fit['p6']:.2f} tau={w_fit['tau_min']:.1f}min  "
          f"loss={w_fit['loss']:.3f} → n_peaks={amp_walker.size}  amp CV={_cv(amp_walker):.3f}  "
          f"IPI CV={_ipi_cv(ipi_walker):.3f}")

    # ── dip tests (amplitude multimodality) at matched n ────────────────────────
    pulsed = [a for a in (amp_data, amp_noise, amp_walker) if a.size >= 5]
    n_match = int(min(a.size for a in pulsed)) if pulsed else 0
    dip_data = _dip_matched(amp_data, n_match=n_match) if n_match else (float("nan"), float("nan"))
    dip_noise = _dip_matched(amp_noise, n_match=n_match) if n_match else (float("nan"), float("nan"))
    dip_walker = _dip_matched(amp_walker, n_match=n_match) if n_match else (float("nan"), float("nan"))
    dip_ourlc = _dip_matched(amp_ourlc, n_match=min(n_match, amp_ourlc.size)) \
        if amp_ourlc.size >= 5 else (float("nan"), float("nan"))

    # ── assemble conditions ─────────────────────────────────────────────────────
    conditions = [
        dict(key="data", color=DATA_COLOR, amp=amp_data, ipi=ipi_data,
             rows=_data_rows(args.peaks_csv), dip=dip_data),
        dict(key="noise", color=NOISE_COLOR, amp=amp_noise, ipi=ipi_noise,
             rows=noise_rows, dip=dip_noise),
        dict(key="our_lc", color=OUR_LC_COLOR, amp=amp_ourlc, ipi=ipi_ourlc,
             rows=ourlc_rows, dip=dip_ourlc),
        dict(key="walker", color=WALKER_COLOR, amp=amp_walker, ipi=ipi_walker,
             rows=walker_rows, dip=dip_walker),
    ]
    _apply_labels(conditions, cp, our_fit, w_fit)

    # ── summary.json ────────────────────────────────────────────────────────────
    def _cond_stats(c):
        return dict(n_peaks=int(c["amp"].size),
                    amp_mean=float(np.mean(c["amp"])) if c["amp"].size else float("nan"),
                    amp_cv=_cv(c["amp"]),
                    ipi_mean=float(np.mean(c["ipi"][np.isfinite(c["ipi"])])) if np.isfinite(c["ipi"]).any() else float("nan"),
                    ipi_cv=_ipi_cv(c["ipi"]),
                    rayleigh_ks_p=_rayleigh_ks_p(c["amp"]),
                    dip_p_median=c["dip"][0], dip_frac_multimodal=c["dip"][1])
    summary = dict(
        circadian=cp,
        human_scaling=dict(acth_half_life_min=ACTH_HALF_LIFE_MIN,
                           cort_half_life_min=CORT_HALF_LIFE_MIN,
                           walker_p3=p3_from_half_lives(CORT_HALF_LIFE_MIN, ACTH_HALF_LIFE_MIN)),
        our_lc=dict(kgr=our_fit["kgr"], tau_min=our_fit["tau_min"], fit_loss=our_fit["loss"],
                    kgr5_max_resid_frac=kgr5["max_resid_frac"], kgr5_max_npeaks=kgr5["max_npeaks"],
                    min_resid_frac=MIN_RESID_FRAC, biological_kgr=5.0,
                    note="free kgr+tau; limit cycle requires low kgr (strong feedback). "
                         "At the fitted biological kgr=5 the model is a stable focus (no limit cycle)."),
        walker=dict(**{k: w_fit[k] for k in WALKER_KEYS}, loss=w_fit["loss"],
                    p3=p3_from_half_lives(CORT_HALF_LIFE_MIN, ACTH_HALF_LIFE_MIN),
                    rodent_defaults=dict(p2=WALKER_P2_RODENT, p4=WALKER_P4_RODENT,
                                         p5=WALKER_P5_RODENT, p6=WALKER_P6_RODENT)),
        dip_test=dict(n_match=n_match, note="Hartigan dip; median p over 300 subsamples; "
                      "frac with p<0.05 = multimodal"),
        conditions={c["key"]: _cond_stats(c) for c in conditions},
    )
    (args.out / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2))
    amp_frames = [pd.DataFrame({"condition": c["key"], "amp_prev_dip_z": c["amp"]})
                  for c in conditions if c["amp"].size]
    pd.concat(amp_frames, ignore_index=True).to_csv(
        args.out / "artifacts" / "amplitudes.csv", index=False)

    # plot payload for instant --replot (title/style tweaks without re-simulating)
    with open(args.out / "artifacts" / "plot_cache.pkl", "wb") as fh:
        pickle.dump(dict(cp=cp, conditions=conditions, n_match=n_match,
                         our_fit=our_fit, w_fit=w_fit), fh)

    print("\n[summary]")
    for c in conditions:
        s = _cond_stats(c)
        print(f"  {c['key']:>8}: n={s['n_peaks']:>5} ampCV={s['amp_cv']:.3f} "
              f"IPIcv={s['ipi_cv']:.3f} dip_p={s['dip_p_median']:.2g} "
              f"({'multimodal' if (s['dip_frac_multimodal'] or 0) >= 0.5 else 'unimodal'})")

    # ── figures ─────────────────────────────────────────────────────────────────
    _figure_bottomline(args, cp, conditions, n_match, our_fit, w_fit,
                       noise_model, noise_drive)
    _figure_metrics(args, conditions)
    _figure_per_bin(args, conditions)
    print(f"[done] {args.out}")


def _dip_str(dip):
    p, frac = dip
    if not np.isfinite(p):
        return "n/a"
    return "multimodal" if (frac or 0) >= 0.5 else "unimodal"


# Display labels live in one place so --replot reflects label edits without
# re-simulating. The noise-driven and delay-LC conditions are both the Karin
# (2020) HPA model — "our model" in the manuscript is that model + noise.
def _condition_label(key, cp, our_fit, w_fit):
    if key == "data":
        return "Data"
    if key == "noise":
        return "Karin 2020, noise only\n($k_{GR}$=5, τ=0, ε=%.2f)" % cp["epsilon"]
    if key == "our_lc":
        return ("Karin 2020, delay only\n($k_{GR}$=%.1f, τ=%.0f min)"
                % (our_fit["kgr"], our_fit["tau_min"]))
    if key == "walker":
        return "Walker 2010 delay LC\n(τ=%.0f min)" % w_fit["tau_min"]
    return key


# Short tags for the in-panel dip-test box (kept compact to fit the text box).
_KEY_SHORT = {"data": "data", "noise": "Karin noise",
              "our_lc": "Karin delay", "walker": "Walker"}


def _apply_labels(conditions, cp, our_fit, w_fit):
    for c in conditions:
        c["label"] = _condition_label(c["key"], cp, our_fit, w_fit)


def _trajectory_traces(args, cp, our_fit, w_fit, noise_model, noise_drive):
    """The four example z-scored cortisol traces.

    All conditions are driven by the SAME circadian input, but the deterministic
    delay limit cycles respond with their own circadian phase lag (the self-
    oscillating LC's amplitude envelope peaks several hours after the drive). For
    this purely illustrative panel each *model* trace is therefore rolled in
    time-of-day so its circadian envelope acrophase lines up with the data's; the
    data trace is the unshifted reference. This is display-only — the quantitative
    panels (IPI CV, amplitude multimodality) are circadian-phase-independent.
    """
    td, rd, acro_data = _data_example_trace(args.variant, args.data_example_id,
                                            dataset=args.data_example_dataset)

    def _align(t_min, x, color, label):
        """Slice a seamless 24h window whose circadian acrophase sits at acro_data."""
        t_min = np.asarray(t_min, float); x = np.asarray(x, float)
        dt = float(np.median(np.diff(t_min)))
        n_day = int(round(1440.0 / dt))
        acro = _circadian_acrophase_min(t_min, x)
        shift = float((acro - acro_data) % 1440.0)
        i0 = int(np.searchsorted(t_min, shift, side="left"))
        i0 = max(0, min(i0, len(t_min) - n_day))
        sl = slice(i0, i0 + n_day)
        tw = t_min[sl] - t_min[i0]; xw = x[sl]
        z = (xw - xw.mean()) / xw.std() if xw.std() > 1e-9 else np.zeros_like(xw)
        print(f"[traj-align] {label.splitlines()[0]}: envelope acrophase "
              f"{acro / 60:.2f}h → shifted {shift / 60:.2f}h to data acrophase "
              f"{acro_data / 60:.2f}h")
        return (tw / 60.0, z, color, label)

    traces = [(td, rd, DATA_COLOR, "Data (Young et al. 2004)")]
    # Simulate 2 days so the aligned 24h window is a contiguous (seamless) slice.
    tn, xn = _sim_window(noise_model, noise_drive, dt_min=args.dt_min,
                         warmup_min=1440.0, duration_min=2880.0, seed=args.seed)
    traces.append(_align(tn, xn, NOISE_COLOR,
                         "Karin 2020, noise only ($k_{GR}$=5, τ=0)"))
    to, xo = _our_lc_window(cp, our_fit["kgr"], our_fit["tau_min"],
                            dt_min=args.dt_min, n_days=2)
    traces.append(_align(to, xo, OUR_LC_COLOR,
                         f"Karin 2020, delay only ($k_{{GR}}$={our_fit['kgr']:.1f}, τ={our_fit['tau_min']:.0f} min)"))
    tw, xw = _walker_window(cp, w_fit, dt_min=args.walker_dt_min, n_days=2)
    traces.append(_align(tw, xw, WALKER_COLOR,
                         f"Walker 2010 delay LC (τ={w_fit['tau_min']:.0f} min)"))
    return traces


def _figure_trajectory_only(args, cp, our_fit, w_fit, noise_model, noise_drive):
    """Standalone example-trajectory comparison (the main figure's Panel A, on its own)."""
    setup_nature_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    traces = _trajectory_traces(args, cp, our_fit, w_fit, noise_model, noise_drive)
    offset = 9.0
    for i, (t, rz, c, lab) in enumerate(traces):
        band = (len(traces) - 1 - i) * offset
        ax.plot(t, rz + band, color=c, lw=1.2)
        ax.text(0.2, band + 3.6, lab, color=c, fontsize=10.5, va="bottom", ha="left")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.set_ylim(-5.0, (len(traces) - 1) * offset + 8.5)
    ax.set_yticks([])
    ax.set_xlabel("Time of day (hours)", fontsize=12)
    ax.set_ylabel("Cortisol (z-scored, offset)", fontsize=12)
    ax.set_title(f"Example cortisol dynamics (shared circadian input; model traces "
                 f"aligned to data acrophase) — drive noise ε = {cp['epsilon']:.2f}",
                 fontsize=11, loc="left")
    apply_paper_style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"trajectory_comparison.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main manuscript figure: traces + scalar bars (with IPI CV) + amplitude dist ──

def _figure_bottomline(args, cp, conditions, n_match, our_fit, w_fit,
                       noise_model, noise_drive):
    import matplotlib.gridspec as gridspec
    setup_nature_style()
    fig = plt.figure(figsize=(17.0, 5.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.15, 0.85, 1.05],
                           wspace=0.24, left=0.045, right=0.985,
                           top=0.90, bottom=0.14)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Panel A: one example z-scored cortisol trace per condition. All share the
    # same circadian input; each model trace is phase-aligned to the data's
    # circadian acrophase for display (see _trajectory_traces — display-only).
    traces = _trajectory_traces(args, cp, our_fit, w_fit, noise_model, noise_drive)

    offset = 9.0
    for i, (t, rz, c, lab) in enumerate(traces):
        band = (len(traces) - 1 - i) * offset
        axA.plot(t, rz + band, color=c, lw=1.2)
        axA.text(0.2, band + 3.6, lab, color=c, fontsize=10.5, va="bottom", ha="left")
    axA.set_xlim(0, 24); axA.set_xticks(range(0, 25, 4))
    axA.set_ylim(-5.0, (len(traces) - 1) * offset + 8.5)
    axA.set_yticks([])
    axA.set_xlabel("Time of day (hours)", fontsize=12)
    axA.set_ylabel("Cortisol (z-scored, offset)", fontsize=12)
    axA.set_title("A  Example cortisol dynamics (shared circadian input; aligned to data)",
                  fontsize=11.5, loc="left")
    apply_paper_style(axA)

    # Panel B: scalar statistics as % of data, INCLUDING IPI CV.
    metrics = [("amp_mean", "Mean\namplitude", lambda v: np.mean(v)),
               ("ipi_mean", "Mean\nIPI", None),
               ("amp_cv", "Amplitude\nCV", _cv),
               ("ipi_cv", "IPI\nCV", _ipi_cv)]
    data_c = conditions[0]
    def metric_val(c, key):
        if key == "amp_mean":
            return float(np.mean(c["amp"])) if c["amp"].size else 0.0
        if key == "amp_cv":
            return _cv(c["amp"]) if c["amp"].size > 1 else 0.0
        if key == "ipi_mean":
            ip = c["ipi"][np.isfinite(c["ipi"]) & (c["ipi"] > 0)]
            return float(np.mean(ip)) if ip.size else 0.0
        if key == "ipi_cv":
            return _ipi_cv(c["ipi"]) if np.isfinite(c["ipi"]).sum() > 1 else 0.0
        return 0.0
    xb = np.arange(len(metrics)); w = 0.20
    for gi, c in enumerate(conditions):
        dvals = [metric_val(data_c, k) for k, _, _ in metrics]
        vals = [100.0 * metric_val(c, k) / dv if dv > 0 else 0.0
                for (k, _, _), dv in zip(metrics, dvals)]
        off = (gi - (len(conditions) - 1) / 2) * w
        axB.bar(xb + off, vals, width=w, color=c["color"], alpha=0.85,
                edgecolor="white", linewidth=0.6)
    axB.axhline(100.0, color="0.5", ls="--", lw=1.0, zorder=1)
    axB.set_xticks(xb); axB.set_xticklabels([lab for _, lab, _ in metrics], fontsize=10)
    axB.set_ylabel("% of data value", fontsize=12)
    axB.set_ylim(0, 160)
    axB.set_title("B  IPI CV exposes the metronomic limit cycle", fontsize=12.5, loc="left")
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor=c["color"], label=c["label"].split("\n")[0])
                        for c in conditions],
               frameon=False, fontsize=8.0, loc="upper left", ncol=2,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    apply_paper_style(axB)

    # Panel C: amplitude distribution + dip test.
    amax = max((c["amp"].max() for c in conditions if c["amp"].size), default=6.0)
    bins = np.linspace(0, amax + 0.5, 30)
    axC.hist(data_c["amp"], bins=bins, density=True, color=DATA_COLOR, alpha=0.30,
             edgecolor="white", linewidth=0.5,
             label=f"Data ({_dip_str(data_c['dip'])})")
    for c in conditions[1:]:
        if c["amp"].size < 5:
            continue
        axC.hist(c["amp"], bins=bins, density=True, histtype="step", color=c["color"],
                 linewidth=2.0, label=f"{c['label'].splitlines()[0]} ({_dip_str(c['dip'])})")
    dip_lines = "\n".join(
        f"  {_KEY_SHORT.get(c['key'], c['key'])}: p={c['dip'][0]:.2g} → {_dip_str(c['dip'])}"
        for c in conditions if np.isfinite(c['dip'][0]))
    axC.text(0.97, 0.97, f"Hartigan dip test (n={n_match}):\n{dip_lines}",
             transform=axC.transAxes, ha="right", va="top", fontsize=8.5,
             bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3})
    axC.set_xlabel("Peak − previous-trough amplitude (within-series Z-score)", fontsize=12)
    axC.set_ylabel("Density", fontsize=12)
    axC.set_title("C  Limit-cycle amplitudes are multimodal (mode-locked)",
                  fontsize=12.5, loc="left")
    axC.legend(frameon=False, fontsize=9.0, loc="center right")
    apply_paper_style(axC)

    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"limit_cycle_vs_noise.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── supporting: 4-panel mean & CV metrics with bootstrap CIs ─────────────────────

def _figure_metrics(args, conditions):
    setup_nature_style()
    mean_fn = lambda v: float(np.mean(v))
    cv_fn = lambda v: float(np.std(v, ddof=1) / np.mean(v))
    panels = [
        ("A  Peak amplitude — mean", "Amplitude mean (Z)", "amp", mean_fn),
        ("B  Peak amplitude — CV", "Amplitude CV", "amp", cv_fn),
        ("C  Inter-peak interval — mean", "IPI mean (min)", "ipi", mean_fn),
        ("D  Inter-peak interval — CV", "IPI CV", "ipi", cv_fn),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    rows = []
    for ax, (title, ylab, field, fn) in zip(axes.ravel(), panels):
        for j, c in enumerate(conditions):
            arr = c[field]
            if field == "ipi":
                arr = arr[np.isfinite(arr) & (arr > 0)]
            val, lo, hi = _boot_ci(arr, fn, seed=j)
            if not np.isfinite(val):
                continue
            ax.bar(j, val, width=0.66, color=c["color"], alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=2)
            ax.errorbar(j, val, yerr=[[max(val - lo, 0)], [max(hi - val, 0)]], fmt="none",
                        ecolor="0.25", elinewidth=1.1, capsize=3.5, zorder=3)
            ax.text(j, hi if np.isfinite(hi) else val,
                    f"{val:.2f}" if val < 10 else f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9, color="0.2")
            rows.append({"metric": title[3:], "condition": c["key"], "value": val,
                         "ci_lo": lo, "ci_hi": hi})
        d = conditions[0][field]
        d = d[np.isfinite(d) & (d > 0)]
        if d.size > 1:
            ax.axhline(fn(d), color=DATA_COLOR, ls="--", lw=1.0, alpha=0.6, zorder=1)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c["label"].splitlines()[0] for c in conditions],
                           fontsize=8.5, rotation=20, ha="right")
        ax.set_ylabel(ylab, fontsize=11.5)
        ax.set_title(title, fontsize=12, loc="left")
        apply_paper_style(ax)
    fig.suptitle("Peak amplitude and IPI — mean & CV vs data "
                 "(dashed = data; bars = 95% bootstrap CI)", fontsize=13, y=1.0)
    fig.tight_layout()
    pd.DataFrame(rows).to_csv(args.out / "artifacts" / "metrics_comparison.csv", index=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"metrics_comparison.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _per_series_bin_stat(df, value_col, stat, min_peaks):
    if df is None or df.empty:
        return pd.DataFrame(columns=["uid", "bin", "value"])
    d = df.copy()
    d["bin"] = pd.cut(d["tod_min"] % 1440.0, bins=BIN_EDGES, labels=BIN_LABELS,
                      right=True, include_lowest=True).astype(str)
    out = []
    for (uid, b), g in d.groupby(["uid", "bin"], observed=True):
        v = g[value_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if value_col == "amp":
            v = v[v > 0]
        if len(v) < min_peaks:
            continue
        if stat == "mean":
            val = float(np.mean(v))
        else:
            m = float(np.mean(v))
            val = float(np.std(v, ddof=1) / m) if m > 0 else np.nan
        out.append({"uid": uid, "bin": str(b), "value": val})
    return pd.DataFrame(out)


def _box_groups(ax, longs, colors, *, ylabel, title, ylim=None):
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
        means = [float(np.mean(v)) if len(v) else np.nan for v in by_bin]
        ax.plot(pos + off, means, color=c, lw=1.4, zorder=5, marker="o", ms=3.0,
                markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.0)
    ax.set_xticks(pos); ax.set_xticklabels(BIN_LABELS, fontsize=9.5, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(BIN_LABELS) - 0.4)
    ax.set_xlabel("Time of day (h)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, loc="left")
    if ylim is not None:
        ax.set_ylim(*ylim)
    apply_paper_style(ax)


def _figure_per_bin(args, conditions):
    setup_nature_style()
    colors = [c["color"] for c in conditions]
    tables = [c["rows"] for c in conditions]

    def longs(col, stat, mp):
        return [_per_series_bin_stat(t, col, stat, mp) for t in tables]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    _box_groups(axes[0, 0], longs("amp", "mean", 1), colors,
                ylabel="Amplitude mean (Z)", title="A  Amplitude mean")
    _box_groups(axes[0, 1], longs("ipi", "mean", 1), colors,
                ylabel="IPI mean (min)", title="B  Inter-peak interval mean")
    _box_groups(axes[1, 0], longs("amp", "cv", 2), colors,
                ylabel="Amplitude CV", title="C  Amplitude CV")
    _box_groups(axes[1, 1], longs("ipi", "cv", 2), colors,
                ylabel="IPI CV", title="D  Inter-peak interval CV")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c["color"], alpha=0.55, edgecolor=c["color"],
                              label=c["label"].splitlines()[0]) for c in conditions],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Per-time-of-day pulse statistics: data vs noise-driven vs delay limit cycles",
                 fontsize=12.5, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"per_bin_comparison.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
