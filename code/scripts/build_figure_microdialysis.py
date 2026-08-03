"""Microdialysis (interstitial) cortisol vs the 4-compartment model (paper Fig 8).

Per-time-of-day-bin peak statistics in the EXACT style of the main-text box-plot
figure (paper Fig 4 / repo ``figure_3``), comparing:

* **MD data** (blue) -- subcutaneous microdialysis cortisol, 213 participants,
  20-min sampling (``habs_microdialysis_cortisol`` ``shifted``; the tissue arm of
  Upton et al. 2023).
* **Model interstitial x4** (orange) -- the canonical noise-driven model
  (``manuscript_figures_final/figure_3/fit_cortisol_drive_noise_v15_acth20_cort15_eps15.yaml``)
  with a passive blood-to-interstitial diffusion compartment, dx4/dt = k (x3 - x4).
  k is fitted independently to the 7 paired serum/microdialysis recordings of
  Upton et al. on the ultradian band (k = 0.027/min, t1/2 = 26 min); x4 is
  averaged over each 20-min collection interval (microdialysis-style integration)
  before peak extraction.

Reuses the canonical box-plot helpers from ``build_figure3_boxplots`` so every plot
attribute (colours, box style, trend lines, Rayleigh line, IPI cutoff) is identical.

Usage:
  PYTHONPATH=src python scripts/build_figureSI3_microdialysis.py \
      --out experiments/runs/figureSI3_microdialysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))      # for build_figure3_boxplots
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.build_figure3_boxplots as B  # noqa: E402  (canonical box-plot helpers)
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from hpa_model.model.three_state_gr_delay import (  # noqa: E402
    ThreeStateGRDelayModel, build_drive,
)
from hpa_model.plotting import apply_paper_style, setup_nature_style  # noqa: E402
from hpa_model.simulate.engine import simulate_trajectory_fit_arrays  # noqa: E402

CANONICAL_CFG = (PROJECT_ROOT / "manuscript_figures_final" / "figure_3"
                 / "fit_cortisol_drive_noise_v15_acth20_cort15_eps15.yaml")
UPTON = PROJECT_ROOT / "data" / "raw_data_input" / "Upton et al. (2023) blood.csv"
MD_CSV = (PROJECT_ROOT / "data" / "catalog" / "datasets"
          / "habs_microdialysis_cortisol" / "shifted" / "data_shifted.csv")

PROM, MIN_DIST_MIN, IPI_CUTOFF, RESAMPLE_DT = 0.5, 60.0, 240.0, 20.0


def _zscore(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def _decircadian(t, x):
    """Remove a least-squares 24 h + 12 h harmonic baseline, leaving the ultradian
    residual (in the signal's own units)."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    w24, w12 = 2.0 * np.pi / 1440.0, 2.0 * np.pi / 720.0
    A = np.column_stack([np.ones_like(t), np.cos(w24 * t), np.sin(w24 * t),
                         np.cos(w12 * t), np.sin(w12 * t)])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _lowpass(t, x, k):
    """Exact first-order low-pass: dx4/dt = k (x3 - x4)."""
    out = np.empty_like(x, float); out[0] = x[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        out[i] = x[i - 1] + (out[i - 1] - x[i - 1]) * np.exp(-k * dt)
    return out


def _block_average(t, y, dt_out):
    """20-min integral: mean of y over each collection window (microdialysis-style)."""
    n = int(round(dt_out / np.median(np.diff(t))))
    m = (len(y) // n) * n
    yb = np.asarray(y[:m], float).reshape(-1, n).mean(axis=1)
    tb = np.asarray(t[:m], float).reshape(-1, n).mean(axis=1)
    return tb, yb


def fit_diffusion_k_on_upton() -> float:
    """Least-squares diffusion rate k on the 7 paired serum/microdialysis
    recordings, fit on the ULTRADIAN band only.

    Both signals are de-circadianized (24 h + 12 h harmonic removed) before the
    fit, so k measures how the diffusion compartment transmits ultradian pulses
    rather than the slow circadian phase lag. This band-limited fit gives t1/2
    about 26 min; fitting the full waveform gives an essentially identical t1/2
    about 27 min, so the result does not depend on the band-limiting choice. A
    per-recording scale + offset is regressed out so the unit difference between
    serum and microdialysis does not bias k.
    """
    df = pd.read_csv(UPTON)
    tt = pd.to_datetime(df["Time"], format="%H:%M:%S")
    df["tmin"] = tt.dt.hour * 60.0 + tt.dt.minute
    series = []
    for _, g in df.groupby("ID"):
        b = g[g["Cortisol"].notna()][["tmin", "Cortisol"]].drop_duplicates("tmin").sort_values("tmin")
        m = g[g["mCortisol"].notna()][["tmin", "mCortisol"]].drop_duplicates("tmin").sort_values("tmin")
        if len(b) < 6 or len(m) < 6:
            continue
        tb = b["tmin"].to_numpy(float); tm = m["tmin"].to_numpy(float)
        series.append((tb, _decircadian(tb, b["Cortisol"].to_numpy(float)),
                       tm, _decircadian(tm, m["mCortisol"].to_numpy(float))))

    def resid(p):
        k = np.exp(p[0])
        parts = []
        for tb, rb, tm, rm in series:
            pred = _lowpass(tm, np.interp(tm, tb, rb), k)
            # regress out a per-recording scale + offset so they do not bias k
            A = np.column_stack([pred, np.ones_like(pred)])
            c, *_ = np.linalg.lstsq(A, rm, rcond=None)
            parts.append(A @ c - rm)
        return np.concatenate(parts)

    r = least_squares(resid, x0=[np.log(1 / 30)],
                      bounds=([np.log(1e-3)], [np.log(2.0)]))
    return float(np.exp(r.x[0]))


def _peaks(t, y, uid) -> list[dict]:
    rz, _ = B._baseline_subtract_residual_z(t, y)
    pt, pa = B._detect_prev_dip_amps(t, rz, prom_sigma=PROM, min_distance_min=MIN_DIST_MIN)
    return B._peaks_to_rows(pt, pa, uid=uid)


def md_data_peaks() -> pd.DataFrame:
    md = pd.read_csv(MD_CSV)
    rows = []
    for mid, g in md.groupby("MasterID"):
        g = g.sort_values("time_min")
        t = g["time_min"].to_numpy(float); y = g["Cortisol"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(y); t, y = t[ok], y[ok]
        if len(y) < 10:
            continue
        rows += _peaks(t, y, uid=f"md:{int(mid)}")
    return pd.DataFrame(rows)


def _build_model_and_drive(cfg: dict, eps: float):
    mp = cfg["model"]["params"]
    model = ThreeStateGRDelayModel(
        a1=float(mp["a1"]), a2=float(mp["a2"]), a3=float(mp["a3"]),
        b1=float(mp["b1"]), b2=float(mp["b2"]), b3=float(mp["b3"]),
        kgr=float(mp["kgr"]), tau_min=float(mp.get("tau_min", 0.0)),
        x3_floor=float(mp.get("x3_floor", 0.01)), hill_coeff=float(mp.get("hill_coeff", 3.0)),
        initial_state=tuple(float(x) for x in mp["initial_state"]),
    )
    dp = {k_: v for k_, v in cfg["drive"]["params"].items() if k_ not in ("dataset", "series_id")}
    dp["epsilon"] = eps
    drive = build_drive("two_harmonic_noise", dp)
    sv = cfg["solver"]
    return (model, drive, float(sv["dt_min"]), float(sv["warmup_min"]),
            float(sv["duration_min"]))


def _sim_x4(model, drive, dt_min, warmup, duration, k, seed):
    """One realization: simulate x3, diffuse to interstitial x4, block-average."""
    a = simulate_trajectory_fit_arrays(
        model, drive, dt_min=dt_min, warmup_min=warmup, duration_min=duration,
        seed=seed, noise_locations=[], noise_epsilons={}, noise_form="lognormal",
    )
    t, x3 = a["time_min"], a["x3"]
    x4 = _lowpass(t, x3, k)
    return _block_average(t, x4, RESAMPLE_DT)  # (tb, x4b) at 20-min resolution


def model_x4_peaks(cfg: dict, eps: float, k: float, n_sims: int, seed: int) -> pd.DataFrame:
    model, drive, dt_min, warmup, duration = _build_model_and_drive(cfg, eps)
    rows = []
    for s in range(n_sims):
        tb, x4b = _sim_x4(model, drive, dt_min, warmup, duration, k, seed + s)
        rows += _peaks(tb, x4b, uid=f"model:{s}")
    return pd.DataFrame(rows)


def md_example_trace(master_id: int):
    """One MD subject's interstitial cortisol, z-scored, over time of day (h)."""
    md = pd.read_csv(MD_CSV)
    g = md[md["MasterID"] == master_id].sort_values("time_min")
    t = g["time_min"].to_numpy(float); y = g["Cortisol"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(y)
    return t[ok] / 60.0, _zscore(y[ok])


def model_x4_example_traces(cfg: dict, eps: float, k: float, seeds) -> list:
    """Model interstitial x4 example realizations, z-scored, over time of day (h)."""
    model, drive, dt_min, warmup, duration = _build_model_and_drive(cfg, eps)
    out = []
    for sd in seeds:
        tb, x4b = _sim_x4(model, drive, dt_min, warmup, duration, k, sd)
        out.append((tb / 60.0, _zscore(x4b)))
    return out


def _apply_ipi_cutoff(dfp: pd.DataFrame, tag: str) -> None:
    ipi = dfp["ipi"].to_numpy(float); finite = np.isfinite(ipi)
    over = finite & (ipi > IPI_CUTOFF)
    dfp.loc[over, "ipi"] = np.nan
    print(f"[{tag}] IPI cutoff {IPI_CUTOFF:g} min: dropped {int(over.sum())}/{int(finite.sum())} "
          f"({over.sum()/max(finite.sum(),1):.1%}) as unresolved merges.")


def render(data_peaks, model_peaks, *, eps, k, out_dir: Path,
           data_trace, model_traces, data_example_id, trend_line: bool = False) -> Path:
    setup_nature_style()
    fig = plt.figure(figsize=(11.0, 11.0))
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[0.8, 1.0, 1.0],
                           hspace=0.42, wspace=0.9,
                           left=0.07, right=0.985, top=0.93, bottom=0.06)

    # ── Row 0 — example interstitial traces: A = MD data, B/C = model x4 ───────
    traces = [("A", f"Microdialysis data (subject {data_example_id})",
               B.DATA_COLOR, data_trace)]
    for j, mt in enumerate(model_traces[:1]):
        traces.append(("B", "Model interstitial $x_4$",
                       B.MODEL_COLOR, mt))
    all_z = np.concatenate([z for _, _, _, (_, z) in traces if len(z)])
    ymax = float(np.nanmax(np.abs(all_z))) * 1.12 if len(all_z) else 2.6
    traj_slots = [gs[0, 0:3], gs[0, 3:6]]
    traj_axes = []
    sharey_ax = None
    for col, (letter, title, color, (th, z)) in enumerate(traces):
        ax = fig.add_subplot(traj_slots[col], sharey=sharey_ax)
        if sharey_ax is None:
            sharey_ax = ax
        traj_axes.append(ax)
        ax.plot(th, z, color=color, lw=1.1)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6))
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("Time of day (h)", fontsize=10.5)
        if col == 0:
            ax.set_ylabel("Cortisol (z-score)", fontsize=10.5)
        ax.set_title(f"{letter}  {title}", fontsize=10.0, loc="left", color=color)
        apply_paper_style(ax)

    # ── Rows 1-2 — per-bin pulse statistics (unchanged 2x2, relabelled D-G) ───
    stat_slots = [gs[1, 0:3], gs[1, 3:6], gs[2, 0:3], gs[2, 3:6]]
    panels = [
        ("amp", "mean", "Amplitude mean (Z-score)", "D  Amplitude mean", stat_slots[0], 1),
        ("ipi", "mean", "Inter-peak interval (IPI) mean (min)",
                         "E  Inter-peak interval (IPI) mean", stat_slots[1], 1),
        ("amp", "cv",   "Amplitude CV",             "F  Amplitude CV",   stat_slots[2], 2),
        ("ipi", "cv",   "Inter-peak interval (IPI) CV",
                         "G  Inter-peak interval (IPI) CV",   stat_slots[3], 2),
    ]
    rayleigh_cv = float(np.sqrt((4.0 - np.pi) / np.pi))
    longs = {(vc, st): (B._per_uid_bin_stat(data_peaks, vc, st, mp),
                        B._per_uid_bin_stat(model_peaks, vc, st, mp))
             for vc, st, _, _, _, mp in panels}
    cv_all = np.concatenate([
        np.concatenate([d["value"].to_numpy(float), m["value"].to_numpy(float)])
        for (vc, st), (d, m) in longs.items() if st == "cv"])
    cv_all = cv_all[np.isfinite(cv_all)]
    cv_ylim = (0.0, float(np.nanquantile(cv_all, 0.97)) * 1.15) if len(cv_all) else None

    stat_axes = []
    for vc, st, ylabel, title, slot, mp in panels:
        ax = fig.add_subplot(slot)
        stat_axes.append(ax)
        d_long, m_long = longs[(vc, st)]
        B._box_pair(ax, d_long, m_long, ylabel=ylabel, title=title,
                    connect_trend=trend_line, ylim=cv_ylim if st == "cv" else None)
        if vc == "amp" and st == "cv":
            ax.axhline(rayleigh_cv, color="#c0392b", linestyle="--", linewidth=1.0, zorder=4,
                       label=f"Theoretical CV = {rayleigh_cv:.3f}")
            ax.legend(loc="upper right", fontsize=9, frameon=False)

    # Data/Model legend in the gap between the trace row and the stats row.
    data_h = plt.Rectangle((0, 0), 1, 1, fc=B.DATA_COLOR, alpha=0.55, ec=B.DATA_COLOR)
    model_h = plt.Rectangle((0, 0), 1, 1, fc=B.MODEL_COLOR, alpha=0.55, ec=B.MODEL_COLOR)
    traj_bottom = min(a.get_position().y0 for a in traj_axes)
    stats_top = max(stat_axes[0].get_position().y1, stat_axes[1].get_position().y1)
    y_legend = (traj_bottom + stats_top) / 2.0
    fig.legend([data_h, model_h],
               [f"MD data (n={data_peaks['series_uid'].nunique()} subjects)",
                f"4-comp model $x_4$ (n={model_peaks['series_uid'].nunique()} simulations)"],
               loc="center", bbox_to_anchor=(0.5, y_legend), ncol=2, frameon=False, fontsize=11)

    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / "figure_microdialysis.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_dir / "figure_microdialysis.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CANONICAL_CFG,
                    help="Canonical model config (model params + two-harmonic drive).")
    ap.add_argument("--eps", type=float, default=None,
                    help="Drive-noise amplitude; default = config value (canonical 1.5).")
    ap.add_argument("--n-sims", type=int, default=213,
                    help="Number of model realizations (default matches MD cohort).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-example-id", type=int, default=71,
                    help="MD MasterID for the example data trace (panel A); "
                         "default 71 has the cohort-median pulse count.")
    ap.add_argument("--example-seeds", type=int, nargs=2, default=(101, 202),
                    help="Two seeds for the model interstitial example traces (panels B, C).")
    ap.add_argument("--trend-line", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Draw the line (and mean markers) connecting per-bin means "
                         "in each panel. Default off (boxplots only); pass "
                         "--trend-line to re-enable.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/figure_microdialysis")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    eps = args.eps if args.eps is not None else float(cfg["drive"]["params"].get("epsilon", 1.5))
    k = fit_diffusion_k_on_upton()
    print(f"Diffusion k (Upton paired fit) = {k:.4f}/min  (t1/2 = {np.log(2)/k:.1f} min);  eps = {eps}")

    data_peaks = md_data_peaks()
    model_peaks = model_x4_peaks(cfg, eps, k, args.n_sims, args.seed)
    print(f"[data]  {len(data_peaks)} peaks / {data_peaks['series_uid'].nunique()} subjects")
    print(f"[model] {len(model_peaks)} peaks / {model_peaks['series_uid'].nunique()} sims")
    _apply_ipi_cutoff(data_peaks, "data")
    _apply_ipi_cutoff(model_peaks, "model")

    art = args.out / "artifacts"; art.mkdir(parents=True, exist_ok=True)
    data_peaks.to_csv(art / "data_peaks.csv", index=False)
    model_peaks.to_csv(art / "model_peaks.csv", index=False)
    pd.DataFrame([{"diffusion_k_per_min": k, "diffusion_t_half_min": float(np.log(2) / k),
                   "epsilon": eps, "n_sims": args.n_sims, "config": str(args.config),
                   "data_example_id": args.data_example_id,
                   "example_seeds": str(tuple(args.example_seeds))}]
                 ).to_csv(art / "params.csv", index=False)

    data_trace = md_example_trace(args.data_example_id)
    model_traces = model_x4_example_traces(cfg, eps, k, args.example_seeds)

    png = render(data_peaks, model_peaks, eps=eps, k=k, out_dir=args.out,
                 data_trace=data_trace, model_traces=model_traces,
                 data_example_id=args.data_example_id, trend_line=args.trend_line)
    print(f"PNG: {png}")
    print(f"PDF: {png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
