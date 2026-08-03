"""Fig 3 (combined) — model example trajectories over per-bin pulse statistics.

Merges the former Fig 3 (example trajectories) and Fig 4 (model-vs-data per-bin
boxplots) into a single figure, in the same spirit as Fig 5: a row of example
traces above the per-time-of-day pulse statistics.

Layout (GridSpec 3x6):
  Row 0 — example z-scored traces: A = one 10-min-sampled data recording
          (Henley et al. 2009), B = one model realization. Within each panel
          cortisol is the full tone and ACTH a lighter shade of the same hue;
          the hue itself encodes data/model role (blue/orange).
  Rows 1-2 — 2x2 boxplots, data vs model: C amplitude mean, D IPI mean,
             E amplitude CV (with the Rayleigh CV=0.523 reference), F IPI CV.

Model from the canonical fit (ACTH t½=20, cort t½=15, ε=1.5 lognormal drive
noise). Reuses the trajectory simulation from plot_3_example_trajectories_v6.py
and the peak/boxplot machinery from build_figure3_boxplots.py. Colour encodes
data/model role (blue/orange) for both the traces and the stats; within a trace
panel, cortisol is the full tone and ACTH the lighter shade of the same hue.

Usage:
  PYTHONPATH=src python scripts/build_figure3_combined.py \
      --fit-dir experiments/runs/eps15_acth20_cort15 \
      --peaks-csv experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv \
      --n-subjects 71 --n-reps 1 --max-ipi-min 240 \
      --out experiments/runs/manuscript_figure3_combined
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

import build_figure3_boxplots as bfb  # noqa: E402
import plot_3_example_trajectories_v6 as traj  # noqa: E402
from hpa_model.data.registry import get_dataset_spec, load_dataset  # noqa: E402
from hpa_model.plotting import apply_paper_style, setup_nature_style  # noqa: E402


def _load_data_example(dataset: str, variant: str, sid: str):
    """One subject's z-scored cortisol + ACTH from a 10-min dataset over TOD (h).

    Returns (t_hr, cort_z, acth_z); acth_z is None if the dataset has no ACTH.
    """
    spec = get_dataset_spec(dataset)
    ccol = next(s.column for s in spec.signals if s.name == "Cortisol")
    acol = next((s.column for s in spec.signals if s.name == "ACTH"), None)
    df = load_dataset(dataset, variant)
    g = df[df[spec.id_col].astype(str) == str(sid)].sort_values(spec.time_col)
    t = g[spec.time_col].to_numpy(float); cort = g[ccol].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(cort)
    cz = traj._zscore(cort[ok])
    az = None
    if acol is not None:
        acth = g[acol].to_numpy(float)[ok]
        if np.isfinite(acth).any():
            az = traj._zscore(acth)
    return t[ok] / 60.0, cz, az


def _model_trace(fit_dir: Path, seed: int):
    """One model realization's z-scored cortisol (x3) + ACTH (x2) over TOD (h)."""
    sim = traj._simulate_model(fit_dir, seed=seed)
    return (sim["time_min"].to_numpy() / 60.0,
            traj._zscore(sim["x3"].to_numpy()), traj._zscore(sim["x2"].to_numpy()))


def _lighten(color, amount=0.55):
    """Blend a color toward white by `amount` (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount,
            g + (1.0 - g) * amount,
            b + (1.0 - b) * amount)


def _render_trace(ax, t_hr, cz, az, *, letter, title, title_color, show_ylabel,
                  show_legend, ylim, cort_color, acth_color):
    """Z-scored cortisol + ACTH, both coloured by data/model role (cortisol full
    tone, ACTH a lighter shade of the same hue)."""
    lc, = ax.plot(t_hr, cz, color=cort_color, lw=1.1, label="Cortisol")
    handles, labels = [lc], ["Cortisol"]
    if az is not None:
        la, = ax.plot(t_hr, az, color=acth_color, lw=1.0, label="ACTH")
        handles.append(la); labels.append("ACTH")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6))
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time of day (h)", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("Z-score (within series)", fontsize=10.5)
    ax.set_title(f"{letter}  {title}", fontsize=10.0, loc="left", color=title_color)
    if show_legend:
        ax.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper right")
    apply_paper_style(ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/eps15_acth20_cort15")
    ap.add_argument("--peaks-csv", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05/artifacts/peak_amplitude_samples.csv")
    ap.add_argument("--n-subjects", type=int, default=71,
                    help="Independent model 'subjects' for the stats panels.")
    ap.add_argument("--n-reps", type=int, default=1, help="Replicates per subject.")
    ap.add_argument("--data-example-dataset", type=str, default="digitize_2019",
                    help="10-min-sampled dataset for the panel-A data trace "
                         "(default digitize_2019 = Henley et al. 2009, cortisol + ACTH).")
    ap.add_argument("--data-example-id", type=str, default="wpd_datasets-2.csv",
                    help="Subject ID for the panel-A data trace (default Henley series "
                         "wpd_datasets-2.csv: full 24 h at 10-min, cortisol + ACTH).")
    ap.add_argument("--data-example-variant", type=str, default="shifted_12h")
    ap.add_argument("--data-example-label", type=str, default="Data (Henley et al.)",
                    help="Panel-A title text for the data trace.")
    ap.add_argument("--model-example-seeds", type=int, nargs=2, default=(42, 1042),
                    help="Seeds for the model example traces; only the first is "
                         "rendered (panel B). Second kept for backward compatibility.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prom-sigma", type=float, default=0.5)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--resample-dt-min", type=float, default=20.0)
    ap.add_argument("--max-ipi-min", type=float, default=240.0)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/manuscript_figure3_combined")
    args = ap.parse_args()

    fig_dir = args.out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = args.out / "artifacts"; art_dir.mkdir(parents=True, exist_ok=True)

    # ── example traces (top row): A = 10-min data, B = model cortisol ─────────
    data_trace = _load_data_example(args.data_example_dataset,
                                    args.data_example_variant, args.data_example_id)
    model_traces = [_model_trace(args.fit_dir, seed=args.model_example_seeds[0])]

    # ── peaks for the stats panels ────────────────────────────────────────────
    data_peaks = bfb._load_data_peaks(args.peaks_csv)
    print(f"[data] {len(data_peaks)} peaks across "
          f"{data_peaks['series_uid'].nunique()} subjects")
    model_peaks = bfb._simulate_model_peaks(
        args.fit_dir, n_subjects=args.n_subjects, n_reps=args.n_reps,
        base_seed=args.seed, prom_sigma=args.prom_sigma,
        min_distance_min=args.min_distance_min, resample_dt_min=args.resample_dt_min,
    )
    print(f"[model] {len(model_peaks)} peaks across "
          f"{model_peaks['series_uid'].nunique()} subj×reps")

    # Physiological IPI cutoff (CLAUDE.md step 5).
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
    data_peaks.to_csv(art_dir / "data_peaks.csv", index=False)
    model_peaks.to_csv(art_dir / "model_peaks.csv", index=False)

    # Stats panels C-F: C amp mean, D IPI mean, E amp CV, F IPI CV.
    stat_panels = [
        ("amp", "mean", "Amplitude mean (Z-score)", "C  Amplitude mean", 1),
        ("ipi", "mean", "Inter-peak interval (IPI) mean (min)",
                        "D  Inter-peak interval (IPI) mean", 1),
        ("amp", "cv",   "Amplitude CV",             "E  Amplitude CV",   2),
        ("ipi", "cv",   "Inter-peak interval (IPI) CV",
                        "F  Inter-peak interval (IPI) CV",   2),
    ]
    rayleigh_cv = float(np.sqrt((4.0 - np.pi) / np.pi))
    longs = {}
    for value_col, stat, _ylabel, _title, mp in stat_panels:
        longs[(value_col, stat)] = (
            bfb._per_uid_bin_stat(data_peaks, value_col, stat, mp),
            bfb._per_uid_bin_stat(model_peaks, value_col, stat, mp),
        )
    cv_all = np.concatenate([
        np.concatenate([d["value"].to_numpy(float), m["value"].to_numpy(float)])
        for (vc, st), (d, m) in longs.items() if st == "cv"])
    cv_all = cv_all[np.isfinite(cv_all)]
    cv_ylim = (0.0, float(np.nanquantile(cv_all, 0.97)) * 1.15) if len(cv_all) else None

    # ── figure assembly ───────────────────────────────────────────────────────
    setup_nature_style()
    fig = plt.figure(figsize=(11.0, 11.0))
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[0.8, 1.0, 1.0],
                           hspace=0.42, wspace=0.9,
                           left=0.07, right=0.985, top=0.93, bottom=0.06)

    # Row 0 — example traces: A = data (10-min, Henley), B = model. Each panel
    # shows cortisol (full tone) + ACTH (lighter shade) in the data/model hue.
    traces = [("A", args.data_example_label, bfb.DATA_COLOR, data_trace)]
    for j, mt in enumerate(model_traces[:1]):
        traces.append(("B", "Model", bfb.MODEL_COLOR, mt))
    all_z = np.concatenate([
        np.concatenate([a for a in (cz, az) if a is not None and len(a)])
        for _, _, _, (_, cz, az) in traces])
    # Tight, asymmetric y-limits spanning the actual min->max across all 3 traces.
    if len(all_z):
        ylo, yhi = float(np.nanmin(all_z)), float(np.nanmax(all_z))
        pad = 0.08 * (yhi - ylo) if yhi > ylo else 0.5
        ylim = (ylo - pad, yhi + pad)
    else:
        ylim = (-2.6, 2.6)
    traj_slots = [gs[0, 0:3], gs[0, 3:6]]
    sharey_ax = None
    traj_axes = []
    for col, (letter, title, tcolor, (th, cz, az)) in enumerate(traces):
        ax = fig.add_subplot(traj_slots[col], sharey=sharey_ax)
        if sharey_ax is None:
            sharey_ax = ax
        traj_axes.append(ax)
        _render_trace(ax, th, cz, az, letter=letter, title=title, title_color=tcolor,
                      show_ylabel=(col == 0), show_legend=True, ylim=ylim,
                      cort_color=tcolor, acth_color=_lighten(tcolor))

    # Rows 1-2 — stats boxplots.
    stat_slots = [gs[1, 0:3], gs[1, 3:6], gs[2, 0:3], gs[2, 3:6]]
    stat_axes = []
    for slot, (value_col, stat, ylabel, title, _mp) in zip(stat_slots, stat_panels):
        ax = fig.add_subplot(slot)
        stat_axes.append(ax)
        d_long, m_long = longs[(value_col, stat)]
        bfb._box_pair(ax, d_long, m_long, ylabel=ylabel, title=title,
                      connect_trend=False,
                      ylim=cv_ylim if stat == "cv" else None)
        if value_col == "amp" and stat == "cv":
            ax.axhline(rayleigh_cv, color="#c0392b", linestyle="--", linewidth=1.0,
                       zorder=4, label=f"Theoretical CV = {rayleigh_cv:.3f}")
            ax.legend(loc="upper right", fontsize=9, frameon=False)

    # Figure-level Data/Model legend, placed in the gap between the realization
    # row and the stats row (it labels the stats panels below it).
    data_h = plt.Rectangle((0, 0), 1, 1, fc=bfb.DATA_COLOR, alpha=0.55, ec=bfb.DATA_COLOR)
    model_h = plt.Rectangle((0, 0), 1, 1, fc=bfb.MODEL_COLOR, alpha=0.55, ec=bfb.MODEL_COLOR)
    n_sims = args.n_subjects * args.n_reps
    traj_bottom = min(a.get_position().y0 for a in traj_axes)
    stats_top = max(stat_axes[0].get_position().y1, stat_axes[1].get_position().y1)
    # Sit just above the D/E panel titles (close to the stats), leaving more room
    # below the trace row.
    y_legend = stats_top + 0.35 * (traj_bottom - stats_top)
    fig.legend(
        [data_h, model_h],
        [f"Data (n={data_peaks['series_uid'].nunique()} subjects)",
         f"Model (n={n_sims} simulations)"],
        loc="center", bbox_to_anchor=(0.5, y_legend),
        ncol=2, frameon=False, fontsize=11,
    )

    out_png = fig_dir / "figure_3_combined.png"
    out_pdf = fig_dir / "figure_3_combined.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"PNG: {out_png}")
    print(f"PDF: {out_pdf}")

    (args.out / "manifest.json").write_text(json.dumps({
        "task": "build_figure3_combined",
        "created_at": datetime.now(UTC).isoformat(),
        "fit_dir": str(args.fit_dir),
        "peaks_csv": str(args.peaks_csv),
        "n_subjects": args.n_subjects, "n_reps": args.n_reps,
        "max_ipi_min": args.max_ipi_min, "base_seed": args.seed,
        "data_example_dataset": args.data_example_dataset,
        "data_example_id": args.data_example_id,
        "data_example_variant": args.data_example_variant,
        "model_example_seeds": list(args.model_example_seeds),
    }, indent=2))
    (args.out / "README.md").write_text(
        "# Combined Fig 3 (example traces + per-bin stats)\n\n"
        "Top row: z-scored cortisol (full tone) + ACTH (lighter shade) traces, "
        "hue = data/model role. A = one 10-min-sampled data recording "
        "(Henley et al. 2009 = digitize_2019), B = one model realization.\n"
        "Rows 2-3 (panels C-F): data-vs-model per-bin boxplots (C amp mean, "
        "D IPI mean, E amp CV, F IPI CV). Canonical model (ACTH t½=20, "
        "cort t½=15, ε=1.5).\n\n"
        "Built by `scripts/build_figure3_combined.py`.\n")


if __name__ == "__main__":
    main()
