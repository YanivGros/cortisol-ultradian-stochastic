"""Build Figure 1 (A, B, C) for the noise-induced oscillator manuscript.

Panels:
  A — Representative cortisol trajectories + two-harmonic baseline (image)
  B — Processing pipeline (3 steps: raw+baseline / residual / z-score+peaks)
  C — Peak amplitude distribution: Weibull vs Rayleigh
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kstest, rayleigh, weibull_min

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpa_model.analysis.plotting.ultradian_demodulated_diagnostics import (
    _load_shift_param_rows, _normalize_series_id, reconstruct_signal_baseline,
)
from hpa_model.data.registry import get_dataset_spec, load_dataset
from hpa_model.plotting import apply_paper_style, setup_nature_style

# ── constants ─────────────────────────────────────────────────────────────────
RAYLEIGH_CV    = float(np.sqrt((4 - np.pi) / np.pi))   # ≈ 0.5227
CORTISOL_BLUE  = "#2F5C85"
RAYLEIGH_RED   = "#C85C3A"
BASELINE_BLACK = "#222222"

PANEL_A_SRC = Path(
    "experiments/runs/manuscript_figure1a_shifted_12h"
    "/figures/figure_1/figure_1a_individual_trajectories_no_baseline.png"
)
PEAKS_CSV = Path(
    "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05"
    "/artifacts/peak_amplitude_samples.csv"
)

PIPELINE_SUBJECT = "2"
PIPELINE_VARIANT = "shifted_12h"


# ── Panel B helpers ───────────────────────────────────────────────────────────

def _load_pipeline_subject(
    series_id: str, *, prominence: float = 0.3, subtract_baseline: bool = True,
    variant: str = PIPELINE_VARIANT,
) -> dict:
    spec   = get_dataset_spec("habs")
    df     = load_dataset("habs", variant).sort_values([spec.id_col, spec.time_col])
    shifts = _load_shift_param_rows("habs", variant)
    sr     = shifts.get(_normalize_series_id(spec.id_col, series_id))
    col    = next(s.column for s in spec.signals if s.name == "Cortisol")

    grp  = df[df[spec.id_col].astype(str) == str(series_id)]
    t    = grp[spec.time_col].to_numpy(float)
    raw  = grp[col].to_numpy(float)
    ok   = np.isfinite(t) & np.isfinite(raw)
    t, raw = t[ok], raw[ok]

    if subtract_baseline:
        baseline, _ = reconstruct_signal_baseline(
            dataset_name="habs", signal_name="Cortisol",
            time_min=t, values=raw, shift_row=sr)
    else:
        baseline = np.full_like(raw, float(np.mean(raw)))

    residual = raw - baseline
    mean_r, std_r = residual.mean(), residual.std()
    rz = (residual - mean_r) / std_r

    dt = float(np.median(np.diff(t)))
    peaks, props = find_peaks(rz, distance=int(round(60.0 / dt)), prominence=prominence)
    prominences = props["prominences"][: peaks.size]

    # Index of the previous-trough minimum used for the peak−prev_dip metric.
    prev_dip_idx = np.empty(peaks.size, dtype=int)
    for i, p in enumerate(peaks):
        lo = 0 if i == 0 else int(peaks[i - 1])
        if p > lo:
            prev_dip_idx[i] = int(lo + np.argmin(rz[lo:p]))
        else:
            prev_dip_idx[i] = int(p)

    return dict(t_hr=t / 60.0, raw=raw, baseline=baseline,
                residual=residual, rz=rz, peaks=peaks,
                prominences=prominences, prev_dip_idx=prev_dip_idx)


def _annotate_metric_arrows(ax: plt.Axes, t, rz, pk, dip) -> None:
    """Illustrate the two ultradian metrics on the z-score panel:
    a vertical trough→peak arrow (= amplitude) and a horizontal
    peak→peak arrow (= inter-peak interval, IPI)."""
    if pk.size < 2:
        return

    # Both metrics are illustrated in the widest (most open) inter-peak gap so
    # the annotations do not collide with neighbouring pulses.
    gaps = np.diff(t[pk])
    k = int(np.argmax(gaps))
    pa, pb = int(pk[k]), int(pk[k + 1])

    # Trough→peak (amplitude): on the right peak of the open pair, offset right.
    d_b = int(dip[k + 1])
    x_amp = t[pb] + 0.45
    ax.annotate(
        "", xy=(x_amp, rz[pb]), xytext=(x_amp, rz[d_b]),
        arrowprops={"arrowstyle": "<->", "color": RAYLEIGH_RED, "lw": 1.6},
        zorder=6,
    )
    ax.text(x_amp + 0.3, 0.5 * (rz[pb] + rz[d_b]), "trough→peak\n(amplitude)",
            color=RAYLEIGH_RED, fontsize=9, va="center", ha="left")

    # Peak→peak (IPI): horizontal arrow spanning the open pair, above the trace.
    y_ipi = float(max(rz[pk])) + 0.7
    ax.annotate(
        "", xy=(t[pb], y_ipi), xytext=(t[pa], y_ipi),
        arrowprops={"arrowstyle": "<->", "color": "#222222", "lw": 1.6},
        zorder=6,
    )
    ax.text(0.5 * (t[pa] + t[pb]), y_ipi + 0.2, "peak→peak (IPI)",
            color="#222222", fontsize=9, va="bottom", ha="center")
    ax.set_ylim(top=y_ipi + 1.3)


def _draw_pipeline(axes: list[plt.Axes], data: dict, *,
                   subtract_baseline: bool = True) -> None:
    t   = data["t_hr"]
    col = CORTISOL_BLUE

    ax = axes[0]
    ax.plot(t, data["raw"], color=col, lw=1.3, label="Cortisol")
    if subtract_baseline:
        ax.plot(t, data["baseline"], color=BASELINE_BLACK, lw=1.2, ls="--",
                label="2-harmonic baseline")
        ax.set_title("Step 1  Raw signal + circadian baseline",
                     fontsize=11, loc="left", pad=3)
    else:
        ax.axhline(float(np.mean(data["raw"])), color=BASELINE_BLACK,
                   lw=1.0, ls=":", label=f"Mean ({np.mean(data['raw']):.0f} nmol/L)")
        ax.set_title("Step 1  Raw signal (no circadian subtraction)",
                     fontsize=11, loc="left", pad=3)
    ax.set_ylabel("nmol/L", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    apply_paper_style(ax)
    ax.set_xticklabels([])

    ax = axes[1]
    ax.plot(t, data["residual"], color=col, lw=1.2)
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_ylabel("Residual (nmol/L)", fontsize=10.5)
    title2 = ("Step 2  Residual  =  raw − baseline" if subtract_baseline
              else "Step 2  Mean-centered raw")
    ax.set_title(title2, fontsize=11, loc="left", pad=3)
    apply_paper_style(ax)
    ax.set_xticklabels([])

    ax = axes[2]
    ax.plot(t, data["rz"], color=col, lw=1.2, label="Z-scored residual")
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    pk  = data["peaks"]
    rz  = data["rz"]
    dip = data["prev_dip_idx"]
    ax.scatter(t[pk], rz[pk], s=32, color=RAYLEIGH_RED,
               zorder=5, label="Detected peaks", marker="v")
    ax.scatter(t[dip], rz[dip], s=22, color="#444444",
               zorder=5, marker="o", label="Previous trough")
    for p_i, d_i in zip(pk, dip):
        ax.vlines(t[p_i], rz[d_i], rz[p_i],
                  color="#444444", lw=0.9, zorder=4)
    ax.plot([], [], color="#444444", lw=1.0,
            label="Trough to peak (Z)")
    _annotate_metric_arrows(ax, t, rz, pk, dip)
    ax.set_ylabel("Z-score", fontsize=10.5)
    ax.set_xlabel("Time (h)", fontsize=11)
    # Legend below the axes so it never overlaps the trace or markers.
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=4,
              handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    ax.set_title("Step 3  Amplitude = trough to peak (Z-score)",
                 fontsize=11, loc="left", pad=3)
    apply_paper_style(ax)

    for ax in axes:
        ax.set_xlim(t[0], t[-1])


# ── Panel C ───────────────────────────────────────────────────────────────────

_METRIC_COLUMN = {
    "amplitude":     ("peak_amplitude_sigma",          "Peak amplitude (within-subject Z-score)"),
    "prominence":    ("prominence",                    "Peak prominence (within-subject Z-score)"),
    "prev_dip":      ("peak_amplitude_prev_dip_sigma", "Trough-to-peak amplitude (within-subject Z-score)"),
}


def _draw_amplitude_dist(
    ax: plt.Axes, peaks_csv: Path, *, metric: str = "prev_dip",
    show_stats_text: bool = True,
) -> dict:
    col_name, x_label = _METRIC_COLUMN[metric]
    df   = pd.read_csv(peaks_csv)
    vals = df[col_name].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    # For raw amplitude (peak z-score) keep only positives to match Rayleigh
    # support; prominence and prev_dip are non-negative by construction.
    if metric == "amplitude":
        vals = vals[vals > 0]

    loc_r, sc_r = rayleigh.fit(vals, floc=0.0)
    _, p_r = kstest(vals, "rayleigh",    args=(loc_r, sc_r))
    c_w, loc_w, sc_w = weibull_min.fit(vals, floc=0.0)
    _, p_w = kstest(vals, "weibull_min", args=(c_w, loc_w, sc_w))
    cv = vals.std() / vals.mean()
    # bootstrap 95% CI for the CV (for the panel-C title; comment "add this
    # to the plot title" → CV with CI that contains the Rayleigh value 0.523).
    _rng = np.random.default_rng(0)
    _boot = np.empty(2000)
    for _i in range(_boot.size):
        _s = _rng.choice(vals, size=vals.size, replace=True)
        _boot[_i] = _s.std() / _s.mean()
    cv_lo, cv_hi = (float(v) for v in np.percentile(_boot, [2.5, 97.5]))

    ax.hist(vals, bins=25, density=True,
            color=CORTISOL_BLUE, alpha=0.30, edgecolor="white", linewidth=0.6)

    x = np.linspace(0.0, float(vals.max()) + 0.3, 512)
    ax.plot(x, rayleigh.pdf(x, loc_r, sc_r),
            color=RAYLEIGH_RED, lw=1.8, ls="-",
            label=f"Rayleigh fitted ($A_0$ = {sc_r:.2f})  KS p = {p_r:.2g}")

    # Extra headroom at the top so the legend (upper-left) and annotation
    # (upper-right) clear the histogram / Rayleigh curve.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    # In-panel annotation box (peak count, series, the Rayleigh PDF and its
    # fitted scale A_0). Omit with show_stats_text=False — the same numbers are
    # written to figure_1_summary.json for the caption.
    if show_stats_text:
        ax.text(0.97, 0.97,
                f"number of peaks = {len(vals)}\n"
                f"n 24 h cortisol series = {int(df['series_uid'].nunique())}\n"
                f"data CV = {cv:.3f}  (theoretical Rayleigh CV = {RAYLEIGH_CV:.3f})\n"
                r"$P(A)=\dfrac{A}{A_0^{2}}\,e^{-A^{2}/(2A_0^{2})}$" + "\n"
                f"$A_0 = {sc_r:.2f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.85, "pad": 2})
    else:
        # Keep only the headline parameter-free result on the panel: the data CV
        # matches the fixed Rayleigh CV. Everything else goes to the caption.
        ax.text(0.97, 0.97,
                f"data CV 95% CI {cv_lo:.2f}–{cv_hi:.2f}\n"
                f"theoretical CV = {RAYLEIGH_CV:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.85, "pad": 2})
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    apply_paper_style(ax)

    return dict(metric=metric, weibull_k=c_w, weibull_p=p_w, rayleigh_p=p_r,
                rayleigh_scale=sc_r, rayleigh_cv_theoretical=RAYLEIGH_CV,
                n=len(vals), n_series=int(df["series_uid"].nunique()), cv=cv,
                cv_lo=cv_lo, cv_hi=cv_hi)


# ── assembly ──────────────────────────────────────────────────────────────────

def build_figure(
    out_dir: Path, peaks_csv: Path = PEAKS_CSV, *,
    metric: str = "prev_dip", prominence: float = 0.3,
    subtract_baseline: bool = True, pipeline_variant: str = PIPELINE_VARIANT,
    panel_a_src: Path = PANEL_A_SRC, show_stats_text: bool = True,
) -> tuple[Path, Path]:
    fig_dir = out_dir / "figures" / "figure_1"
    art_dir = out_dir / "artifacts" / "figure_1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    setup_nature_style()

    # Panel A image is 4179×1265 px (aspect ≈ 3.30). Usable width ≈ 11.83 in
    # → natural height ≈ 3.58 in. height_ratios=[1.5,1.7] with figsize H=10.5
    # gives A ≈ 3.54 in (fills the image naturally) and BC ≈ 4.01 in.
    fig = plt.figure(figsize=(13.0, 10.5))
    gs0 = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.5, 1.7],
        width_ratios=[1.0, 1.4],
        hspace=0.22, wspace=0.32,
        left=0.06, right=0.97, top=0.96, bottom=0.11,
    )

    # ── A: full top row ───────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs0[0, :])
    ax_a.imshow(mpimg.imread(panel_a_src))
    ax_a.set_axis_off()
    ax_a.text(-0.02, 1.07, "A", transform=ax_a.transAxes,
              fontsize=18, fontweight="bold", va="bottom")
    ax_a.text(0.055, 1.07,
              "Representative cortisol trajectories",
              transform=ax_a.transAxes, fontsize=12.5, color="#222", va="bottom")

    # ── B: pipeline, 3 stacked sub-panels in col 0 ───────────────────────────
    gs_b = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs0[1, 0], hspace=0.52)
    ax_b = [fig.add_subplot(gs_b[i]) for i in range(3)]

    pipeline_data = _load_pipeline_subject(PIPELINE_SUBJECT, prominence=prominence,
                                           subtract_baseline=subtract_baseline,
                                           variant=pipeline_variant)
    _draw_pipeline(ax_b, pipeline_data, subtract_baseline=subtract_baseline)

    ax_b[0].text(-0.18, 1.32, "B", transform=ax_b[0].transAxes,
                 fontsize=18, fontweight="bold", va="bottom")
    ax_b[0].text(-0.05, 1.32,
                 "Extracting ultradian pulses from cortisol",
                 transform=ax_b[0].transAxes, fontsize=12.5, color="#222", va="bottom")

    # ── C: amplitude distribution in col 1 ───────────────────────────────────
    ax_c = fig.add_subplot(gs0[1, 1])
    stats = _draw_amplitude_dist(ax_c, peaks_csv, metric=metric,
                                 show_stats_text=show_stats_text)
    panel_c_title = {
        "amplitude":  "Peak amplitude distribution",
        "prominence": "Peak prominence distribution",
        "prev_dip":   "Trough-to-peak amplitude distribution",
    }[metric]
    # CV (with bootstrap 95% CI) is shown in the in-panel annotation box, not
    # the title.
    ax_c.text(-0.14, 1.06, "C", transform=ax_c.transAxes,
              fontsize=18, fontweight="bold", va="bottom")
    ax_c.text(-0.03, 1.06, panel_c_title,
              transform=ax_c.transAxes, fontsize=11.5, color="#222", va="bottom")

    # ── save ──────────────────────────────────────────────────────────────────
    png_path = fig_dir / "figure_1.png"
    pdf_path = fig_dir / "figure_1.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (art_dir / "figure_1_summary.json").write_text(json.dumps(stats, indent=2))
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/runs/manuscript_figure1_abc_prom05_prev_dip"))
    parser.add_argument("--peaks-csv", type=Path, default=PEAKS_CSV)
    parser.add_argument("--metric", choices=("prev_dip", "prominence", "amplitude"),
                        default="prev_dip",
                        help="Panel C metric: prev_dip (default), prominence, or amplitude")
    parser.add_argument("--prom", type=float, default=0.5,
                        help="Peak prominence for Panel B pipeline detection")
    parser.add_argument("--subtract-baseline", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Subtract two-harmonic circadian baseline in Panel B "
                             "(disable with --no-subtract-baseline).")
    parser.add_argument("--pipeline-variant", type=str, default=PIPELINE_VARIANT,
                        help="Dataset variant for the Panel B pipeline subject "
                             "(default: shifted_12h).")
    parser.add_argument("--panel-a-src", type=Path, default=PANEL_A_SRC,
                        help="Pre-rendered Panel A trajectory PNG "
                             "(default: shifted_12h figure_1a image).")
    parser.add_argument("--stats-text", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Draw the Panel C annotation box (peak count, series, "
                             "data/theoretical CV, Rayleigh PDF + A_0). Use "
                             "--no-stats-text to omit it (move to the caption); values "
                             "are still written to figure_1_summary.json. The legend "
                             "KS-p is always kept.")
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "task": "build_figure1_abc",
        "created_at": datetime.now(UTC).isoformat(),
        "panels": {"A": str(args.panel_a_src), "B": f"HABS subject {PIPELINE_SUBJECT}",
                   "C": str(args.peaks_csv)},
    }, indent=2))
    png, pdf = build_figure(out_dir, peaks_csv=args.peaks_csv,
                            metric=args.metric,
                            prominence=args.prom,
                            subtract_baseline=args.subtract_baseline,
                            pipeline_variant=args.pipeline_variant,
                            panel_a_src=args.panel_a_src,
                            show_stats_text=args.stats_text)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
