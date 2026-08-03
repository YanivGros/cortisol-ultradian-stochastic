"""Build Figure 2: per-bin peak amplitude and IPI statistics across datasets.

Layout: 2 rows × 2 cols, grouped boxplots
  (0,0) — Amplitude mean per bin    (0,1) — Amplitude CV per bin
  (1,0) — IPI mean per bin          (1,1) — IPI CV per bin

Within each panel, each time-of-day bin shows one box per dataset
(HABS blood, Russell & Lightman, All Digitized by default).

A dashed reference line at Rayleigh CV = 0.523 is overlaid in the amp-CV panel.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hpa_model.plotting import apply_paper_style, setup_nature_style

# ── constants ─────────────────────────────────────────────────────────────────
RAYLEIGH_CV  = float(np.sqrt((4 - np.pi) / np.pi))   # ≈ 0.5227
RAYLEIGH_RED = "#C85C3A"

BIN_EDGES  = [0, 240, 480, 720, 960, 1440]
# Morning-first bins on a clock shifted to a 04:00 origin, so the overnight
# window 20:00-04:00 forms a single contiguous (wrapping) bin placed last.
BIN_LABELS = ["04-08", "08-12", "12-16", "16-20", "20-04"]
TOD_ORIGIN_MIN = 240.0  # shift so 04:00 -> 0 and 20:00-04:00 is the last bin

# Dataset order + colors
DATASET_ORDER = [
    "HABS",
    "Russell & Lightman",
    "All Digitized",
    "HABS Microdialysis",
]
DATASET_COLORS = {
    "HABS":               "#2F5C85",
    "Russell & Lightman": "#7A6FB0",
    "All Digitized":      "#C7943C",
    "HABS Microdialysis": "#3E8E5A",
    "All datasets":       "#2F5C85",
}
POOLED_LABEL = "All datasets"

# Default CSV source (prom 0.50 blood-pooled across habs / digitize_2019 / all_digitized,
# with the peak−prev_dip amplitude column).
DEFAULT_BLOOD_CSV = Path(
    "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_prom05"
    "/artifacts/peak_amplitude_samples.csv"
)
DEFAULT_MD_CSV = Path(
    "experiments/runs/manuscript_peak_amplitude_direct_rayleigh_microdialysis_prom05"
    "/artifacts/peak_amplitude_samples.csv"
)


# ── data helpers ──────────────────────────────────────────────────────────────

def _bin_by_tod(df: pd.DataFrame, time_col: str = "time_min") -> pd.DataFrame:
    df = df.copy()
    df["tod"] = (df[time_col] % 1440.0 - TOD_ORIGIN_MIN) % 1440.0
    df["bin"] = pd.cut(
        df["tod"], bins=BIN_EDGES, labels=BIN_LABELS,
        right=True, include_lowest=True,
    ).astype(str)
    return df


def _per_subject_bin_mean(
    df: pd.DataFrame, value_col: str, min_peaks: int = 1,
) -> pd.DataFrame:
    rows = []
    for (uid, b), grp in df.groupby(["series_uid", "bin"], observed=True):
        vals = grp[value_col].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= min_peaks:
            rows.append({"series_uid": uid, "bin": str(b),
                         "value": float(np.mean(vals))})
    return pd.DataFrame(rows)


def _per_subject_bin_cv(
    df: pd.DataFrame, value_col: str, min_peaks: int = 2,
) -> pd.DataFrame:
    rows = []
    for (uid, b), grp in df.groupby(["series_uid", "bin"], observed=True):
        vals = grp[value_col].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= min_peaks:
            cv = float(vals.std(ddof=1) / vals.mean())
            rows.append({"series_uid": uid, "bin": str(b), "value": cv})
    return pd.DataFrame(rows)


def _compute_ipi(df: pd.DataFrame, max_ipi_min: float | None = None) -> pd.DataFrame:
    rows = []
    for uid, grp in df.groupby("series_uid"):
        times = grp["time_min"].sort_values().to_numpy(float)
        if len(times) < 2:
            continue
        ipis = np.diff(times)
        for t, ipi in zip(times[:-1], ipis):
            # Physiological cutoff (CLAUDE.md step 5): drop intervals above the
            # cutoff as unresolved merges; amplitude panels are unaffected.
            if ipi > 0 and (max_ipi_min is None or ipi <= max_ipi_min):
                rows.append({"series_uid": uid, "time_min": t, "ipi_min": ipi})
    if not rows:
        return pd.DataFrame(columns=["series_uid", "time_min", "ipi_min"])
    return pd.DataFrame(rows)


def _build_panel_table(
    peaks: pd.DataFrame, kind: str, *, amp_col: str = "peak_amplitude_prev_dip_sigma",
    max_ipi_min: float | None = None,
) -> pd.DataFrame:
    """
    Build long table {series_uid, bin, value, dataset_label} for one metric.
    kind ∈ {"amp_mean","amp_cv","ipi_mean","ipi_cv"}.
    """
    out: list[pd.DataFrame] = []
    for label, sub in peaks.groupby("dataset_label", sort=False):
        if kind in ("amp_mean", "amp_cv"):
            sub_b = _bin_by_tod(sub, "time_min")
            if kind == "amp_mean":
                df = _per_subject_bin_mean(sub_b, amp_col, min_peaks=1)
            else:
                df = _per_subject_bin_cv(sub_b, amp_col, min_peaks=2)
        else:  # IPI metrics
            ipi = _compute_ipi(sub, max_ipi_min=max_ipi_min)
            if ipi.empty:
                continue
            ipi_b = _bin_by_tod(ipi, "time_min")
            if kind == "ipi_mean":
                df = _per_subject_bin_mean(ipi_b, "ipi_min", min_peaks=1)
            else:
                df = _per_subject_bin_cv(ipi_b, "ipi_min", min_peaks=2)
        if df.empty:
            continue
        df["dataset_label"] = label
        out.append(df)
    if not out:
        return pd.DataFrame(columns=["series_uid", "bin", "value", "dataset_label"])
    return pd.concat(out, ignore_index=True)


# ── pairwise post-hoc tests ──────────────────────────────────────────────────

def _pairwise_mannwhitney_bonferroni(
    long_df: pd.DataFrame, bins: list[str], *, min_n: int = 6,
) -> list[dict]:
    """Unpaired Mann-Whitney U between every pair of bins (independent samples;
    no shared-`series_uid` requirement), Bonferroni-corrected over the total
    number of attempted pairs (C(n,2)). Used for the mean panels because too few
    subjects have a pulse in every bin for a well-powered paired test, so the
    paired Wilcoxon/Friedman discarded most of the per-subject bin means."""
    pairs = list(combinations(range(len(bins)), 2))
    n_pairs = len(pairs)
    out = []
    for i, j in pairs:
        bi, bj = bins[i], bins[j]
        a = long_df.loc[long_df["bin"] == bi, "value"].dropna().to_numpy(float)
        b = long_df.loc[long_df["bin"] == bj, "value"].dropna().to_numpy(float)
        if len(a) < min_n or len(b) < min_n:
            continue
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        out.append({"i": i, "j": j, "bin_i": bi, "bin_j": bj,
                    "n_i": int(len(a)), "n_j": int(len(b)),
                    "p_raw": float(p),
                    "p_bonferroni": float(min(1.0, p * n_pairs))})
    return out


def _kruskal_panel_test(
    long_df: pd.DataFrame, bins: list[str], *, min_per_group: int = 5,
) -> dict | None:
    """Kruskal-Wallis across bins (independent samples). Used for every panel
    (means and CVs), because too few subjects have a value in every bin for a
    paired Friedman test."""
    groups, n_used = [], []
    for b in bins:
        v = long_df.loc[long_df["bin"] == b, "value"].dropna().to_numpy(float)
        if len(v) >= min_per_group:
            groups.append(v)
            n_used.append(int(len(v)))
    if len(groups) < 3:
        return None
    try:
        stat, p = kruskal(*groups)
    except ValueError:
        return None
    return {"H": float(stat), "p": float(p),
            "k_bins": int(len(groups)), "n_total": int(sum(n_used))}


def _annotate_kruskal(ax: plt.Axes, result: dict | None) -> None:
    if not result:
        return
    p = result["p"]
    p_str = f"p = {p:.1e}" if p < 0.01 else f"p = {p:.3f}"
    txt = (f"Kruskal–Wallis H({result['k_bins'] - 1}) = {result['H']:.2f}\n"
           f"{p_str}  (n = {result['n_total']})")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.85, "pad": 2})


def _draw_significance_brackets(
    ax: plt.Axes, results: list[dict], *, alpha: float = 0.05,
) -> None:
    sig = [r for r in results if r["p_bonferroni"] < alpha]
    if not sig:
        return
    # Short brackets first so they stack underneath longer ones.
    sig.sort(key=lambda r: (r["j"] - r["i"], r["i"]))

    y_lo, y_hi = ax.get_ylim()
    step = (y_hi - y_lo) * 0.05
    base = y_hi + step * 0.5
    level_gap = 2.1  # vertical spacing between stacked brackets (in units of step)
    for level, r in enumerate(sig):
        y = base + step * level * level_gap
        i, j = r["i"], r["j"]
        ax.plot([i, i, j, j], [y - step * 0.25, y, y, y - step * 0.25],
                color="black", lw=0.8, clip_on=False)
        p = r["p_bonferroni"]
        if   p < 0.001: marker = "***"
        elif p < 0.01:  marker = "**"
        else:           marker = "*"
        ax.text((i + j) / 2, y + step * 0.1, marker,
                ha="center", va="bottom", fontsize=11, clip_on=False)
    ax.set_ylim(y_lo, base + step * (len(sig) * level_gap + 1.2))


# ── plotting ──────────────────────────────────────────────────────────────────

def _grouped_box_panel(
    ax: plt.Axes,
    long_df: pd.DataFrame,
    bins: list[str],
    datasets: list[str],
    *,
    ylabel: str,
    title: str,
    reference_line: float | None = None,
    reference_label: str | None = None,
    show_n: bool = True,
    connect_trend: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    present = [d for d in datasets if d in long_df["dataset_label"].unique()]
    n_ds = len(present)
    if n_ds == 0:
        ax.set_title(title, fontsize=12.5, loc="left")
        return

    group_w = 0.78
    box_w   = group_w / n_ds * 0.85

    legend_handles = []
    for k, ds in enumerate(present):
        color = DATASET_COLORS.get(ds, "#555555")
        offsets = (np.arange(n_ds) - (n_ds - 1) / 2) * (group_w / n_ds)
        offset = offsets[k]

        data_by_bin: list[np.ndarray] = []
        pos: list[float] = []
        for i, b in enumerate(bins):
            vals = long_df.loc[
                (long_df["dataset_label"] == ds) & (long_df["bin"] == b),
                "value",
            ].dropna().to_numpy(float)
            data_by_bin.append(vals)
            pos.append(i + offset)

        # boxplot
        ax.boxplot(
            [v if len(v) else np.array([np.nan]) for v in data_by_bin],
            positions=pos, widths=box_w, patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.4},
            boxprops={"facecolor": color, "alpha": 0.55, "linewidth": 0.6,
                      "edgecolor": color},
            whiskerprops={"linewidth": 0.6, "color": color},
            capprops={"linewidth": 0.6, "color": color},
            showfliers=False,
        )
        # individual data points (jittered strip overlay)
        rng = np.random.default_rng(42 + k)
        jitter_w = box_w * 0.35
        for p, vals in zip(pos, data_by_bin):
            if not len(vals):
                continue
            xs = p + rng.uniform(-jitter_w, jitter_w, size=len(vals))
            ax.scatter(
                xs, vals, s=14, color=color, alpha=0.65,
                edgecolors="none", zorder=3,
            )
        # n annotations below x-axis (one tiny number per dataset per bin)
        if show_n:
            y0 = ax.get_ylim()[0]
            for p, vals in zip(pos, data_by_bin):
                if not len(vals):
                    continue
                ax.annotate(
                    f"{len(vals)}", xy=(p, y0),
                    xytext=(0, -8 - 6 * k), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color=color,
                    annotation_clip=False,
                )
        # Trend line connecting the per-bin means (shows the time-of-day trend).
        if connect_trend:
            bin_means = [float(np.mean(v)) if len(v) else np.nan
                         for v in data_by_bin]
            ax.plot(pos, bin_means, color=color, lw=1.8, zorder=4,
                    marker="o", markersize=4.5, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.2)

        legend_handles.append(mpatches.Patch(facecolor=color, alpha=0.55,
                                             edgecolor=color, label=ds))

    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(bins, fontsize=10.5)
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.set_xlabel("Time of day (h)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    apply_paper_style(ax)

    if reference_line is not None:
        ax.axhline(reference_line, color=RAYLEIGH_RED, lw=1.3, ls="--",
                   zorder=4,
                   label=reference_label or f"= {reference_line:.3f}")

    # clip extreme outliers for readable y-axis (or use a caller-supplied ylim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        all_vals = long_df["value"].to_numpy(float)
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals):
            upper = float(np.nanquantile(all_vals, 0.97))
            ax.set_ylim(0, min(upper * 1.3, float(all_vals.max()) * 1.05))

    ax.set_title(title, fontsize=12.5, loc="left")
    if reference_line is not None:
        ax.legend(frameon=False, fontsize=10, loc="upper right",
                  bbox_to_anchor=(1.0, 1.06))

    return legend_handles


# ── main assembly ─────────────────────────────────────────────────────────────

def build_figure(
    csv_paths: list[Path],
    out_dir: Path,
    *,
    show_n: bool = True,
    pool: bool = False,
    amp_col: str = "peak_amplitude_prev_dip_sigma",
    signal: str = "Cortisol",
    connect_trend: bool = True,
    max_ipi_min: float | None = 240.0,
    stats_text: bool = True,
) -> tuple[Path, Path]:
    fig_dir = out_dir / "figures" / "figure_2"
    art_dir = out_dir / "artifacts" / "figure_2"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    setup_nature_style()

    # Load & concat all peak tables, keep only the requested signal
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "signal" in df.columns:
            df = df[df["signal"] == signal]
        frames.append(df)
    peaks = pd.concat(frames, ignore_index=True)
    if pool:
        # Treat all datasets as one pooled group, but keep series_uid namespaced.
        peaks["dataset_label"] = POOLED_LABEL
        dataset_order = [POOLED_LABEL]
    else:
        dataset_order = DATASET_ORDER
    peaks.to_csv(art_dir / "combined_peaks.csv", index=False)

    # Build long tables per metric
    amp_mean_df = _build_panel_table(peaks, "amp_mean", amp_col=amp_col)
    amp_cv_df   = _build_panel_table(peaks, "amp_cv",   amp_col=amp_col)
    ipi_mean_df = _build_panel_table(peaks, "ipi_mean", amp_col=amp_col,
                                     max_ipi_min=max_ipi_min)
    ipi_cv_df   = _build_panel_table(peaks, "ipi_cv",   amp_col=amp_col,
                                     max_ipi_min=max_ipi_min)

    amp_mean_df.to_csv(art_dir / "amp_mean_per_bin.csv", index=False)
    amp_cv_df  .to_csv(art_dir / "amp_cv_per_bin.csv",   index=False)
    ipi_mean_df.to_csv(art_dir / "ipi_mean_per_bin.csv", index=False)
    ipi_cv_df  .to_csv(art_dir / "ipi_cv_per_bin.csv",   index=False)

    fig = plt.figure(figsize=(11.5, 8.5))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.48, wspace=0.30,
        left=0.07, right=0.985, top=0.90, bottom=0.13,
    )

    def _pooled(df: pd.DataFrame) -> pd.DataFrame:
        return (df.copy() if pool
                else df.groupby(["series_uid", "bin"], as_index=False)["value"].mean())

    def _kruskal_and_annotate(ax, df, name):
        # All panels: too few subjects have a value in every bin for a paired
        # Friedman, so use Kruskal-Wallis across bins (independent samples).
        res = _kruskal_panel_test(_pooled(df), BIN_LABELS)
        if res:
            pd.DataFrame([res]).to_csv(art_dir / f"{name}_kruskal.csv", index=False)
        if stats_text:
            _annotate_kruskal(ax, res)
        return res

    # Shared y-limit for the two CV panels (bottom row): common 0..upper scale.
    cv_vals = np.concatenate([
        amp_cv_df["value"].to_numpy(float), ipi_cv_df["value"].to_numpy(float)])
    cv_vals = cv_vals[np.isfinite(cv_vals)]
    cv_upper = float(np.nanquantile(cv_vals, 0.97)) * 1.3 if len(cv_vals) else 1.0
    cv_ylim = (0.0, cv_upper)

    # ── Top row: means ────────────────────────────────────────────────────────
    # (0,0) — amplitude mean
    ax = fig.add_subplot(gs[0, 0])
    handles = _grouped_box_panel(
        ax, amp_mean_df, BIN_LABELS, dataset_order,
        ylabel="Amplitude mean (Z-score)",
        title="A  Amplitude mean",
        show_n=show_n, connect_trend=connect_trend,
    )
    amp_mean_pooled = _pooled(amp_mean_df)
    # Unpaired omnibus + post-hoc: too few subjects have a pulse in every bin for
    # a paired Friedman/Wilcoxon, so use Kruskal-Wallis across bins with pairwise
    # Mann-Whitney (Bonferroni), matching the CV panels (no shared-ID requirement).
    _kruskal_and_annotate(ax, amp_mean_df, "amp_mean")
    mannwhitney_results = _pairwise_mannwhitney_bonferroni(amp_mean_pooled, BIN_LABELS)
    pd.DataFrame(mannwhitney_results).to_csv(
        art_dir / "amp_mean_pairwise_mannwhitney.csv", index=False,
    )
    _draw_significance_brackets(ax, mannwhitney_results)

    # (0,1) — IPI mean
    ax = fig.add_subplot(gs[0, 1])
    _grouped_box_panel(
        ax, ipi_mean_df, BIN_LABELS, dataset_order,
        ylabel="IPI mean (min)",
        title="B  Inter-peak interval (IPI) mean",
        show_n=show_n, connect_trend=connect_trend,
    )
    ipi_mean_pooled = _pooled(ipi_mean_df)
    # Unpaired Kruskal-Wallis + pairwise Mann-Whitney (Bonferroni), matching panel A.
    _kruskal_and_annotate(ax, ipi_mean_df, "ipi_mean")
    ipi_mannwhitney_results = _pairwise_mannwhitney_bonferroni(ipi_mean_pooled, BIN_LABELS)
    pd.DataFrame(ipi_mannwhitney_results).to_csv(
        art_dir / "ipi_mean_pairwise_mannwhitney.csv", index=False,
    )
    _draw_significance_brackets(ax, ipi_mannwhitney_results)

    # ── Bottom row: CVs (shared y-limit) ──────────────────────────────────────
    # (1,0) — amplitude CV
    ax = fig.add_subplot(gs[1, 0])
    _grouped_box_panel(
        ax, amp_cv_df, BIN_LABELS, dataset_order,
        ylabel="Amplitude CV", title="C  Amplitude CV",
        reference_line=RAYLEIGH_CV,
        reference_label=f"Theoretical CV = {RAYLEIGH_CV:.3f}",
        show_n=show_n, ylim=cv_ylim, connect_trend=connect_trend,
    )
    _kruskal_and_annotate(ax, amp_cv_df, "amp_cv")

    # (1,1) — IPI CV
    ax = fig.add_subplot(gs[1, 1])
    _grouped_box_panel(
        ax, ipi_cv_df, BIN_LABELS, dataset_order,
        ylabel="IPI CV", title="D  Inter-peak interval (IPI) CV",
        show_n=show_n, ylim=cv_ylim, connect_trend=connect_trend,
    )
    _kruskal_and_annotate(ax, ipi_cv_df, "ipi_cv")

    png_path = fig_dir / "figure_2.png"
    pdf_path = fig_dir / "figure_2.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # per-dataset, per-bin summary
    summary_rows = []
    for kind, df in [("amp_mean", amp_mean_df), ("amp_cv", amp_cv_df),
                     ("ipi_mean", ipi_mean_df), ("ipi_cv", ipi_cv_df)]:
        for (ds, b), grp in df.groupby(["dataset_label", "bin"]):
            v = grp["value"].dropna().to_numpy(float)
            if not len(v):
                continue
            summary_rows.append({
                "metric": kind, "dataset_label": ds, "bin": b,
                "n_subjects": int(len(v)),
                "median": float(np.median(v)),
                "mean":   float(np.mean(v)),
                "iqr_lo": float(np.quantile(v, 0.25)),
                "iqr_hi": float(np.quantile(v, 0.75)),
            })
    pd.DataFrame(summary_rows).to_csv(art_dir / "figure_2_summary.csv", index=False)

    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, action="append", default=None,
        help="Peak-amplitude CSV(s). Pass multiple times to combine datasets.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("experiments/runs/manuscript_figure2_peak_stats_prom05_prev_dip"),
    )
    parser.add_argument("--show-n", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Show per-bin n= annotations below the x-axis "
                             "(default: hidden)")
    parser.add_argument("--no-pool", dest="pool", action="store_false",
                        help="Plot each dataset separately instead of pooling")
    parser.add_argument("--amp-col", type=str,
                        default="peak_amplitude_prev_dip_sigma",
                        help="Amplitude column name in the peak CSV")
    parser.add_argument("--signal", type=str, default="Cortisol",
                        help="Signal name to keep from the peak CSV (e.g. Cortisol, ACTH)")
    parser.add_argument("--no-trend-line", dest="connect_trend",
                        action="store_false",
                        help="Do not draw the per-bin trend line joining the means")
    parser.add_argument("--max-ipi-min", type=float, default=240.0,
                        help="Drop inter-peak intervals above this cutoff (min) as "
                             "unresolved merges; <=0 disables (default 240)")
    parser.add_argument("--no-stats-text", dest="stats_text",
                        action="store_false",
                        help="Omit in-panel Friedman/Kruskal text (kept in the caption); "
                             "significance brackets are still drawn")
    parser.set_defaults(pool=True, connect_trend=True, stats_text=True)
    args = parser.parse_args()

    csv_paths = args.csv or [DEFAULT_BLOOD_CSV]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "task": "build_figure2_peak_stats",
        "created_at": datetime.now(UTC).isoformat(),
        "csv_paths": [str(p) for p in csv_paths],
    }, indent=2))
    max_ipi = args.max_ipi_min if args.max_ipi_min and args.max_ipi_min > 0 else None
    png, pdf = build_figure(csv_paths, out_dir, show_n=args.show_n,
                            pool=args.pool, amp_col=args.amp_col,
                            signal=args.signal, connect_trend=args.connect_trend,
                            max_ipi_min=max_ipi, stats_text=args.stats_text)
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
