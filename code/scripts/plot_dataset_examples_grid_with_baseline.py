"""Plot a 1x3 grid of example trajectories with two-harmonic baseline overlaid."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from hpa_model.data.registry import load_dataset, load_shift_params, get_dataset_spec
from hpa_model.plotting import setup_nature_style, apply_paper_style
from hpa_model.data.two_harmonic_shift import evaluate_two_harmonic


def _clock_xy(time_min, values):
    """Map each sample to its clock-of-day hour (mod 24) and sort by clock, so
    the trace runs monotonically 0 -> 24 as one representative day with no
    break and no backward seam across midnight."""
    t = np.asarray(time_min, dtype=float)
    y = np.asarray(values, dtype=float)
    clock = (t % 1440.0) / 60.0
    order = np.argsort(clock)
    return clock[order], y[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/runs/manuscript_figures_two_harmonic"))
    parser.add_argument("--variant", type=str, default="shifted")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Omit the two-harmonic baseline overlay and use suffix '_no_baseline' on output")
    args = parser.parse_args()

    setup_nature_style()
    
    run_dir = args.run_dir
    fig_dir = run_dir / "figures/figure_1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    CORTISOL_COLOR = "#2F5C85"  # Data blue (cortisol)
    ACTH_COLOR = "#A1B6C8"      # Lighter blue (ACTH)
    BASELINE_COLOR = "#222222"  # Dark dashed circadian baseline
    
    # Inferred units based on literature and raw values:
    # HABS: nmol/L (Cortisol), pmol/L (ACTH)
    # digitize_2019 (Russell & Lightman): nmol/L (Cortisol), pg/mL (ACTH)
    # all_digitized (Young et al.): µg/dL (Cortisol)
    
    UNIT_MAP = {
        "habs": {"Cortisol": "nmol/L", "ACTH": "pmol/L"},
        "digitize_2019": {"Cortisol": "nmol/L", "ACTH": "pg/mL"},
        "all_digitized": {"Cortisol": "µg/dL"},
    }
    
    target_datasets = ["habs", "digitize_2019", "all_digitized"]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Mapping for cleaner labels
    DISPLAY_LABELS = {
        "habs": "Upton et al. (2023)",
        "digitize_2019": "Henley et al. (2009)",
        "all_digitized": "Young et al. (2004)"
    }
    
    # IDs to use for each dataset
    SELECTED_IDS = {
        "habs": "1",
        "digitize_2019": "wpd_datasets-2.csv",
        "all_digitized": 34
    }
    
    for i, ds_name in enumerate(target_datasets):
        ax1 = axes[i]
        spec = get_dataset_spec(ds_name)
        units = UNIT_MAP.get(ds_name, {})
        display_label = DISPLAY_LABELS.get(ds_name, spec.label)
        
        try:
            df = load_dataset(ds_name, args.variant)
            shift_params = load_shift_params(ds_name, args.variant)
        except FileNotFoundError:
            print(f"Skipping {ds_name}: variant '{args.variant}' not found.")
            continue
            
        series_id = SELECTED_IDS.get(ds_name, df[spec.id_col].unique()[0])
        subset = df[df[spec.id_col].astype(str) == str(series_id)].copy()
        subset["time_hr"] = subset[spec.time_col].to_numpy(dtype=float) / 60.0
        
        cort_spec = next((s for s in spec.signals if s.name == "Cortisol"), None)
        acth_spec = next((s for s in spec.signals if s.name == "ACTH"), None)
        
        t_dense = np.linspace(0, 1440, 500)
        
        # Plot Cortisol
        if cort_spec:
            c_unit = units.get("Cortisol", "units")
            cx, cy = _clock_xy(subset[spec.time_col], subset[cort_spec.column])
            ax1.plot(
                cx,
                cy,
                color=CORTISOL_COLOR,
                linewidth=1.2,
                label=f"Cortisol ({c_unit})"
            )
            row_matches = shift_params[shift_params[spec.id_col].astype(str) == str(series_id)]
            if not row_matches.empty and not args.no_baseline:
                row = row_matches.iloc[0]
                period_min = float(row.get("period_min", 1440.0))
                second_period_min = float(row.get("second_period_min", 720.0))
                phase24 = float(row["phase24"])
                phase12 = float(row["phase12"])
                if args.variant != "raw":
                    applied_shift = float(row.get("applied_shift_min", 0.0))
                    w24 = 2.0 * np.pi / period_min
                    w12 = 2.0 * np.pi / second_period_min
                    phase24 = float((phase24 - w24 * applied_shift) % (2.0 * np.pi))
                    phase12 = float((phase12 - w12 * applied_shift) % (2.0 * np.pi))
                baseline_params = {
                    "a24": float(row["a24"]),
                    "phase24": phase24,
                    "a12": float(row["a12"]),
                    "phase12": phase12,
                    "c": float(row["c"]),
                    "period_min": period_min,
                    "second_period_min": second_period_min,
                }
                baseline_curve = evaluate_two_harmonic(t_dense, baseline_params)
                ax1.plot(
                    t_dense / 60.0,
                    baseline_curve,
                    color=BASELINE_COLOR,
                    linewidth=1.1,
                    linestyle="--",
                    label="Fitted circadian",
                )
            
            ax1.set_ylabel(f"Cortisol ({c_unit})", color=CORTISOL_COLOR)
            ax1.tick_params(axis='y', labelcolor=CORTISOL_COLOR)
            
        # Plot ACTH on right axis if available
        if acth_spec:
            a_unit = units.get("ACTH", "units")
            ax2 = ax1.twinx()
            ax_x, ax_y = _clock_xy(subset[spec.time_col], subset[acth_spec.column])
            ax2.plot(
                ax_x,
                ax_y,
                color=ACTH_COLOR,
                linewidth=1.2,
                label=f"ACTH ({a_unit})"
            )
            
            ax2.set_ylabel(f"ACTH ({a_unit})", color=ACTH_COLOR)
            ax2.tick_params(axis='y', labelcolor=ACTH_COLOR)
            
            # Match nature style for twin axes
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(True)
            ax2.spines["left"].set_visible(False)
            ax2.grid(False)
            
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
        else:
            apply_paper_style(ax1)
            
        ax1.set_xlabel("Clock time of day (h)")
        ax1.set_title(f"{display_label}", fontsize=9.5)
        ax1.set_xlim(0, 24)
        ax1.set_xticks([0, 4, 8, 12, 16, 20, 24])

    # Add a global legend
    legend_elements = [
        Line2D([0], [0], color=CORTISOL_COLOR, lw=1.5, label='Cortisol'),
        Line2D([0], [0], color=ACTH_COLOR, lw=1.5, label='ACTH'),
    ]
    if not args.no_baseline:
        legend_elements.append(
            Line2D([0], [0], color=BASELINE_COLOR, lw=1.3, linestyle="--", label='Fitted circadian'),
        )
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05)
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    
    suffix = "_no_baseline" if args.no_baseline else ""
    save_stem = fig_dir / f"figure_1a_individual_trajectories{suffix}"
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_stem}.png")

if __name__ == "__main__":
    main()
