#!/usr/bin/env python3
"""Regenerate every main-text figure of
*The stochastic nature of cortisol ultradian rhythms* — self-contained.

This bundle ships everything needed: the model code (``code/src/hpa_model``),
the figure scripts (``code/scripts``), the data catalog (``code/data``) and the
fitted-model checkpoint (``code/checkpoint``). No other repo files are required.

Pipeline
--------
    data catalog (shifted_12h cortisol/ACTH traces)
      -> peak detection (2-harmonic baseline, z-score, find_peaks 0.5 sigma)
      -> data figures (Fig 1, Fig 2)
    fitted checkpoint (ACTH t1/2 = 20 min, cortisol t1/2 = 15 min, k_GR = 5,
                       lognormal drive noise eps = 1.5)
      -> model figures (Fig 4, 5, 6, 7, 8, 9), each through the same
         peak-extraction pipeline applied to simulated output.

The paper has nine displayed figures. **Fig 3 is a hand-drawn model schematic**
(``docs/manuscript_figures/fig_model_schematic.png``) and is not produced by any
script, so it is the one main figure this runner does not generate. The other
eight are generated here. (SI figures are out of scope.)

Display figure  ->  manuscript file  ->  script
    Fig 1   fig1   build_figure1_abc.py             (peak analysis + Rayleigh)
    Fig 2   fig2   build_figure2_peak_stats.py      (per-bin amplitude/timing)
    Fig 3   --     (hand-drawn schematic; not generated)
    Fig 4   fig3   build_figure3_combined.py        (model+noise vs data)
    Fig 5   fig5   build_sensitivity_ipi_amplitude.py (clearance/production)
    Fig 6   fig6   build_figure6_kgr_sweep.py       (GR feedback strength)
    Fig 7   fig7   build_figure5_ABE.py             (pulsatile-cue entrainment)
    Fig 8   fig8   build_figure_microdialysis.py    (interstitial cortisol)
    Fig 9   fig4   build_walker_lc_combined_figure.py (limit cycle +/- noise)

(The manuscript file names are historical: fig3.* is displayed as Fig 4 and
fig4.* as Fig 9. This runner names its collected outputs by *display* number.)

Run
---
    python run_all.py                 # all eight figures
    python run_all.py --only fig6     # one figure (names: fig1 fig2 fig4 fig5 fig6 fig7 fig8 fig9)
    python run_all.py --only fig1 fig2  # a subset; prerequisites are added automatically

Outputs land in ``output/<step>/...``; the final panels are collected (PNG+PDF)
into ``output/figures/`` named ``Fig1`` ... ``Fig9``.

Needs: numpy scipy pandas matplotlib pyyaml seaborn (see requirements.txt).
The model figures (Fig 5, 6, 9) run hundreds of stochastic simulations and take
a few minutes each; Fig 9 reuses a cached Walker fit so it does not re-optimise.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
CODE = BUNDLE / "code"
SCRIPTS = CODE / "scripts"
OUT = BUNDLE / "output"
CKPT = CODE / "checkpoint"                                  # --fit-dir (has artifacts/fitted_config.yaml)
CKPT_CFG = CKPT / "artifacts" / "fitted_config.yaml"        # --config (microdialysis)
WALKER_CACHE = CKPT / "walker_fit_cache.json"               # --fit-cache (limit cycle; avoids slow refit)
PY = sys.executable

ENV = dict(os.environ, PYTHONPATH=str(CODE / "src"))
PEAKS_CORT = OUT / "peaks_cortisol_prom05" / "artifacts" / "peak_amplitude_samples.csv"
FIG1A = OUT / "fig1a" / "figures" / "figure_1" / "figure_1a_individual_trajectories_no_baseline.png"


def run(script: str, *args) -> None:
    cmd = [PY, str(SCRIPTS / script), *map(str, args)]
    print(f"\n$ {script} {' '.join(map(str, args))}", flush=True)
    subprocess.run(cmd, cwd=CODE, env=ENV, check=True)


def collect(src: Path, dest_stem: str) -> None:
    """Copy a produced figure (PNG+PDF) into output/figures/ as Fig<N>."""
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, figs / (dest_stem + src.suffix))
        print(f"  collected -> figures/{dest_stem}{src.suffix}")
    else:
        print(f"  [WARN] missing expected output: {src}")


def both(src_stem_dir: Path, base: str, dest_stem: str) -> None:
    for ext in ("png", "pdf"):
        collect(src_stem_dir / f"{base}.{ext}", dest_stem)


# ---------------------------------------------------------------- prerequisites
def step_peaks():
    """Pooled cortisol peak table (Figs 1, 2, 4, 9)."""
    run("plot_peak_amplitude_direct_rayleigh_fit.py",
        "--prom-sigma", 0.5, "--variant", "shifted_12h", "--no-drop-negative",
        "--out", OUT / "peaks_cortisol_prom05")


# -------------------------------------------------------------- data figures
def step_fig1():
    run("plot_dataset_examples_grid_with_baseline.py",
        "--run-dir", OUT / "fig1a", "--variant", "shifted_12h", "--no-baseline")
    run("build_figure1_abc.py",
        "--out", OUT / "figure1", "--panel-a-src", FIG1A,
        "--peaks-csv", PEAKS_CORT, "--pipeline-variant", "shifted_12h", "--no-stats-text")
    both(OUT / "figure1" / "figures" / "figure_1", "figure_1", "Fig1")


def step_fig2():
    run("build_figure2_peak_stats.py",
        "--csv", PEAKS_CORT, "--signal", "Cortisol", "--max-ipi-min", 240, "--no-stats-text",
        "--out", OUT / "figure2")
    both(OUT / "figure2" / "figures" / "figure_2", "figure_2", "Fig2")


# -------------------------------------------------------------- model figures
def step_fig4():  # manuscript fig3.* — model + noise reproduces the data
    run("build_figure3_combined.py",
        "--fit-dir", CKPT, "--peaks-csv", PEAKS_CORT,
        "--n-subjects", 71, "--n-reps", 1, "--max-ipi-min", 240,
        "--out", OUT / "figure4")
    both(OUT / "figure4" / "figures", "figure_3_combined", "Fig4")


def step_fig5():  # manuscript fig5.* — clearance sets timing, production sets amplitude
    run("build_sensitivity_ipi_amplitude.py",
        "--fit-dir", CKPT, "--factor-min", 0.5, "--factor-max", 2.0,
        "--n-points", 13, "--n-reps", 30,
        "--out", OUT / "figure5")
    both(OUT / "figure5" / "figures", "sensitivity_grid", "Fig5")


def step_fig6():  # manuscript fig6.* — sensitivity to GR feedback strength k_GR
    run("build_figure6_kgr_sweep.py",
        "--fit-dir", CKPT, "--kgr-min", 2, "--kgr-max", 10,
        "--n-points", 33, "--n-reps", 400, "--min-distance-min", 60,
        "--out", OUT / "figure6")
    both(OUT / "figure6" / "figures", "figure_6", "Fig6")


def step_fig7():  # manuscript fig7.* — pulsatile-cue entrainment
    run("build_figure5_ABE.py",
        "--fit-dir", CKPT, "--cue-mode", "additive",
        "--stim-amplitude", 4, "--stim-period-min", 120,
        "--out", OUT / "figure7")
    both(OUT / "figure7" / "figures", "figure_5_ABE", "Fig7")


def step_fig8():  # manuscript fig8.* — interstitial (microdialysis) cortisol
    run("build_figure_microdialysis.py",
        "--config", CKPT_CFG, "--out", OUT / "figure8")
    both(OUT / "figure8" / "figures", "figure_microdialysis", "Fig8")


def step_fig9():  # manuscript fig4.* — Walker limit cycle +/- noise
    run("build_walker_lc_combined_figure.py",
        "--fit-dir", CKPT, "--peaks-csv", PEAKS_CORT, "--fit-cache", WALKER_CACHE,
        "--walker-noise-eps", 1.5, "--lc-n-days", 120, "--n-reps", 200,
        "--data-example-id", 1, "--noisy-traj-seed", 9,
        "--out", OUT / "figure9")
    both(OUT / "figure9" / "figures", "walker_lc_combined_figure", "Fig9")


# ordered; data + combined + limit-cycle figures share the cortisol peaks step
STEPS = {
    "peaks": step_peaks,
    "fig1": step_fig1,
    "fig2": step_fig2,
    "fig4": step_fig4,
    "fig5": step_fig5,
    "fig6": step_fig6,
    "fig7": step_fig7,
    "fig8": step_fig8,
    "fig9": step_fig9,
}
NEED_PEAKS = {"fig1", "fig2", "fig4", "fig9"}


REQUIRED_DATA = (
    CODE / "data" / "catalog" / "datasets" / "habs" / "shifted_12h" / "data_shifted.csv",
    CODE / "data" / "catalog" / "datasets" / "all_digitized" / "shifted_12h" / "data_shifted.csv",
    CODE / "data" / "catalog" / "datasets" / "digitize_2019" / "shifted_12h" / "data_shifted.csv",
    CODE / "data" / "catalog" / "datasets" / "habs_microdialysis_cortisol" / "shifted" / "data_shifted.csv",
    CODE / "data" / "raw_data_input" / "Upton et al. (2023) blood.csv",
)


def preflight_data() -> None:
    """Fail early and legibly when the data catalog has not been assembled.

    This repository ships code only; the hormone recordings are third-party and
    must be obtained separately (see DATA.md).
    """
    missing = [p for p in REQUIRED_DATA if not p.exists()]
    if not missing:
        return
    print("ERROR: the data catalog is missing or incomplete.\n", file=sys.stderr)
    print("This repository contains code only. The human hormone recordings were", file=sys.stderr)
    print("published by other groups and are not redistributed here.\n", file=sys.stderr)
    print("Missing:", file=sys.stderr)
    for p in missing:
        print(f"  - {p.relative_to(BUNDLE)}", file=sys.stderr)
    print(f"\nSee {BUNDLE / 'DATA.md'} for where to obtain each dataset", file=sys.stderr)
    print("(the primary one is openly deposited at https://doi.org/10.18710/5TW8YF),", file=sys.stderr)
    print("the expected column schema, and the shapes to verify against.", file=sys.stderr)
    raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=[k for k in STEPS if k != "peaks"],
                    help="Run a subset of figures (default: all). The peak step is "
                         "added automatically when a figure needs it.")
    args = ap.parse_args()

    preflight_data()

    names = args.only or [k for k in STEPS if k != "peaks"]
    ordered = [s for s in STEPS if s in names]
    if any(n in NEED_PEAKS for n in ordered):
        ordered = ["peaks"] + ordered

    for name in ordered:
        STEPS[name]()

    print(f"\nDone. Collected figures in: {OUT / 'figures'}")
    if PEAKS_CORT.exists():
        import pandas as pd
        n = len(pd.read_csv(PEAKS_CORT))
        print(f"  cortisol peaks n = {n} (expect 518)")


if __name__ == "__main__":
    main()
