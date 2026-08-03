# The stochastic nature of cortisol ultradian rhythms

Model, analysis, and figure-generation code for the paper *"The stochastic nature
of cortisol ultradian rhythms"* (Grosskopf, Danan, Pidham, Doenyas-Barak, Mayo &
Alon). This repository regenerates every **main-text** figure of the paper from a
shipped model checkpoint plus the published source recordings.

> **This repository contains code only.** The human hormone recordings were
> published by other groups and are not redistributed here. See **[DATA.md](DATA.md)**
> for where to obtain each dataset (the primary one is openly deposited at
> [doi:10.18710/5TW8YF](https://doi.org/10.18710/5TW8YF)) and exactly where to
> place it. `run_all.py` will not run until the data catalog is in place.

> **SI figures are out of scope** (this covers the main text only).
> The paper has nine displayed figures; **Fig 3 is a hand-drawn model schematic**
> that is not produced by any script. The other **eight figures are all generated
> here.**

## Quick start
```bash
pip install -r requirements.txt          # numpy scipy pandas matplotlib pyyaml seaborn
# assemble the data catalog first — see DATA.md
python run_all.py                         # all eight figures  ->  output/figures/
python run_all.py --only fig5 fig6        # a subset (prerequisites added automatically)
```
Pure Python >= 3.10; no `pip install -e .`, no compilation. The model figures
(Fig 5, 6, 9) run hundreds of stochastic simulations and take a few minutes each;
Fig 9 reuses a cached Walker fit, so it does not re-optimise. Expect ~15-25 min
for the full run; PNG + PDF land in `output/figures/`, named `Fig1` ... `Fig9`.

With the catalog in place the run ends by self-checking the pooled pulse count
against the paper: `cortisol peaks n = 518 (expect 518)`.

## Citation
If you use this code, please cite the paper. When using any of the hormone
recordings, cite the original study that produced them (see [DATA.md](DATA.md)).

## License
Code is released under the [MIT License](LICENSE). The license covers the code
only, not the third-party datasets.

## The canonical model
All model figures simulate forward from one shipped checkpoint,
`code/checkpoint/artifacts/fitted_config.yaml`:

- CRH t½ = 5 min, **ACTH t½ = 20 min**, **cortisol t½ = 15 min** (kinetic symmetry a_i = b_i)
- k_GR = 5, τ = 0, Hill = 3
- two-harmonic circadian drive (24 h + 12 h) from the pooled Stage-1 fit, baseline = 1.0
- **lognormal drive noise ε = 1.5**, mark-selected (swept and chosen to jointly
  match the data's pulse statistics — Rayleigh peak-amplitude CV ≈ 0.523, the
  full-signal CV, the inter-peak-interval distribution and the pulse rate — *not*
  an optimizer output). The model is **not re-fit** by this bundle.

## Figures produced (`output/figures/`)

| Display | File | Generated as | Content | Script |
|---|---|---|---|---|
| **Fig 1** | `Fig1` | `fig1` | Peak analysis of longitudinal data + Rayleigh amplitude distribution | `build_figure1_abc.py` |
| **Fig 2** | `Fig2` | `fig2` | Per-bin cortisol amplitude/timing statistics (amplitude noisier than timing) | `build_figure2_peak_stats.py` |
| Fig 3 | — | — | Model schematic (hand-drawn; **not generated**) | — |
| **Fig 4** | `Fig4` | `fig3` | Model with noise reproduces the stochastic pulses (example traces + per-bin stats vs data) | `build_figure3_combined.py` |
| **Fig 5** | `Fig5` | `fig5` | Inter-peak timing set by clearance rates, amplitude by production rates | `build_sensitivity_ipi_amplitude.py` |
| **Fig 6** | `Fig6` | `fig6` | Sensitivity of amplitude and timing to GR feedback strength k_GR | `build_figure6_kgr_sweep.py` |
| **Fig 7** | `Fig7` | `fig7` | A pulsatile cue entrains the pulses; maximal amplitude at a 2-h cue | `build_figure5_ABE.py` |
| **Fig 8** | `Fig8` | `fig8` | Model reproduces interstitial (microdialysis) cortisol statistics | `build_figure_microdialysis.py` |
| **Fig 9** | `Fig9` | `fig4` | A limit cycle with noise reproduces the data's pulse statistics | `build_walker_lc_combined_figure.py` |

> **Why the "Generated as" column differs from the display number.** The internal
> script names are historical: a model schematic was inserted as Fig 3 and the
> limit-cycle figure was moved to the end, so what the scripts call `fig3` is
> displayed as **Fig 4** and `fig4` as **Fig 9**. This runner names its collected
> outputs by **display number** (`Fig1` ... `Fig9`) to match what a reader sees,
> so the files in `output/figures/` map directly onto the paper.

## Pipeline
```
code/data/catalog/datasets/<habs|all_digitized|digitize_2019>/shifted_12h/data_shifted.csv
      │  peak detection: per-subject 2-harmonic baseline → z-score residual
      │  → scipy.signal.find_peaks (prominence 0.5σ, 60-min min distance)
      │  → amplitude = peak − previous trough; IPI = time to next peak (240-min cutoff)
      ▼
output/peaks_cortisol_prom05/...      ──►  Fig 1, Fig 2  (data figures)
                                      └─►  Fig 4, Fig 9   (data reference overlaid on the model)

code/checkpoint/artifacts/fitted_config.yaml  (ACTH t½=20, cort t½=15, k_GR=5, ε=1.5)
      │  forward simulation through the *same* peak-extraction pipeline
      ▼
Fig 4 (combined), Fig 5 (clearance/production), Fig 6 (k_GR), Fig 7 (entrainment),
Fig 8 (microdialysis), Fig 9 (limit cycle + noise; Walker fit reused from cache)
```

## Layout
```
├── run_all.py            # orchestrator — the only entry point
├── requirements.txt
├── DATA.md               # how to obtain the datasets and where to put them
├── code/
│   ├── src/hpa_model/    # the full model package
│   ├── scripts/          # the figure-builder + helper scripts
│   ├── data/             # NOT IN THIS REPO — you create it, see DATA.md
│   │   ├── catalog/datasets/<habs|all_digitized|digitize_2019|habs_microdialysis_cortisol>/
│   │   └── raw_data_input/Upton et al. (2023) blood.csv   # serum/MD pairs for Fig 8 diffusion fit
│   └── checkpoint/
│       ├── artifacts/fitted_config.yaml   # the canonical model (ε=1.5 / ACTH t½=20 / cort 15)
│       └── walker_fit_cache.json          # cached Walker limit-cycle fit (Fig 9; avoids slow refit)
└── output/               # generated: all intermediates + output/figures/ (final Fig1..Fig9 panels)
```

## Notes
- Self-contained: the scripts add `code/src` and `code/scripts` to `sys.path` and
  resolve the data catalog relative to the model package, so nothing outside this
  directory is read.
- The pipeline uses the `shifted_12h` cortisol traces (per-subject 24 h acrophase
  aligned to 10:00). See [DATA.md](DATA.md) for how that alignment is derived from
  the `raw` variant, and for the shapes to verify your catalog against.
- `cma` / `emcee` are **not** required — they are only used for re-fitting, which
  this code does not do (the model and the Walker limit cycle ship as
  checkpoints). Pass `--refit` to `build_walker_lc_combined_figure.py` only if you
  want to re-run the Walker DE fit (then `cma`/`scipy` differential evolution apply).
- The model is a three-state delay-free ODE system (CRH → ACTH → cortisol) with
  GR-mediated feedback and lognormal noise on the circadian drive; the definition
  lives in `code/src/hpa_model/model/three_state_gr_delay.py` and the integrator in
  `code/src/hpa_model/simulate/engine.py`.
