"""Three example z-scored ACTH/cortisol trajectories from the v6 fitted model.

Modelled on ``experiments/scripts/plot_3_example_trajectories.py`` but uses
the new global v6 fit (drive-only lognormal noise, in-phase Stage 1 drive,
no per-subject drives). Each "subject" panel is one stochastic realization;
overlay panel shows real HABS data on top for visual comparison.

Pipeline matches Figure 3: each realization uses the same global drive,
different seed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpa_model.data.registry import get_dataset_spec, load_dataset  # noqa: E402
from hpa_model.fit.habs_dual_peak_stats import _build_model  # noqa: E402
from hpa_model.model.three_state_gr_delay import build_drive  # noqa: E402
from hpa_model.plotting import setup_nature_style  # noqa: E402
from hpa_model.simulate.engine import simulate_trajectory  # noqa: E402


ACTH_COLOR     = "#8C8C8C"
CORTISOL_COLOR = "#1A1A1A"
DATA_COLOR     = "#222222"


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)] if arr.size else arr
    if arr.size == 0:
        return np.zeros_like(values, dtype=float)
    std = float(np.std(arr))
    if std <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (np.asarray(values, dtype=float) - float(np.mean(arr))) / std


def _load_habs_subject(series_id: int, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_min, acth, cortisol) for one HABS subject."""
    spec = get_dataset_spec("habs")
    df = load_dataset("habs", variant)
    sub = df[df[spec.id_col].astype(str) == str(series_id)]
    sub = sub[[spec.time_col, "ACTH", "Cortisol"]].dropna().sort_values(spec.time_col)
    return (
        sub[spec.time_col].to_numpy(dtype=float),
        sub["ACTH"].to_numpy(dtype=float),
        sub["Cortisol"].to_numpy(dtype=float),
    )


def _simulate_model(fit_dir: Path, seed: int) -> pd.DataFrame:
    cfg = yaml.safe_load((fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    model = _build_model(cfg)
    dp = {k: v for k, v in cfg["drive"]["params"].items()
          if k not in ("dataset", "series_id")}
    drive = build_drive(str(cfg["drive"]["kind"]), dp)
    solver = cfg["solver"]
    runtime = cfg.get("runtime", {}) or {}
    return simulate_trajectory(
        model, drive,
        dt_min=float(solver["dt_min"]),
        warmup_min=float(solver["warmup_min"]),
        duration_min=float(solver["duration_min"]),
        seed=int(seed),
        noise_locations=list(runtime.get("noise_locations", []) or []),
        noise_epsilons=dict(runtime.get("noise_epsilons", {}) or {}),
        noise_form=str(runtime.get("noise_form", "multiplicative")),
    )


def _render(*,
            subjects: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
            sims: list[pd.DataFrame],
            epsilon: float,
            noise_form: str,
            out_path: Path,
            show_data: bool,
            xlabel: str = "Time (h, shifted clock)") -> None:
    setup_nature_style()
    n = len(subjects)
    # Model and data are z-scored per series (within each trace), so the model
    # trajectory and the data points share a common within-subject Z-score scale
    # and can be compared directly on one y-axis.
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for col, ((sid, t_data, acth_data, cort_data), sim) in enumerate(zip(subjects, sims)):
        sim_t_hr = sim["time_min"].to_numpy() / 60.0
        ax = axes[col]
        ln_c, = ax.plot(sim_t_hr, _zscore(sim["x3"].to_numpy()),
                        color=CORTISOL_COLOR, lw=1.0,
                        label="model Cortisol (x3)")
        ln_a, = ax.plot(sim_t_hr, _zscore(sim["x2"].to_numpy()),
                        color=ACTH_COLOR, lw=1.0,
                        label="model ACTH (x2)")
        if show_data and t_data.size:
            obs_hr = t_data / 60.0
            sc_c = ax.scatter(obs_hr, _zscore(cort_data),
                              facecolors="none", edgecolors=CORTISOL_COLOR,
                              linewidths=0.8, s=18, zorder=3,
                              label="data Cortisol")
            sc_a = ax.scatter(obs_hr, _zscore(acth_data),
                              facecolors="none", edgecolors=ACTH_COLOR,
                              linewidths=0.8, s=18, zorder=3,
                              label="data ACTH")
        title = f"HABS ID {sid}" if show_data else f"realization {col + 1}"
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Z-score (within series)", fontsize=12)
        if col == 0:
            handles = [ln_c, ln_a]
            labels = ["model Cortisol (x3)", "model ACTH (x2)"]
            if show_data and t_data.size:
                handles += [sc_c, sc_a]
                labels += ["data Cortisol", "data ACTH"]
            ax.legend(handles, labels, frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fit-dir", type=Path,
        default=PROJECT_ROOT / "experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v6_lognormal_eps3",
        help="Stage 2 fit run dir with artifacts/fitted_config.yaml.",
    )
    ap.add_argument("--variant", type=str, default="shifted_12h")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 3, 5],
                    help="HABS subject IDs to overlay (3 default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/example_trajectories_v6_lognormal")
    args = ap.parse_args()

    cfg = yaml.safe_load((args.fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    epsilon = float(cfg["drive"]["params"].get("epsilon", 0.0))
    noise_form = str(cfg["drive"]["params"].get("noise_form", "multiplicative"))

    subjects = [(sid, *_load_habs_subject(sid, args.variant))
                for sid in args.subjects]
    sims = [_simulate_model(args.fit_dir, seed=args.seed + i * 1000)
            for i in range(len(subjects))]

    xlabel = ("Time (h, clock)" if args.variant.startswith("known_clock")
              else "Time (h, shifted clock)")

    fig_dir = args.out_dir / "figures"
    art_dir = args.out_dir / "artifacts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    out_with = fig_dir / "example_trajectories_3subjects_with_data.png"
    out_model = fig_dir / "example_trajectories_3subjects_model_only.png"
    _render(subjects=subjects, sims=sims, epsilon=epsilon,
            noise_form=noise_form, out_path=out_with, show_data=True, xlabel=xlabel)
    _render(subjects=subjects, sims=sims, epsilon=epsilon,
            noise_form=noise_form, out_path=out_model, show_data=False, xlabel=xlabel)

    # Dump combined trajectories CSV
    rows: list[dict] = []
    for (sid, _, _, _), sim in zip(subjects, sims):
        for t, x2, x3 in zip(sim["time_min"], sim["x2"], sim["x3"]):
            rows.append({"subject_id": int(sid), "time_min": float(t),
                         "x2_acth": float(x2), "x3_cortisol": float(x3)})
    pd.DataFrame(rows).to_csv(art_dir / "trajectories.csv", index=False)

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "fit_dir": str(args.fit_dir),
        "variant": args.variant,
        "subjects": list(args.subjects),
        "epsilon": epsilon,
        "drive_noise_form": noise_form,
        "base_seed": args.seed,
    }, indent=2))

    print(f"[ok] {out_with}")
    print(f"[ok] {out_model}")
    print(f"[ok] {art_dir / 'trajectories.csv'}")


if __name__ == "__main__":
    main()
