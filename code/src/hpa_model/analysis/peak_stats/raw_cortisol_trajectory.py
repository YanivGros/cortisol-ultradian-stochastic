"""Plot RAW cortisol trajectories (no z-scoring) for a fitted no-delay
stochastic peak-stat run.

Re-simulates each subject with the run's ``fitted_config.yaml`` and overlays
model ``x3`` (raw model units) against the observed raw Cortisol, per subject.
This complements the per-ID z-scored ``trajectory_comparison.png`` written by
the fit pipeline, which hides absolute scale and offset.

Usage:
    PYTHONPATH=src python -m hpa_model.analysis.peak_stats.raw_cortisol_trajectory \
        --run experiments/runs/<run_name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ...data.registry import get_dataset_spec, load_dataset
from ...fit.habs_dual_peak_stats import (
    _build_model,
    _config_runtime_noise_epsilons,
    _config_runtime_noise_locations,
)
from ...model.three_state_gr_delay import build_drive
from ...plotting import setup_nature_style
from ...simulate.engine import sample_trajectory, simulate_trajectory


def build_raw_frame(config: dict, *, seed: int) -> "tuple[list[int], dict]":
    """Return (series_ids, per_id dict) of raw observed/model cortisol."""
    spec = get_dataset_spec(str(config["dataset"]["name"]))
    frame = load_dataset(str(config["dataset"]["name"]), str(config["dataset"]["variant"]))
    solver = config["solver"]
    model = _build_model(config)

    out: dict[int, dict] = {}
    for series_id, group in frame.groupby(spec.id_col, sort=True):
        obs = group.sort_values(spec.time_col)
        obs_times = obs[spec.time_col].to_numpy(dtype=float)
        drive = build_drive(
            str(config["drive"]["kind"]),
            {**config["drive"]["params"], "series_id": int(series_id)},
        )
        traj = simulate_trajectory(
            model,
            drive,
            dt_min=float(solver["dt_min"]),
            warmup_min=float(solver["warmup_min"]),
            duration_min=float(solver["duration_min"]),
            seed=int(seed) + int(series_id),
            noise_locations=_config_runtime_noise_locations(config),
            noise_epsilons=_config_runtime_noise_epsilons(config),
            noise_form=str(config.get("runtime", {}).get("noise_form", "multiplicative")),
        )
        sampled = sample_trajectory(traj, obs_times)
        out[int(series_id)] = {
            "time_hr": obs_times / 60.0,
            "cort_obs": obs["Cortisol"].to_numpy(dtype=float),
            "cort_sim": sampled["x3"].to_numpy(dtype=float),
        }
    return sorted(out), out


def plot_raw(run_dir: Path, *, seed: int = 42) -> Path:
    config = yaml.safe_load((run_dir / "artifacts" / "fitted_config.yaml").read_text())
    ids, data = build_raw_frame(config, seed=seed)

    setup_nature_style()
    n = len(ids)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 2.0 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    twins = []
    for i, sid in enumerate(ids):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on")
        d = data[sid]
        # Observed raw cortisol on the left axis (physical units).
        l_obs, = ax.plot(d["time_hr"], d["cort_obs"], color="#1f5fbf", lw=1.2,
                         marker="o", ms=2.5, label="Observed cortisol (raw)")
        # Model x3 in its own raw (dimensionless) units on a twin right axis,
        # since the two scales differ by ~100x. No z-scoring of either signal.
        ax2 = ax.twinx()
        twins.append(ax2)
        l_sim, = ax2.plot(d["time_hr"], d["cort_sim"], color="#d62728", lw=1.2,
                          label="Model x3 (raw)")
        ax.set_title(f"ID {sid}", fontsize=9)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.tick_params(axis="y", colors="#1f5fbf")
        ax2.tick_params(axis="y", colors="#d62728")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Observed cortisol", color="#1f5fbf", fontsize=8)
        ax2.set_ylabel("Model x3 (raw)", color="#d62728", fontsize=8)
    fig.legend([l_obs, l_sim], ["Observed cortisol (raw)", "Model x3 (raw)"],
               loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Raw cortisol: model x3 (right axis) vs observed (left axis) — no z-scoring", y=1.0)
    fig.tight_layout()

    out_path = run_dir / "figures" / "raw_cortisol_trajectory_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Run directory with artifacts/fitted_config.yaml")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    out = plot_raw(args.run, seed=args.seed)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
