"""Command-line entrypoint for running simulations and fits."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "hpa_model-mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPLCONFIGDIR))

import pandas as pd

from .analysis.plotting.summary import build_run_readme, plot_dual_fit, plot_simulation_replicates, summarize_trajectory_frame
from .config import dump_yaml, load_config
from .fit.habs_unified_circadian_fit import fit_pooled_circadian_input_from_config
from .fit.habs_dual_peak_stats import fit_habs_dual_peak_stats_from_config
from .fit.habs_no_delay_stochastic_peak_stats import fit_habs_no_delay_stochastic_peak_stats_from_config
from .fit.ultradian_psd import fit_ultradian_psd_from_config
from .model.three_state_gr_delay import ThreeStateGRDelayModel, build_drive
from .simulate.engine import aggregate_replicates, simulate_replicates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HPA model simulations and fits.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--out", required=True, help="Output run directory.")
    return parser.parse_args()


def _setup_run_dir(run_dir: Path) -> dict[str, Path]:
    paths = {
        "root": run_dir,
        "artifacts": run_dir / "artifacts",
        "figures": run_dir / "figures",
        "logs": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _write_run_readme(path: Path, task: str, highlights: list[str]) -> None:
    path.write_text(build_run_readme(task, highlights))


def _build_model(config: dict[str, Any]) -> ThreeStateGRDelayModel:
    params = config["model"]["params"]
    return ThreeStateGRDelayModel(
        a1=float(params["a1"]),
        a2=float(params["a2"]),
        a3=float(params["a3"]),
        b1=float(params["b1"]),
        b2=float(params["b2"]),
        b3=float(params["b3"]),
        kgr=float(params["kgr"]),
        tau_min=float(params["tau_min"]),
        x3_floor=float(params["x3_floor"]),
        hill_coeff=float(params["hill_coeff"]),
        initial_state=tuple(float(x) for x in params["initial_state"]),
    )


def _run_simulate(config: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    model = _build_model(config)
    drive = build_drive(config["drive"]["kind"], config["drive"]["params"])
    solver = config["solver"]
    n_reps = int(config["runtime"].get("n_reps", 1))
    seed = int(config["runtime"]["seed"])

    replicates = simulate_replicates(
        model,
        drive,
        dt_min=float(solver["dt_min"]),
        warmup_min=float(solver["warmup_min"]),
        duration_min=float(solver["duration_min"]),
        n_reps=n_reps,
        seed=seed,
    )
    summary = aggregate_replicates(replicates)
    per_rep_summary = pd.DataFrame(
        [summarize_trajectory_frame(rep_df) | {"rep": float(rep)} for rep, rep_df in replicates.groupby("rep")]
    )

    replicates.to_csv(paths["artifacts"] / "trajectory_replicates.csv", index=False)
    summary.to_csv(paths["artifacts"] / "trajectory_summary.csv", index=False)
    per_rep_summary.to_csv(paths["artifacts"] / "replicate_metrics.csv", index=False)
    plot_simulation_replicates(replicates, paths["figures"] / "simulation_replicates.png")

    x3_peak = float(per_rep_summary["x3_peak"].max())
    return [
        f"Simulated {n_reps} replicate(s) with seed base {seed}.",
        f"Peak cortisol across replicates: {x3_peak:.3f}.",
        "Artifacts: trajectory_replicates.csv, trajectory_summary.csv, replicate_metrics.csv.",
    ]


def _run_fit_habs_dual_peak_stats(config: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    result = fit_habs_dual_peak_stats_from_config(config, paths["root"])
    metrics = result["metrics"]
    params = result["params"]
    noise_params = {name: float(value) for name, value in params.items() if name.startswith("epsilon")}
    if "epsilon" in noise_params:
        noise_summary = f"sigma/epsilon={noise_params['epsilon']:.3f}"
    elif noise_params:
        noise_summary = ", ".join(
            f"{name}={value:.3f}"
            for name, value in sorted(noise_params.items())
        )
    else:
        noise_summary = "no fitted epsilon terms"
    return [
        f"Peak-stat fit dataset: {config['dataset']['name']} ({config['dataset']['variant']}).",
        f"Fitted tau={params['tau_min']:.3f} min, kgr={params['kgr']:.3f}, {noise_summary}.",
        (
            "Artifacts: fit_summary.csv, fit_params.csv, peak_profile_comparison.csv, "
            "trajectory_comparison.csv, fitted_config.yaml."
        ),
        (
            f"Peak-stat RMSE ACTH amplitude={metrics.get('rmse_mean_amplitude_acth', 0.0):.4f}, "
            f"cortisol amplitude={metrics.get('rmse_mean_amplitude_cortisol', 0.0):.4f}."
        ),
    ]


def _run_fit_pooled_circadian_input(
    config: dict[str, Any],
    paths: dict[str, Path],
    logger: logging.Logger,
) -> list[str]:
    result = fit_pooled_circadian_input_from_config(config, paths["root"], logger=logger)
    theta = result["theta"]
    return [
        f"Pooled circadian-input fit across {result['n_subjects']} subjects "
        f"({config['dataset'].get('datasets', [config['dataset']['name']])}).",
        f"Best objective={result['objective_value']:.5f}; "
        f"harmonic_split={theta['harmonic_split']:.3f}, "
        f"phase24={theta['phase24']:.3f}, phase12={theta['phase12']:.3f}.",
        "Artifacts: pooled_circadian_params.csv, pooled_cortisol_curve_comparison.csv.",
        "Figure: pooled_circadian_fit.png.",
    ]


def _run_fit_habs_no_delay_stochastic_peak_stats(
    config: dict[str, Any],
    paths: dict[str, Path],
    logger: logging.Logger,
) -> list[str]:
    result = fit_habs_no_delay_stochastic_peak_stats_from_config(config, paths["root"], logger=logger)
    summary = result["summary_row"]
    params = result["params"]
    return [
        f"No-delay stochastic peak-stat fit dataset: {config['dataset']['name']} ({config['dataset']['variant']}).",
        (
            f"Fitted tau fixed at {params['tau_min']:.1f} min, kgr={params['kgr']:.3f}, "
            f"epsilon_x1={params.get('epsilon_x1', 0.0):.3f}, "
            f"epsilon_x2={params.get('epsilon_x2', 0.0):.3f}, "
            f"epsilon_x3={params.get('epsilon_x3', 0.0):.3f}."
        ),
        (
            f"Final stochastic objective={float(summary['objective_value']):.4f} "
            f"from {int(summary['final_n_reps'])} replicate(s)."
        ),
        (
            "Artifacts: fit_summary.csv, fit_params.csv, peak_profile_comparison.csv, "
            "trajectory_comparison.csv, fitted_config.yaml."
        ),
    ]


def _run_fit_ultradian_psd(
    config: dict[str, Any],
    paths: dict[str, Path],
    logger: logging.Logger,
) -> list[str]:
    result = fit_ultradian_psd_from_config(config, paths["root"], logger=logger)
    row = result["summary_row"]
    params = result["params"]
    per_signal = result["per_signal"]
    param_str = ", ".join(f"{name}={value:.3f}" for name, value in params.items())
    match_str = "; ".join(
        f"{s}: r={m['inband_pearson_r']:.2f}, peak data/model="
        f"{m['peak_period_data_min']:.0f}/{m['peak_period_model_min']:.0f} min"
        for s, m in per_signal.items()
    )
    return [
        f"Ultradian-band PSD fit ({row['psd_mode']}, signals={row['signals']}): "
        f"{config['dataset']['name']} ({config['dataset']['variant']}), "
        f"{int(row['n_subjects'])} subject(s) pooled, {int(row['n_reps'])} replicate(s).",
        f"Fitted {param_str}; objective={float(row['objective_value']):.4f}.",
        f"In-band match: {match_str}.",
        "Artifacts: fit_params.csv, fit_summary.csv, psd_comparison.csv, fitted_config.yaml.",
        "Figure: psd_data_vs_model.png.",
    ]


def _run_analyze(config: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    input_dir = Path(config["analysis"]["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    summary: dict[str, Any] = {"input_dir": str(input_dir)}

    fit_summary = input_dir / "artifacts" / "fit_summary.csv"
    trajectory_summary = input_dir / "artifacts" / "trajectory_summary.csv"
    trajectory_reps = input_dir / "artifacts" / "trajectory_replicates.csv"

    if fit_summary.exists():
        fit_df = pd.read_csv(fit_summary)
        summary["fit_summary"] = fit_df.iloc[0].to_dict()
    if trajectory_summary.exists():
        traj_summary_df = pd.read_csv(trajectory_summary)
        summary["trajectory_summary_head"] = traj_summary_df.head(10).to_dict(orient="records")
    if trajectory_reps.exists():
        reps_df = pd.read_csv(trajectory_reps)
        plot_simulation_replicates(reps_df, paths["figures"] / "analyzed_simulation.png")

    (paths["artifacts"] / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return [
        f"Analyzed run: {input_dir}.",
        "Artifacts: analysis_summary.json.",
        "Generated analyzed_simulation.png when trajectory replicates were available.",
    ]


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    out_dir = Path(args.out)

    config = load_config(config_path)
    paths = _setup_run_dir(out_dir)
    logger = _setup_logging(paths["logs"] / "run.log")

    (paths["root"] / "resolved_config.yaml").write_text(dump_yaml(config))
    manifest = {
        "task": config["task"],
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "run_dir": str(out_dir.resolve()),
        "python_version": platform.python_version(),
        "seed": int(config["runtime"]["seed"]),
        "git_commit": _git_commit(),
        "dataset": config["dataset"],
    }

    logger.info("Running task %s", config["task"])
    if config["task"] == "simulate":
        highlights = _run_simulate(config, paths)
    elif config["task"] == "fit_habs_dual_peak_stats":
        highlights = _run_fit_habs_dual_peak_stats(config, paths)
    elif config["task"] == "fit_pooled_circadian_input":
        highlights = _run_fit_pooled_circadian_input(config, paths, logger)
    elif config["task"] == "fit_habs_no_delay_stochastic_peak_stats":
        highlights = _run_fit_habs_no_delay_stochastic_peak_stats(config, paths, logger)
    elif config["task"] == "fit_ultradian_psd":
        highlights = _run_fit_ultradian_psd(config, paths, logger)
    elif config["task"] == "analyze_run":
        highlights = _run_analyze(config, paths)
    else:
        raise ValueError(f"Unsupported task: {config['task']}")

    _write_manifest(paths["root"] / "manifest.json", manifest)
    _write_run_readme(paths["root"] / "README.md", config["task"], highlights)
    logger.info("Completed task %s", config["task"])


if __name__ == "__main__":
    main()
