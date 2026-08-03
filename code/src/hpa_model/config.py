"""Configuration loading and validation helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .model.three_state_gr_delay import (
    DEFAULT_A1,
    DEFAULT_A2,
    DEFAULT_A3,
    DEFAULT_HILL_COEFF,
    DEFAULT_INITIAL_STATE,
    DEFAULT_KGR,
    DEFAULT_TAU_MIN,
    DEFAULT_X3_FLOOR,
)


DEFAULT_MODEL_PARAMS = {
    "a1": DEFAULT_A1,
    "a2": DEFAULT_A2,
    "a3": DEFAULT_A3,
    "b1": DEFAULT_A1,
    "b2": DEFAULT_A2,
    "b3": DEFAULT_A3,
    "kgr": DEFAULT_KGR,
    "tau_min": DEFAULT_TAU_MIN,
    "x3_floor": DEFAULT_X3_FLOOR,
    "hill_coeff": DEFAULT_HILL_COEFF,
    "initial_state": list(DEFAULT_INITIAL_STATE),
}

DEFAULT_DRIVE_PARAMS = {
    "level": 1.0,
    "baseline": 1.0,
    "amplitude": 0.0,
    "phase_min": 0.0,
    "period_min": 1440.0,
    "epsilon": 0.0,
}

DEFAULT_SOLVER = {
    "dt_min": 10.0,
    "warmup_min": 720.0,
    "duration_min": 1440.0,
}

DEFAULT_FIT = {
    "loss": {
        "normalize": "per_id_zscore",
        "mode": "mean_std",
        "std_weight": 0.5,
        "cv_weight": 0.0,
    },
    "min_drive_amplitude": 1.0,
    "bounds": {},
    "n_reps": 1,
    "max_nfev": 32,
}

DEFAULT_RUNTIME = {
    "seed": 123,
    "n_reps": 1,
}

DEFAULT_COMPARISON = {
    "bootstrap_reps": 500,
    "n_reps_per_damped_candidate": 8,
    "winner_margin_ratio": 1.10,
    "optimizer_maxiter": 6,
    "optimizer_popsize": 6,
}

DEFAULT_ANALYSIS = {
    "normalize": "per_id_zscore",
    "arrow_stride_min": 240.0,
    "bandpass_min_period_hours": 1.0,
    "bandpass_max_period_hours": 6.0,
    "filter_order": 2,
    "detrend": True,
    "edge_trim_hours": 2.0,
    "resample_dt_min": 20.0,
    "max_crosscorr_lag_hours": 6.0,
    "autocorr_min_period_hours": 1.0,
    "autocorr_max_period_hours": 6.0,
    "flow_taus_min": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
    "flow_grid_size": 14,
    "flow_bandwidth_scale": 0.45,
    "flow_min_effective_samples": 2.0,
    "flow_padding": 0.35,
}

DEFAULT_DAMPED_MODEL = {
    "bounds": {
        "period_hr": [1.0, 6.0],
        "decay_rate_per_hr": [0.05, 2.0],
        "sigma": [0.01, 3.0],
    }
}

DEFAULT_DELAY_MODEL = {
    "bounds": {
        "kgr": [0.5, 100.0],
        "tau_min": [0.0, 120.0],
        "drive_baseline": [0.5, 2.0],
    }
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")
    resolved = apply_defaults(data)
    validate_config(resolved)
    return resolved


def apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)
    resolved.setdefault("dataset", {})
    resolved.setdefault("model", {})
    resolved.setdefault("drive", {})
    resolved.setdefault("solver", {})
    resolved.setdefault("fit", {})
    resolved.setdefault("runtime", {})
    resolved.setdefault("comparison", {})
    resolved.setdefault("analysis", {})
    resolved.setdefault("damped_model", {})
    resolved.setdefault("delay_model", {})

    model = resolved["model"]
    model.setdefault("params", {})
    model.setdefault("free_params", [])
    for key, value in DEFAULT_MODEL_PARAMS.items():
        model["params"].setdefault(key, deepcopy(value))

    drive = resolved["drive"]
    drive.setdefault("kind", "sine_noise")
    drive.setdefault("params", {})
    for key, value in DEFAULT_DRIVE_PARAMS.items():
        drive["params"].setdefault(key, deepcopy(value))

    solver = resolved["solver"]
    for key, value in DEFAULT_SOLVER.items():
        solver.setdefault(key, value)

    fit = resolved["fit"]
    fit.setdefault("loss", {})
    fit.setdefault("bounds", {})
    for key, value in DEFAULT_FIT["loss"].items():
        fit["loss"].setdefault(key, value)
    fit.setdefault("min_drive_amplitude", DEFAULT_FIT["min_drive_amplitude"])
    fit.setdefault("n_reps", DEFAULT_FIT["n_reps"])
    fit.setdefault("max_nfev", DEFAULT_FIT["max_nfev"])

    runtime = resolved["runtime"]
    for key, value in DEFAULT_RUNTIME.items():
        runtime.setdefault(key, value)

    comparison = resolved["comparison"]
    for key, value in DEFAULT_COMPARISON.items():
        comparison.setdefault(key, deepcopy(value))

    analysis = resolved["analysis"]
    for key, value in DEFAULT_ANALYSIS.items():
        analysis.setdefault(key, deepcopy(value))

    damped_model = resolved["damped_model"]
    damped_model.setdefault("bounds", {})
    for key, value in DEFAULT_DAMPED_MODEL["bounds"].items():
        damped_model["bounds"].setdefault(key, deepcopy(value))

    delay_model = resolved["delay_model"]
    delay_model.setdefault("bounds", {})
    for key, value in DEFAULT_DELAY_MODEL["bounds"].items():
        delay_model["bounds"].setdefault(key, deepcopy(value))

    return resolved


def validate_config(config: dict[str, Any]) -> None:
    required = ["task", "dataset", "model", "drive", "solver", "fit", "runtime"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    task = config["task"]
    if task not in {
        "simulate",
        "fit_habs_dual",
        "fit_habs_dual_peak_stats",
        "fit_habs_no_delay_stochastic_peak_stats",
        "fit_ultradian_psd",
        "fit_habs_circadian_input_per_id",
        "fit_pooled_circadian_input",
        "fit_habs_drive_noise_to_peak_stats",
        "fit_habs_population_steady_state",
        "compare_habs_multisite_production_noise",
        "compare_joint_dataset_noise",
        "fit_joint_dataset_multistart",
        "analyze_run",
        "plot_joint_model_data_phase_comparison",
        "plot_habs_phase_coherence",
        "compare_habs_ultradian_origin_models",
    }:
        raise ValueError(f"Unsupported task: {task}")

    dataset = config["dataset"]
    if "name" not in dataset or "variant" not in dataset:
        raise ValueError("dataset.name and dataset.variant are required")

    model_params = config["model"]["params"]
    initial_state = model_params.get("initial_state", [])
    if len(initial_state) != 3:
        raise ValueError("model.params.initial_state must have length 3")

    if task == "analyze_run":
        analysis = config.get("analysis", {})
        if "input_dir" not in analysis:
            raise ValueError("analysis.input_dir is required for analyze_run")

    if task == "plot_joint_model_data_phase_comparison":
        if dataset["name"] != "joint_habs_digitize" or dataset["variant"] != "shifted":
            raise ValueError("plot_joint_model_data_phase_comparison requires dataset joint_habs_digitize/shifted")
        analysis = config.get("analysis", {})
        if "source_run_dir" not in analysis:
            raise ValueError("analysis.source_run_dir is required for plot_joint_model_data_phase_comparison")
        normalize = str(analysis.get("normalize", "per_id_zscore"))
        if normalize not in {"per_id_zscore", "raw"}:
            raise ValueError(
                "plot_joint_model_data_phase_comparison analysis.normalize must be per_id_zscore or raw"
            )
        if float(analysis.get("arrow_stride_min", 240.0)) <= 0.0:
            raise ValueError(
                "plot_joint_model_data_phase_comparison analysis.arrow_stride_min must be positive"
            )

    if task == "plot_habs_phase_coherence":
        if dataset["name"] != "habs" or dataset["variant"] != "shifted":
            raise ValueError("plot_habs_phase_coherence currently requires dataset habs/shifted")
        analysis = config.get("analysis", {})
        normalize = str(analysis.get("normalize", "per_id_zscore"))
        if normalize not in {"per_id_zscore", "raw"}:
            raise ValueError("plot_habs_phase_coherence analysis.normalize must be per_id_zscore or raw")
        min_period = float(analysis.get("bandpass_min_period_hours", 1.0))
        max_period = float(analysis.get("bandpass_max_period_hours", 6.0))
        if min_period <= 0.0 or max_period <= 0.0 or max_period <= min_period:
            raise ValueError(
                "plot_habs_phase_coherence requires positive bandpass periods with max > min"
            )
        if int(analysis.get("filter_order", 2)) < 1:
            raise ValueError("plot_habs_phase_coherence analysis.filter_order must be >= 1")
        if float(analysis.get("edge_trim_hours", 2.0)) < 0.0:
            raise ValueError("plot_habs_phase_coherence analysis.edge_trim_hours must be non-negative")

    if task == "compare_habs_ultradian_origin_models":
        if dataset["name"] != "habs" or dataset["variant"] != "shifted":
            raise ValueError("compare_habs_ultradian_origin_models requires dataset habs/shifted")
        analysis = config.get("analysis", {})
        normalize = str(analysis.get("normalize", "per_id_zscore"))
        if normalize not in {"per_id_zscore", "raw"}:
            raise ValueError("compare_habs_ultradian_origin_models analysis.normalize must be per_id_zscore or raw")
        min_period = float(analysis.get("bandpass_min_period_hours", 1.0))
        max_period = float(analysis.get("bandpass_max_period_hours", 6.0))
        if min_period <= 0.0 or max_period <= 0.0 or max_period <= min_period:
            raise ValueError("compare_habs_ultradian_origin_models requires positive bandpass periods with max > min")
        damped_bounds = config.get("damped_model", {}).get("bounds", {})
        for key in ("period_hr", "decay_rate_per_hr", "sigma"):
            if key not in damped_bounds:
                raise ValueError(f"compare_habs_ultradian_origin_models requires damped_model.bounds.{key}")
        delay_bounds = config.get("delay_model", {}).get("bounds", {})
        for key in ("kgr", "tau_min", "drive_baseline"):
            if key not in delay_bounds:
                raise ValueError(f"compare_habs_ultradian_origin_models requires delay_model.bounds.{key}")
        comparison = config.get("comparison", {})
        if int(comparison.get("bootstrap_reps", 0)) < 1:
            raise ValueError("compare_habs_ultradian_origin_models comparison.bootstrap_reps must be >= 1")
        if int(comparison.get("n_reps_per_damped_candidate", 0)) < 1:
            raise ValueError("compare_habs_ultradian_origin_models comparison.n_reps_per_damped_candidate must be >= 1")
        if float(comparison.get("winner_margin_ratio", 0.0)) < 1.0:
            raise ValueError("compare_habs_ultradian_origin_models comparison.winner_margin_ratio must be >= 1.0")

    loss_mode = str(config["fit"]["loss"].get("mode", "mean_std"))
    if task == "fit_habs_dual":
        if loss_mode not in {"mean_std", "trajectory_mse"}:
            raise ValueError(f"Unsupported fit.loss.mode: {loss_mode}")

    if task == "fit_habs_dual_peak_stats":
        if dataset["name"] != "habs" or dataset["variant"] != "shifted":
            raise ValueError("fit_habs_dual_peak_stats requires dataset habs/shifted")
        drive_params = config["drive"]["params"]
        for key in ["dataset", "variant", "baseline"]:
            if key not in drive_params:
                raise ValueError(f"fit_habs_dual_peak_stats requires drive.params.{key}")
        free_params = list(config["model"].get("free_params", []))
        if "tau_min" not in free_params:
            raise ValueError("fit_habs_dual_peak_stats free params must include tau_min")
        free_set = set(free_params)
        drive_noise = free_set in ({"tau_min", "epsilon"}, {"kgr", "tau_min", "epsilon"}) and len(free_params) == len(free_set)
        secretion_noise_params = {"epsilon_x1", "epsilon_x2", "epsilon_x3"}
        secretion_noise = (
            free_set.issubset({"tau_min", "kgr", *secretion_noise_params})
            and len(free_set & secretion_noise_params) >= 1
            and "epsilon" not in free_set
        )
        if not drive_noise and not secretion_noise:
            raise ValueError(
                "fit_habs_dual_peak_stats requires either [tau_min, epsilon] or [kgr, tau_min, epsilon] "
                "or secretion-noise free params among epsilon_x1/epsilon_x2/epsilon_x3"
            )
        expected_drive_kind = "two_harmonic_noise" if drive_noise else "two_harmonic"
        if config["drive"]["kind"] != expected_drive_kind:
            raise ValueError(f"fit_habs_dual_peak_stats requires drive.kind={expected_drive_kind}")
        if loss_mode != "peak_stats":
            raise ValueError("fit_habs_dual_peak_stats requires fit.loss.mode=peak_stats")
        noise_form = str(config.get("runtime", {}).get("noise_form", "multiplicative"))
        if noise_form not in {"multiplicative", "additive", "lognormal"}:
            raise ValueError("fit_habs_dual_peak_stats runtime.noise_form must be multiplicative, additive, or lognormal")

    if task == "fit_habs_population_steady_state":
        if config["drive"]["kind"] != "constant":
            raise ValueError(
                "fit_habs_population_steady_state requires drive.kind=constant"
            )
        if float(model_params.get("tau_min", 0.0)) != 0.0:
            raise ValueError(
                "fit_habs_population_steady_state requires model.params.tau_min=0.0"
            )
        free_params = list(config["model"].get("free_params", []))
        if not free_params:
            raise ValueError(
                "fit_habs_population_steady_state requires non-empty model.free_params"
            )
        allowed = {"b1", "b2", "b3", "kgr"}
        invalid = sorted(set(free_params) - allowed)
        if invalid:
            raise ValueError(
                "fit_habs_population_steady_state unsupported free params: "
                f"{invalid}; allowed: {sorted(allowed)}"
            )

    if task == "fit_habs_circadian_input_per_id":
        if dataset["name"] != "habs":
            raise ValueError("fit_habs_circadian_input_per_id requires dataset.name=habs")
        if float(model_params.get("tau_min", 0.0)) != 0.0:
            raise ValueError("fit_habs_circadian_input_per_id requires model.params.tau_min=0.0")
        runtime_cfg = config.get("runtime", {})
        noise_epsilons = runtime_cfg.get("noise_epsilons", {}) or {}
        if any(float(value) != 0.0 for value in noise_epsilons.values()):
            raise ValueError("fit_habs_circadian_input_per_id requires runtime.noise_epsilons to all be 0.0")
        drive_epsilon = float(config["drive"]["params"].get("epsilon", 0.0))
        if drive_epsilon != 0.0:
            raise ValueError("fit_habs_circadian_input_per_id requires drive.params.epsilon=0.0")

    if task == "fit_habs_drive_noise_to_peak_stats":
        if dataset["name"] != "habs":
            raise ValueError("fit_habs_drive_noise_to_peak_stats requires dataset.name=habs")
        if float(model_params.get("tau_min", 0.0)) != 0.0:
            raise ValueError("fit_habs_drive_noise_to_peak_stats requires model.params.tau_min=0.0")
        if config["drive"]["kind"] != "two_harmonic_noise":
            raise ValueError("fit_habs_drive_noise_to_peak_stats requires drive.kind=two_harmonic_noise")
        if "circadian_params_path" not in config.get("fit", {}):
            raise ValueError("fit_habs_drive_noise_to_peak_stats requires fit.circadian_params_path")
        runtime_cfg = config.get("runtime", {})
        noise_epsilons = runtime_cfg.get("noise_epsilons", {}) or {}
        if any(float(value) != 0.0 for value in noise_epsilons.values()):
            raise ValueError(
                "fit_habs_drive_noise_to_peak_stats requires all runtime.noise_epsilons (secretion sites) to be 0.0"
            )
        bounds = config.get("fit", {}).get("bounds", {})
        if "epsilon" not in bounds:
            raise ValueError("fit_habs_drive_noise_to_peak_stats requires fit.bounds.epsilon")

    if task == "fit_habs_no_delay_stochastic_peak_stats":
        # Two drive modes are supported:
        #   per-subject: drive.params.dataset + series_id resolves the
        #     per-subject two-harmonic baseline (legacy default).
        #   global: explicit (a24, phase24, a12, phase12, baseline) literals in
        #     drive.params; same drive shape across all subjects/realizations
        #     (Stage 2 of the cortisol-only manuscript pipeline).
        drive_kind = config["drive"]["kind"]
        if drive_kind not in {"two_harmonic", "two_harmonic_noise"}:
            raise ValueError(
                "fit_habs_no_delay_stochastic_peak_stats requires "
                "drive.kind ∈ {two_harmonic, two_harmonic_noise}"
            )
        drive_params = config["drive"]["params"]
        if "baseline" not in drive_params:
            raise ValueError("fit_habs_no_delay_stochastic_peak_stats requires drive.params.baseline")
        if "dataset" in drive_params:
            # Per-subject mode — needs the matching variant.
            if "variant" not in drive_params:
                raise ValueError(
                    "fit_habs_no_delay_stochastic_peak_stats: per-subject drive requires drive.params.variant"
                )
        else:
            # Global-drive mode — needs the explicit two-harmonic literals.
            for key in ("a24", "phase24", "a12", "phase12"):
                if key not in drive_params:
                    raise ValueError(
                        f"fit_habs_no_delay_stochastic_peak_stats global drive requires drive.params.{key}"
                    )
        if float(model_params.get("tau_min", 0.0)) != 0.0:
            raise ValueError("fit_habs_no_delay_stochastic_peak_stats requires model.params.tau_min=0.0")
        free_params = [str(name) for name in config.get("fit", {}).get("free_params", [])]
        if not free_params:
            raise ValueError("fit_habs_no_delay_stochastic_peak_stats requires fit.free_params")
        allowed = {"kgr", "epsilon", "baseline", "epsilon_x1", "epsilon_x2", "epsilon_x3"}
        invalid = set(free_params) - allowed
        if invalid:
            raise ValueError(
                f"fit_habs_no_delay_stochastic_peak_stats unsupported free params: {sorted(invalid)}"
            )
        if "tau_min" in free_params:
            raise ValueError("fit_habs_no_delay_stochastic_peak_stats keeps tau_min fixed and cannot fit tau_min")
        if "epsilon" in free_params and drive_kind != "two_harmonic_noise":
            raise ValueError(
                "fit_habs_no_delay_stochastic_peak_stats: fitting drive 'epsilon' requires drive.kind=two_harmonic_noise"
            )
        if loss_mode != "peak_stats":
            raise ValueError("fit_habs_no_delay_stochastic_peak_stats requires fit.loss.mode=peak_stats")
        optimizer_name = str(config.get("fit", {}).get("optimizer", {}).get("name", "differential_evolution"))
        if optimizer_name != "differential_evolution":
            raise ValueError(
                "fit_habs_no_delay_stochastic_peak_stats requires fit.optimizer.name=differential_evolution"
            )
        runtime = config.get("runtime", {})
        noise_form = str(runtime.get("noise_form", "multiplicative"))
        if noise_form not in {
            "multiplicative", "additive", "lognormal",
            "normal_positive", "multiplicative_positive",
        }:
            raise ValueError(
                "fit_habs_no_delay_stochastic_peak_stats runtime.noise_form must be "
                "multiplicative, additive, lognormal, normal_positive, or multiplicative_positive"
            )
        noise_locations = [str(location) for location in runtime.get("noise_locations", [])]
        # Either secretion-site noise (noise_locations) or drive noise
        # (drive.kind=two_harmonic_noise + 'epsilon' free param) must be
        # active. Allow noise_locations to be empty when drive noise is on.
        drive_noise_active = (
            drive_kind == "two_harmonic_noise" and "epsilon" in free_params
        )
        if not noise_locations and not drive_noise_active:
            raise ValueError(
                "fit_habs_no_delay_stochastic_peak_stats requires either "
                "runtime.noise_locations or drive noise (two_harmonic_noise + 'epsilon' free param)"
            )
        invalid_locations = set(noise_locations) - {"x1_secretion", "x2_secretion", "x3_secretion"}
        if invalid_locations:
            raise ValueError(
                f"fit_habs_no_delay_stochastic_peak_stats unsupported noise locations: {sorted(invalid_locations)}"
            )
        bounds = config.get("fit", {}).get("bounds", {})
        for param in free_params:
            if param not in bounds:
                raise ValueError(f"fit_habs_no_delay_stochastic_peak_stats requires fit.bounds.{param}")

    if task == "fit_ultradian_psd":
        if config["drive"]["kind"] != "two_harmonic_noise":
            raise ValueError("fit_ultradian_psd requires drive.kind=two_harmonic_noise")
        if float(model_params.get("tau_min", 0.0)) != 0.0:
            raise ValueError("fit_ultradian_psd requires model.params.tau_min=0.0")
        fit_cfg = config.get("fit", {})
        series_ids = fit_cfg.get("series_ids", "all")
        if series_ids != "all" and not list(series_ids):
            raise ValueError("fit_ultradian_psd requires fit.series_ids to be 'all' or a non-empty list")
        free_params = [str(name) for name in fit_cfg.get("free_params", [])]
        if not free_params:
            raise ValueError("fit_ultradian_psd requires fit.free_params")
        allowed = {"kgr", "epsilon", "b1", "b2", "b3"}
        invalid = set(free_params) - allowed
        if invalid:
            raise ValueError(
                f"fit_ultradian_psd unsupported free params: {sorted(invalid)}; allowed: {sorted(allowed)}"
            )
        bounds = fit_cfg.get("bounds", {})
        for param in free_params:
            if param not in bounds:
                raise ValueError(f"fit_ultradian_psd requires fit.bounds.{param}")
        if "epsilon" in free_params and config["drive"]["kind"] != "two_harmonic_noise":
            raise ValueError(
                "fit_ultradian_psd: fitting drive 'epsilon' requires drive.kind=two_harmonic_noise"
            )
        psd_mode = str(fit_cfg.get("psd_mode", "zscore"))
        if psd_mode not in {"zscore", "fractional", "absolute"}:
            raise ValueError("fit_ultradian_psd fit.psd_mode must be zscore, fractional, or absolute")
        signals = [str(s) for s in fit_cfg.get("signals", ["Cortisol"])]
        if any(s not in {"Cortisol", "ACTH"} for s in signals):
            raise ValueError("fit_ultradian_psd fit.signals must be among {Cortisol, ACTH}")
        if psd_mode == "absolute" and len(signals) > 1:
            raise ValueError(
                "fit_ultradian_psd psd_mode=absolute supports a single signal "
                "(absolute scale is ill-defined across signals with different units); use fractional"
            )
        optimizer_name = str(fit_cfg.get("optimizer", {}).get("name", "differential_evolution"))
        if optimizer_name not in {"differential_evolution", "nelder_mead", "nelder-mead", "local"}:
            raise ValueError(
                "fit_ultradian_psd requires fit.optimizer.name in "
                "{differential_evolution, nelder_mead, local}"
            )

    if task == "compare_habs_multisite_production_noise":
        if dataset["name"] != "habs" or dataset["variant"] != "shifted":
            raise ValueError("compare_habs_multisite_production_noise requires dataset habs/shifted")
        if config["drive"]["kind"] != "two_harmonic":
            raise ValueError("compare_habs_multisite_production_noise requires drive.kind=two_harmonic")
        drive_params = config["drive"]["params"]
        for key in ["dataset", "variant", "baseline"]:
            if key not in drive_params:
                raise ValueError(f"compare_habs_multisite_production_noise requires drive.params.{key}")
        if loss_mode != "peak_stats":
            raise ValueError("compare_habs_multisite_production_noise requires fit.loss.mode=peak_stats")
        noise_combinations = config.get("comparison", {}).get("noise_combinations", [])
        if not noise_combinations:
            raise ValueError("compare_habs_multisite_production_noise requires comparison.noise_combinations")
        noise_forms = config.get("comparison", {}).get("noise_forms", [])
        if not noise_forms:
            raise ValueError("compare_habs_multisite_production_noise requires comparison.noise_forms")

    if task == "fit_joint_dataset_multistart":
        if config["drive"]["kind"] != "two_harmonic":
            raise ValueError("fit_joint_dataset_multistart requires drive.kind=two_harmonic")
        free_params = config["fit"].get("free_params", [])
        if not free_params:
            raise ValueError("fit_joint_dataset_multistart requires fit.free_params")
        bounds = config["fit"].get("bounds", {})
        for p in free_params:
            if p not in bounds:
                raise ValueError(f"fit_joint_dataset_multistart requires bounds for {p}")

    if task == "compare_joint_dataset_noise":
        if config["drive"]["kind"] != "two_harmonic":
            raise ValueError("compare_joint_dataset_noise requires drive.kind=two_harmonic")
        comparison = config.get("comparison", {})
        if "noise_combinations" not in comparison:
            raise ValueError("compare_joint_dataset_noise requires comparison.noise_combinations")
        if "noise_forms" not in comparison:
            raise ValueError("compare_joint_dataset_noise requires comparison.noise_forms")


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)
