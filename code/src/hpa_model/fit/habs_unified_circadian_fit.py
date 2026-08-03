"""Pooled (population-level) circadian-input fitter.

Stage 1 of the cortisol-only manuscript pipeline: fit ONE shared two-harmonic
circadian drive shape across all subjects from habs + digitize_2019 +
all_digitized so Stage 2 can layer drive noise on a fixed, population-level
circadian input.

* Model: canonical three-state ODE with ``tau_min=0`` and no noise.
* Drive: ``two_harmonic`` with literal parameters; no per-subject lookup.
* Free parameters: ``harmonic_split, phase24, phase12``. ``total_amplitude``
  and ``baseline`` are pinned (z-scoring removes their identifiability).
* Objective: per-subject z-scored cortisol MSE, summed across subjects.
* Optimizer: CMA-ES.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "hpa_model-mpl"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cma

from ..data.registry import get_dataset_spec, load_dataset
from ..data.two_harmonic_shift import fit_two_harmonic_params
from ..model.three_state_gr_delay import (
    ThreeStateGRDelayModel,
    TwoHarmonicDrive,
    TwoHarmonicNoiseDrive,
)
from ..plotting import setup_nature_style
from ..simulate.engine import simulate_trajectory_fit_arrays


FREE_PARAM_ORDER: tuple[str, ...] = ("harmonic_split", "phase24", "phase12")

_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "harmonic_split": (0.0, 1.0),
    "phase24": (0.0, 2.0 * math.pi),
    "phase12": (0.0, 2.0 * math.pi),
}

_DEFAULT_X0: dict[str, float] = {
    "harmonic_split": 0.7,
    "phase24": math.pi,
    "phase12": math.pi,
}

DEFAULT_TOTAL_AMPLITUDE: float = 0.5
DEFAULT_BASELINE: float = 1.0
DEFAULT_DATASETS: tuple[str, ...] = ("habs", "digitize_2019", "all_digitized")


def _split_amplitudes(
    harmonic_split: float, *, total_amplitude: float
) -> tuple[float, float]:
    split = float(np.clip(harmonic_split, 0.0, 1.0))
    return float(total_amplitude) * split, float(total_amplitude) * (1.0 - split)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr))
    if std <= 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


@dataclass(frozen=True)
class _PooledSubject:
    dataset: str
    series_id: str
    series_uid: str
    times_min: np.ndarray
    cortisol_z: np.ndarray          # per-subject z-score (shape only)
    cortisol_mean_norm: np.ndarray  # per-subject mean-normalised: x / mean(x)
    cortisol_raw: np.ndarray
    # ACTH is available for HABS only; ``None`` otherwise. When present it shares
    # the cortisol observation times (HABS samples both signals together).
    acth_times_min: np.ndarray | None = None
    acth_z: np.ndarray | None = None
    acth_mean_norm: np.ndarray | None = None


def _load_pooled_cortisol(
    datasets: tuple[str, ...], variant: str, *, min_obs: int = 10,
) -> list[_PooledSubject]:
    """Load per-subject cortisol (all datasets) and ACTH (where available, HABS).

    ACTH and cortisol are sampled together in HABS, so when an ACTH column exists
    we keep rows with both signals present and attach the per-subject z-scored /
    mean-normalised ACTH alongside cortisol.
    """
    subjects: list[_PooledSubject] = []
    for ds_name in datasets:
        spec = get_dataset_spec(ds_name)
        cort_col = next((s.column for s in spec.signals if s.name == "Cortisol"), None)
        if cort_col is None:
            continue
        acth_col = next((s.column for s in spec.signals if s.name == "ACTH"), None)
        frame = load_dataset(ds_name, variant)
        for series_id, group in frame.groupby(spec.id_col, sort=True):
            cols = [spec.time_col, cort_col]
            sub = group[cols].dropna()
            if len(sub) < min_obs:
                continue
            sub = sub.sort_values(spec.time_col)
            times = sub[spec.time_col].to_numpy(dtype=float)
            cort = sub[cort_col].to_numpy(dtype=float)
            mean_c = float(np.mean(cort))
            if not np.isfinite(mean_c) or mean_c <= 0:
                continue
            acth_times = acth_z = acth_mn = None
            if acth_col is not None:
                asub = group[[spec.time_col, acth_col]].dropna().sort_values(spec.time_col)
                if len(asub) >= min_obs:
                    a_t = asub[spec.time_col].to_numpy(dtype=float)
                    a_v = asub[acth_col].to_numpy(dtype=float)
                    mean_a = float(np.mean(a_v))
                    if np.isfinite(mean_a) and mean_a > 0:
                        acth_times = a_t
                        acth_z = _zscore(a_v)
                        acth_mn = a_v / mean_a
            subjects.append(
                _PooledSubject(
                    dataset=ds_name,
                    series_id=str(series_id),
                    series_uid=f"{ds_name}:{series_id}",
                    times_min=times,
                    cortisol_z=_zscore(cort),
                    cortisol_mean_norm=cort / mean_c,
                    cortisol_raw=cort,
                    acth_times_min=acth_times,
                    acth_z=acth_z,
                    acth_mean_norm=acth_mn,
                )
            )
    if not subjects:
        raise ValueError(
            f"No subjects with >= {min_obs} cortisol observations across "
            f"datasets={datasets} (variant={variant})."
        )
    return subjects


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
        tau_min=0.0,
        x3_floor=float(params["x3_floor"]),
        hill_coeff=float(params["hill_coeff"]),
        initial_state=tuple(float(x) for x in params["initial_state"]),
    )


def _simulate_pooled(
    *,
    model: ThreeStateGRDelayModel,
    theta: dict[str, float],
    solver: dict[str, float],
    period_min: float,
    second_period_min: float,
    total_amplitude: float,
    baseline: float,
    noise_epsilon: float = 0.0,
    noise_form: str = "lognormal",
    noise_reps: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the ODE for the given circadian shape.

    Returns ``(sim_time_min, sim_x3, sim_x2)`` (cortisol, ACTH). With
    ``noise_epsilon > 0`` and ``noise_reps > 1`` the drive carries multiplicative
    lognormal noise and the returned x2/x3 are the ensemble mean over reps — the
    quantity the deterministic-then-frozen pipeline silently ignores.
    """
    a24, a12 = _split_amplitudes(theta["harmonic_split"], total_amplitude=total_amplitude)
    dt_min = float(solver["dt_min"])
    warmup_min = float(solver["warmup_min"])
    duration_min = float(solver["duration_min"])
    if noise_epsilon > 0.0 and noise_reps > 1:
        drive = TwoHarmonicNoiseDrive(
            a24=a24, phase24=float(theta["phase24"]),
            a12=a12, phase12=float(theta["phase12"]),
            baseline=float(baseline), epsilon=float(noise_epsilon),
            period_min=period_min, second_period_min=second_period_min,
            noise_form=str(noise_form),
        )
        acc_x3 = None
        acc_x2 = None
        t_ref = None
        for rep in range(noise_reps):
            sim = simulate_trajectory_fit_arrays(
                model, drive, dt_min=dt_min, warmup_min=warmup_min,
                duration_min=duration_min, seed=int(seed) + rep,
                noise_form=str(noise_form),
            )
            if acc_x3 is None:
                t_ref = sim["time_min"]
                acc_x3 = np.zeros_like(sim["x3"], dtype=float)
                acc_x2 = np.zeros_like(sim["x2"], dtype=float)
            acc_x3 += sim["x3"]
            acc_x2 += sim["x2"]
        return t_ref, acc_x3 / noise_reps, acc_x2 / noise_reps

    drive = TwoHarmonicDrive(
        a24=a24,
        phase24=float(theta["phase24"]),
        a12=a12,
        phase12=float(theta["phase12"]),
        baseline=float(baseline),
        period_min=period_min,
        second_period_min=second_period_min,
    )
    sim = simulate_trajectory_fit_arrays(
        model,
        drive,
        dt_min=dt_min,
        warmup_min=warmup_min,
        duration_min=duration_min,
        seed=int(seed),
        noise_form="multiplicative",
    )
    return sim["time_min"], sim["x3"], sim["x2"]


def _objective(
    theta_vec: np.ndarray,
    *,
    subjects: list[_PooledSubject],
    model: ThreeStateGRDelayModel,
    solver: dict[str, float],
    period_min: float,
    second_period_min: float,
    total_amplitude: float,
    baseline: float,
    acth_weight: float = 0.0,
    noise_epsilon: float = 0.0,
    noise_form: str = "lognormal",
    noise_reps: int = 1,
    seed: int = 0,
) -> float:
    """Pooled z-scored MSE on cortisol, optionally plus a co-weighted ACTH term.

    With ``acth_weight > 0`` the objective is
    ``cort_mse + acth_weight * acth_mse`` where each term is the per-subject
    z-scored MSE averaged over its own subject pool. ``acth_weight = 1`` gives
    equal signal weight despite ACTH having fewer subjects (HABS only).
    """
    theta = {name: float(theta_vec[i]) for i, name in enumerate(FREE_PARAM_ORDER)}
    sim_t, sim_x3, sim_x2 = _simulate_pooled(
        model=model, theta=theta, solver=solver,
        period_min=period_min, second_period_min=second_period_min,
        total_amplitude=total_amplitude, baseline=baseline,
        noise_epsilon=noise_epsilon, noise_form=noise_form,
        noise_reps=noise_reps, seed=seed,
    )
    if not (np.all(np.isfinite(sim_x3)) and np.all(np.isfinite(sim_x2))):
        return 1e6
    duration = float(solver["duration_min"])
    cort_total = 0.0
    n_cort = 0
    acth_total = 0.0
    n_acth = 0
    for subject in subjects:
        obs_mod = np.mod(subject.times_min, duration)
        sim_c = _zscore(np.interp(obs_mod, sim_t, sim_x3))
        if np.all(np.isfinite(sim_c)):
            cort_total += float(np.mean((sim_c - subject.cortisol_z) ** 2))
            n_cort += 1
        if acth_weight > 0.0 and subject.acth_z is not None:
            a_mod = np.mod(subject.acth_times_min, duration)
            sim_a = _zscore(np.interp(a_mod, sim_t, sim_x2))
            if np.all(np.isfinite(sim_a)):
                acth_total += float(np.mean((sim_a - subject.acth_z) ** 2))
                n_acth += 1
    if n_cort == 0:
        return 1e6
    err = cort_total / n_cort
    if acth_weight > 0.0 and n_acth > 0:
        err += acth_weight * (acth_total / n_acth)
    if not np.isfinite(err):
        return 1e6
    return err


def _resolve_bounds(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds_cfg = dict(config.get("fit", {}).get("bounds", {}))
    x0_cfg = dict(config.get("fit", {}).get("x0", {}))
    lower: list[float] = []
    upper: list[float] = []
    x0_values: list[float] = []
    for name in FREE_PARAM_ORDER:
        lo, hi = _DEFAULT_BOUNDS[name]
        if name in bounds_cfg:
            lo, hi = float(bounds_cfg[name][0]), float(bounds_cfg[name][1])
        x0 = float(x0_cfg.get(name, _DEFAULT_X0[name]))
        x0 = min(max(x0, lo), hi)
        lower.append(lo); upper.append(hi); x0_values.append(x0)
    return (np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            np.asarray(x0_values, dtype=float))


def _run_cma_es(
    *,
    subjects: list[_PooledSubject],
    model: ThreeStateGRDelayModel,
    solver: dict[str, float],
    period_min: float,
    second_period_min: float,
    total_amplitude: float,
    baseline: float,
    lower: np.ndarray,
    upper: np.ndarray,
    x0: np.ndarray,
    optimizer_cfg: dict[str, Any],
    seed: int,
    logger: logging.Logger | None,
    acth_weight: float = 0.0,
    noise_epsilon: float = 0.0,
    noise_form: str = "lognormal",
    noise_reps: int = 1,
) -> tuple[dict[str, float], float, int]:
    sigma0 = float(optimizer_cfg.get("sigma0", 0.25))
    popsize = int(optimizer_cfg.get("popsize", 16))
    maxiter = int(optimizer_cfg.get("maxiter", 400))
    tolfun = float(optimizer_cfg.get("tolfun", 1e-6))
    tolx = float(optimizer_cfg.get("tolx", 1e-6))
    opts: dict[str, Any] = {
        "bounds": [list(lower), list(upper)],
        "popsize": popsize,
        "maxiter": maxiter,
        "tolfun": tolfun,
        "tolx": tolx,
        "seed": int(seed),
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [
            _objective(
                np.asarray(sol, dtype=float),
                subjects=subjects,
                model=model,
                solver=solver,
                period_min=period_min,
                second_period_min=second_period_min,
                total_amplitude=total_amplitude,
                baseline=baseline,
                acth_weight=acth_weight,
                noise_epsilon=noise_epsilon,
                noise_form=noise_form,
                noise_reps=noise_reps,
                seed=seed,
            )
            for sol in solutions
        ]
        es.tell(solutions, fitnesses)
    best_vec = np.asarray(es.result.xbest, dtype=float)
    best_val = float(es.result.fbest)
    n_evals = int(es.result.evaluations)
    if logger is not None:
        logger.info(
            "Pooled circadian fit: objective=%.5f over %d subjects (%d evals)",
            best_val, len(subjects), n_evals,
        )
    return {name: float(best_vec[i]) for i, name in enumerate(FREE_PARAM_ORDER)}, best_val, n_evals


def _binned_pooled_curve(subjects: list[_PooledSubject], *,
                          duration_min: float, n_bins: int = 48,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin all (time mod 24h, z_cortisol) points and return (centers, mean, std)."""
    edges = np.linspace(0.0, duration_min, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    all_t: list[float] = []
    all_z: list[float] = []
    for sub in subjects:
        t = np.mod(sub.times_min, duration_min)
        all_t.extend(t.tolist())
        all_z.extend(sub.cortisol_z.tolist())
    arr_t = np.asarray(all_t, dtype=float)
    arr_z = np.asarray(all_z, dtype=float)
    means = np.full(n_bins, np.nan)
    stds  = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (arr_t >= edges[i]) & (arr_t < edges[i + 1])
        if int(mask.sum()) >= 3:
            means[i] = float(np.mean(arr_z[mask]))
            stds[i]  = float(np.std(arr_z[mask]))
    return centers, means, stds


def _plot_pooled_overlay(
    *,
    subjects: list[_PooledSubject],
    theta: dict[str, float],
    model: ThreeStateGRDelayModel,
    solver: dict[str, float],
    period_min: float,
    second_period_min: float,
    total_amplitude: float,
    baseline: float,
    out_path: Path,
) -> None:
    setup_nature_style()
    sim_t, sim_x3, _sim_x2 = _simulate_pooled(
        model=model, theta=theta, solver=solver,
        period_min=period_min, second_period_min=second_period_min,
        total_amplitude=total_amplitude, baseline=baseline,
    )
    sim_z = _zscore(sim_x3)

    duration = float(solver["duration_min"])
    centers, b_means, b_stds = _binned_pooled_curve(subjects, duration_min=duration)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    # Background: every subject's z-cortisol points by time-of-day.
    for sub in subjects:
        t = np.mod(sub.times_min, duration) / 60.0
        ax.scatter(t, sub.cortisol_z, s=4, color="#888888", alpha=0.2,
                   edgecolors="none", zorder=1)
    # Population mean ± std bands.
    ok = np.isfinite(b_means)
    ax.fill_between(centers[ok] / 60.0,
                    (b_means - b_stds)[ok], (b_means + b_stds)[ok],
                    color="#2F5C85", alpha=0.20, zorder=2,
                    label="Pooled mean ± SD")
    ax.plot(centers[ok] / 60.0, b_means[ok], color="#2F5C85", lw=1.4, zorder=3,
            label="Pooled mean")
    # Fitted model curve.
    ax.plot(sim_t / 60.0, sim_z, color="#C85C3A", lw=2.0, zorder=4,
            label="Fitted ODE (z-cortisol)")
    ax.axhline(0, color="#aaaaaa", lw=0.6, ls="--", zorder=1)
    ax.set_xlim(0, duration / 60.0)
    ax.set_xlabel("Time of day (h, shifted clock)")
    ax.set_ylabel("Z-scored cortisol")
    n_subj = len(subjects)
    ax.set_title(
        f"Pooled circadian-input fit  (n = {n_subj} subjects)",
        fontsize=10, loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _fit_direct_to_pooled_cortisol(
    subjects: list[_PooledSubject], *,
    period_min: float, second_period_min: float,
    target: str = "mean_norm",
) -> tuple[dict[str, float], float, int]:
    """Curve-fit a two-harmonic directly to the pooled (t mod 24h, y_per-subject).

    ``target`` selects the per-subject normalisation:
      * ``"mean_norm"``: y = x / mean(x). Then the fitted constant c maps
        directly to the drive baseline (≈ 1) and (a24, a12) are dimensionless
        modulation depths around the daily mean.
      * ``"zscore"``: legacy z-score (loses baseline information).
    """
    all_t = np.concatenate([np.mod(sub.times_min, period_min) for sub in subjects])
    if target == "mean_norm":
        all_y = np.concatenate([sub.cortisol_mean_norm for sub in subjects])
    elif target == "zscore":
        all_y = np.concatenate([sub.cortisol_z for sub in subjects])
    else:
        raise ValueError(f"Unknown direct-fit target: {target!r}")
    params = fit_two_harmonic_params(
        all_t, all_y,
        period_min=period_min, second_period_min=second_period_min,
    )
    if params is None:
        raise RuntimeError("fit_two_harmonic_params returned None on pooled data")
    a24 = float(params["a24"])
    a12 = float(params["a12"])
    phase24 = float(params["phase24"])
    phase12 = float(params["phase12"])
    c_fit = float(params.get("c", 0.0))
    from ..data.two_harmonic_shift import _harmonic_model
    sse = 0.0; n = 0
    for sub in subjects:
        t_mod = np.mod(sub.times_min, period_min)
        y_obs = sub.cortisol_mean_norm if target == "mean_norm" else sub.cortisol_z
        y_fit = _harmonic_model(
            t_mod, a24=a24, phi24=phase24, a12=a12, phi12=phase12, c=c_fit,
            period_min=period_min, second_period_min=second_period_min,
        )
        sse += float(np.sum((y_obs - y_fit) ** 2))
        n += y_obs.size
    mse = sse / max(n, 1)
    total_amplitude = a24 + a12
    harmonic_split = a24 / total_amplitude if total_amplitude > 0 else 1.0
    return ({"harmonic_split": harmonic_split,
             "phase24": phase24, "phase12": phase12,
             "_a24": a24, "_a12": a12, "_c": c_fit,
             "_target": target}, mse, n)


def fit_pooled_circadian_input_from_config(
    config: dict[str, Any],
    run_dir: Path,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    datasets_cfg = dataset_cfg.get("datasets") or [dataset_cfg.get("name", "habs")]
    datasets = tuple(str(d) for d in datasets_cfg)
    variant = str(dataset_cfg.get("variant", "shifted_12h"))

    subjects = _load_pooled_cortisol(datasets, variant)
    if logger is not None:
        logger.info(
            "Loaded %d subjects (cortisol, %s, %s).",
            len(subjects), ", ".join(datasets), variant,
        )

    work_config = deepcopy(config)
    work_config["model"]["params"]["tau_min"] = 0.0
    model = _build_model(work_config)
    solver = {
        "dt_min": float(work_config["solver"]["dt_min"]),
        "warmup_min": float(work_config["solver"]["warmup_min"]),
        "duration_min": float(work_config["solver"]["duration_min"]),
    }
    drive_cfg = work_config.get("drive", {}).get("params", {})
    period_min = float(drive_cfg.get("period_min", 1440.0))
    second_period_min = float(drive_cfg.get("second_period_min", 720.0))
    total_amplitude = float(
        work_config.get("fit", {}).get("total_amplitude", DEFAULT_TOTAL_AMPLITUDE)
    )
    baseline = float(work_config.get("fit", {}).get("baseline", DEFAULT_BASELINE))
    if total_amplitude <= 0.0:
        raise ValueError("fit.total_amplitude must be positive")
    if baseline <= total_amplitude:
        raise ValueError("fit.baseline must exceed fit.total_amplitude.")

    method = str(work_config.get("fit", {}).get("method", "direct")).lower()

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if method == "direct":
        # Curve-fit two-harmonic directly to pooled (per-subject mean-normalised)
        # cortisol. The fitted constant c maps directly to the drive baseline,
        # and (a24, a12) are the modulation amplitudes around it. This makes
        # baseline a real fitted parameter — no rescaling required.
        direct_target = str(
            work_config.get("fit", {}).get("direct_target", "mean_norm")
        )
        theta_full, objective_value, n_evals = _fit_direct_to_pooled_cortisol(
            subjects, period_min=period_min, second_period_min=second_period_min,
            target=direct_target,
        )
        theta = {k: v for k, v in theta_full.items() if not k.startswith("_")}
        a24_fit = float(theta_full["_a24"])
        a12_fit = float(theta_full["_a12"])
        c_fit = float(theta_full["_c"])
        # Use the fitted values directly: baseline = c, total_amplitude = a24+a12.
        baseline = c_fit
        total_amplitude = a24_fit + a12_fit
        if logger is not None:
            logger.info(
                "Direct two-harmonic fit on pooled %s-cortisol: "
                "a24=%.3f a12=%.3f phase24=%.3f phase12=%.3f c=%.3f (MSE=%.4f).",
                direct_target, a24_fit, a12_fit,
                theta["phase24"], theta["phase12"], c_fit, objective_value,
            )
    elif method == "ode":
        lower, upper, x0 = _resolve_bounds(work_config)
        optimizer_cfg = dict(work_config.get("fit", {}).get("optimizer", {}))
        seed = int(work_config.get("runtime", {}).get("seed", 123))
        fit_cfg = dict(work_config.get("fit", {}))
        acth_weight = float(fit_cfg.get("acth_weight", 0.0))
        noise_epsilon = float(fit_cfg.get("noise_epsilon", 0.0))
        noise_form = str(fit_cfg.get("noise_form", "lognormal"))
        noise_reps = int(fit_cfg.get("noise_reps", 1))
        if logger is not None:
            logger.info(
                "ODE circadian fit: acth_weight=%.2f noise_epsilon=%.3f "
                "noise_reps=%d.", acth_weight, noise_epsilon, noise_reps,
            )
        theta, objective_value, n_evals = _run_cma_es(
            subjects=subjects,
            model=model,
            solver=solver,
            period_min=period_min,
            second_period_min=second_period_min,
            total_amplitude=total_amplitude,
            baseline=baseline,
            lower=lower, upper=upper, x0=x0,
            optimizer_cfg=optimizer_cfg,
            seed=seed,
            logger=logger,
            acth_weight=acth_weight,
            noise_epsilon=noise_epsilon,
            noise_form=noise_form,
            noise_reps=noise_reps,
        )
    else:
        raise ValueError(f"Unknown fit.method: {method!r} (expected 'direct' or 'ode')")
    a24, a12 = _split_amplitudes(theta["harmonic_split"], total_amplitude=total_amplitude)
    params_row = {
        "harmonic_split": float(theta["harmonic_split"]),
        "phase24": float(theta["phase24"]),
        "phase12": float(theta["phase12"]),
        "a24": float(a24),
        "a12": float(a12),
        "baseline": float(baseline),
        "total_amplitude": float(total_amplitude),
        "period_min": float(period_min),
        "second_period_min": float(second_period_min),
        "objective_value": float(objective_value),
        "n_subjects": int(len(subjects)),
        "n_evaluations": int(n_evals),
        "datasets": "|".join(datasets),
        "dataset_variant": variant,
        "method": method,
        "acth_weight": float(work_config.get("fit", {}).get("acth_weight", 0.0)),
        "noise_epsilon": float(work_config.get("fit", {}).get("noise_epsilon", 0.0)),
        "noise_reps": int(work_config.get("fit", {}).get("noise_reps", 1)),
    }
    summary = pd.DataFrame([params_row])
    summary.to_csv(artifacts_dir / "pooled_circadian_params.csv", index=False)

    _plot_pooled_overlay(
        subjects=subjects, theta=theta, model=model, solver=solver,
        period_min=period_min, second_period_min=second_period_min,
        total_amplitude=total_amplitude, baseline=baseline,
        out_path=figures_dir / "pooled_circadian_fit.png",
    )

    # Also export per-subject (sim, data) z-score residuals for diagnostics.
    sim_t, sim_x3, _sim_x2 = _simulate_pooled(
        model=model, theta=theta, solver=solver,
        period_min=period_min, second_period_min=second_period_min,
        total_amplitude=total_amplitude, baseline=baseline,
    )
    diag_rows: list[dict[str, Any]] = []
    duration = float(solver["duration_min"])
    for sub in subjects:
        obs_mod = np.mod(sub.times_min, duration)
        sim_at_obs = np.interp(obs_mod, sim_t, sim_x3)
        sim_z = _zscore(sim_at_obs)
        for t, dz, sz in zip(sub.times_min, sub.cortisol_z, sim_z):
            diag_rows.append({
                "dataset": sub.dataset, "series_uid": sub.series_uid,
                "time_min": float(t),
                "data_z": float(dz), "sim_z": float(sz),
                "residual": float(dz - sz),
            })
    pd.DataFrame(diag_rows).to_csv(
        artifacts_dir / "pooled_cortisol_curve_comparison.csv", index=False,
    )

    return {
        "summary": summary,
        "n_subjects": int(len(subjects)),
        "objective_value": float(objective_value),
        "theta": theta,
        "a24": float(a24),
        "a12": float(a12),
        "baseline": float(baseline),
    }
