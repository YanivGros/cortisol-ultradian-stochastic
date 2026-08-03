"""Fixed-step simulation engine for the delayed HPA model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ..model.three_state_gr_delay import Drive, ThreeStateGRDelayModel


@dataclass(frozen=True)
class SolverSpec:
    dt_min: float
    warmup_min: float
    duration_min: float


VALID_NOISE_LOCATIONS = {"drive", "x1_secretion", "x2_secretion", "x3_secretion"}
VALID_SECRETION_NOISE_LOCATIONS = VALID_NOISE_LOCATIONS - {"drive"}
VALID_NOISE_FORMS = {
    "multiplicative", "additive", "lognormal",
    "normal_positive", "multiplicative_positive",
}


def _noise_scale_from_draw(*, noise_form: str, epsilon: float, noise_draw: float) -> float:
    if noise_form == "multiplicative":
        return 1.0 + epsilon * noise_draw
    if noise_form == "lognormal":
        # Center the multiplicative factor at 1.0 while keeping it strictly positive.
        return float(math.exp(epsilon * noise_draw - 0.5 * epsilon * epsilon))
    if noise_form == "normal_positive":
        # Truncated-normal multiplier: 1 + ε·N(0,1) clipped at 0, so the drive
        # stays non-negative without the heavy right tail of lognormal.
        return float(max(0.0, 1.0 + epsilon * noise_draw))
    if noise_form == "multiplicative_positive":
        # One-sided positive shocks only: 1 + ε·|N(0,1)|. Drive is always
        # ≥ baseline (noise can only boost u(t), never suppress it). Mean of
        # the multiplier is 1 + ε·√(2/π) ≈ 1 + 0.798·ε.
        return 1.0 + float(epsilon) * abs(float(noise_draw))
    raise ValueError(f"Unsupported multiplicative-style noise_form: {noise_form}")


def _project_state_for_update(state: np.ndarray) -> np.ndarray:
    projected = np.asarray(state, dtype=float).copy()
    projected[1:] = np.maximum(projected[1:], 0.0)
    return projected


def _delayed_x2_value(
    query_t_min: float,
    times: np.ndarray,
    x2_history: np.ndarray,
    fallback: float,
) -> float:
    if query_t_min <= 0:
        return float(fallback)
    if query_t_min >= float(times[-1]):
        return float(x2_history[-1])
    return float(np.interp(query_t_min, times, x2_history))


def _resolve_noise_settings(
    *,
    noise_location: str | None,
    noise_epsilon: float,
    noise_locations: list[str] | tuple[str, ...] | None,
    noise_epsilons: dict[str, float] | None,
    noise_form: str,
) -> tuple[list[str], dict[str, float]]:
    if noise_location is not None and noise_location not in VALID_NOISE_LOCATIONS:
        raise ValueError(f"Unsupported noise_location: {noise_location}")
    if noise_locations is None:
        selected_noise_locations: list[str] = []
    else:
        selected_noise_locations = [str(location) for location in noise_locations]
        invalid_locations = set(selected_noise_locations) - VALID_SECRETION_NOISE_LOCATIONS
        if invalid_locations:
            raise ValueError(f"Unsupported noise_locations: {sorted(invalid_locations)}")
    if noise_location is not None and noise_location != "drive":
        if noise_location in selected_noise_locations:
            raise ValueError(f"Duplicate secretion noise location: {noise_location}")
        selected_noise_locations.append(str(noise_location))
    resolved_noise_epsilons = {str(key): float(value) for key, value in (noise_epsilons or {}).items()}
    if noise_location is not None and noise_location != "drive":
        resolved_noise_epsilons.setdefault(str(noise_location), float(noise_epsilon))
    invalid_epsilon_locations = set(resolved_noise_epsilons) - VALID_SECRETION_NOISE_LOCATIONS
    if invalid_epsilon_locations:
        raise ValueError(f"Unsupported noise_epsilons locations: {sorted(invalid_epsilon_locations)}")
    if noise_form not in VALID_NOISE_FORMS:
        raise ValueError(f"Unsupported noise_form: {noise_form}")
    return selected_noise_locations, resolved_noise_epsilons


def _delayed_x2_value_indexed(
    step_idx: int,
    *,
    tau_min: float,
    dt_min: float,
    x2_history: np.ndarray,
    fallback: float,
) -> float:
    if tau_min <= 0.0:
        return float(x2_history[step_idx])
    query_idx = float(step_idx) - float(tau_min) / float(dt_min)
    if query_idx <= 0.0:
        return float(fallback)
    lower_idx = int(math.floor(query_idx))
    upper_idx = int(math.ceil(query_idx))
    if upper_idx > step_idx:
        return float(x2_history[step_idx])
    if lower_idx == upper_idx:
        return float(x2_history[lower_idx])
    weight = query_idx - float(lower_idx)
    return float((1.0 - weight) * x2_history[lower_idx] + weight * x2_history[upper_idx])


def simulate_trajectory_fit_arrays(
    model: ThreeStateGRDelayModel,
    drive: Drive,
    *,
    dt_min: float,
    warmup_min: float,
    duration_min: float,
    seed: int,
    noise_location: str | None = None,
    noise_epsilon: float = 0.0,
    noise_locations: list[str] | tuple[str, ...] | None = None,
    noise_epsilons: dict[str, float] | None = None,
    noise_form: str = "multiplicative",
) -> dict[str, np.ndarray]:
    selected_noise_locations, resolved_noise_epsilons = _resolve_noise_settings(
        noise_location=noise_location,
        noise_epsilon=noise_epsilon,
        noise_locations=noise_locations,
        noise_epsilons=noise_epsilons,
        noise_form=noise_form,
    )
    total_min = float(warmup_min + duration_min)
    n_steps = int(round(total_min / dt_min))
    times = np.arange(n_steps + 1, dtype=float) * dt_min
    observe_from_idx = int(round(float(warmup_min) / float(dt_min)))

    state = np.asarray(model.initial_state, dtype=float).copy()
    x2_history = np.empty(n_steps + 1, dtype=float)
    x3_history = np.empty(n_steps + 1, dtype=float)
    x2_history[0] = float(state[1])
    x3_history[0] = float(state[2])
    rng = np.random.default_rng(seed)

    for idx, t_min in enumerate(times[:-1]):
        u_t = float(drive.sample(float(t_min), rng))
        x2_delay = _delayed_x2_value_indexed(
            idx,
            tau_min=float(model.tau_min),
            dt_min=float(dt_min),
            x2_history=x2_history,
            fallback=float(x2_history[0]),
        )
        x1_scale = 1.0
        x2_scale = 1.0
        x3_scale = 1.0
        x1_additive = 0.0
        x2_additive = 0.0
        x3_additive = 0.0
        for selected_location in selected_noise_locations:
            epsilon = max(0.0, float(resolved_noise_epsilons.get(selected_location, 0.0)))
            noise_draw = float(rng.normal())
            if noise_form in {"multiplicative", "lognormal"}:
                noise_factor = _noise_scale_from_draw(
                    noise_form=noise_form,
                    epsilon=epsilon,
                    noise_draw=noise_draw,
                )
                if selected_location == "x1_secretion":
                    x1_scale = noise_factor
                elif selected_location == "x2_secretion":
                    x2_scale = noise_factor
                elif selected_location == "x3_secretion":
                    x3_scale = noise_factor
            else:
                additive_noise = epsilon * noise_draw
                if selected_location == "x1_secretion":
                    x1_additive = additive_noise
                elif selected_location == "x2_secretion":
                    x2_additive = additive_noise
                elif selected_location == "x3_secretion":
                    x3_additive = additive_noise
        drift = model.drift(
            float(t_min),
            state,
            u_t,
            x2_delay=x2_delay,
            x1_secretion_scale=x1_scale,
            x2_secretion_scale=x2_scale,
            x3_secretion_scale=x3_scale,
            x1_secretion_additive=x1_additive,
            x2_secretion_additive=x2_additive,
            x3_secretion_additive=x3_additive,
        )
        state = _project_state_for_update(state + dt_min * drift)
        x2_history[idx + 1] = float(state[1])
        x3_history[idx + 1] = float(state[2])

    observed_slice = slice(observe_from_idx, None)
    return {
        "time_min": times[observed_slice] - float(warmup_min),
        "x2": x2_history[observed_slice],
        "x3": x3_history[observed_slice],
    }


def simulate_trajectory(
    model: ThreeStateGRDelayModel,
    drive: Drive,
    *,
    dt_min: float,
    warmup_min: float,
    duration_min: float,
    seed: int,
    noise_location: str | None = None,
    noise_epsilon: float = 0.0,
    noise_locations: list[str] | tuple[str, ...] | None = None,
    noise_epsilons: dict[str, float] | None = None,
    noise_form: str = "multiplicative",
) -> pd.DataFrame:
    selected_noise_locations, resolved_noise_epsilons = _resolve_noise_settings(
        noise_location=noise_location,
        noise_epsilon=noise_epsilon,
        noise_locations=noise_locations,
        noise_epsilons=noise_epsilons,
        noise_form=noise_form,
    )
    total_min = float(warmup_min + duration_min)
    n_steps = int(round(total_min / dt_min))
    times = np.arange(n_steps + 1, dtype=float) * dt_min

    states = np.zeros((n_steps + 1, 3), dtype=float)
    states[0] = np.asarray(model.initial_state, dtype=float)
    u_values = np.zeros(n_steps + 1, dtype=float)
    rng = np.random.default_rng(seed)

    for idx, t_min in enumerate(times[:-1]):
        u_t = float(drive.sample(float(t_min), rng))
        u_values[idx] = u_t
        x2_delay = _delayed_x2_value_indexed(
            idx,
            tau_min=float(model.tau_min),
            dt_min=float(dt_min),
            x2_history=states[:, 1],
            fallback=float(states[0, 1]),
        )
        x1_scale = 1.0
        x2_scale = 1.0
        x3_scale = 1.0
        x1_additive = 0.0
        x2_additive = 0.0
        x3_additive = 0.0
        for selected_location in selected_noise_locations:
            epsilon = max(0.0, float(resolved_noise_epsilons.get(selected_location, 0.0)))
            noise_draw = float(rng.normal())
            if noise_form in {"multiplicative", "lognormal"}:
                noise_factor = _noise_scale_from_draw(
                    noise_form=noise_form,
                    epsilon=epsilon,
                    noise_draw=noise_draw,
                )
                if selected_location == "x1_secretion":
                    x1_scale = noise_factor
                elif selected_location == "x2_secretion":
                    x2_scale = noise_factor
                elif selected_location == "x3_secretion":
                    x3_scale = noise_factor
            else:
                additive_noise = epsilon * noise_draw
                if selected_location == "x1_secretion":
                    x1_additive = additive_noise
                elif selected_location == "x2_secretion":
                    x2_additive = additive_noise
                elif selected_location == "x3_secretion":
                    x3_additive = additive_noise
        drift = model.drift(
            float(t_min),
            states[idx],
            u_t,
            x2_delay=x2_delay,
            x1_secretion_scale=x1_scale,
            x2_secretion_scale=x2_scale,
            x3_secretion_scale=x3_scale,
            x1_secretion_additive=x1_additive,
            x2_secretion_additive=x2_additive,
            x3_secretion_additive=x3_additive,
        )
        states[idx + 1] = _project_state_for_update(states[idx] + dt_min * drift)

    u_values[-1] = float(drive.sample(float(times[-1]), rng))

    frame = pd.DataFrame(
        {
            "time_min": times,
            "x1": states[:, 0],
            "x2": states[:, 1],
            "x3": states[:, 2],
            "u": u_values,
        }
    )
    observed = frame.loc[frame["time_min"] >= warmup_min].copy()
    observed["time_min"] = observed["time_min"] - warmup_min
    return observed.reset_index(drop=True)


def sample_trajectory(
    trajectory: pd.DataFrame,
    observation_times: np.ndarray,
) -> pd.DataFrame:
    sampled = {"time_min": observation_times}
    for column in ["x1", "x2", "x3", "u"]:
        sampled[column] = np.interp(observation_times, trajectory["time_min"], trajectory[column])
    return pd.DataFrame(sampled)


def simulate_replicates(
    model: ThreeStateGRDelayModel,
    drive: Drive,
    *,
    dt_min: float,
    warmup_min: float,
    duration_min: float,
    n_reps: int,
    seed: int,
    observation_times: np.ndarray | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rep in range(n_reps):
        traj = simulate_trajectory(
            model,
            drive,
            dt_min=dt_min,
            warmup_min=warmup_min,
            duration_min=duration_min,
            seed=seed + rep,
        )
        if observation_times is not None:
            traj = sample_trajectory(traj, observation_times)
        traj["rep"] = rep
        frames.append(traj)
    return pd.concat(frames, ignore_index=True)


def aggregate_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("time_min", as_index=False)
    summary = grouped.agg(
        x1_mean=("x1", "mean"),
        x1_std=("x1", "std"),
        x2_mean=("x2", "mean"),
        x2_std=("x2", "std"),
        x3_mean=("x3", "mean"),
        x3_std=("x3", "std"),
        u_mean=("u", "mean"),
        u_std=("u", "std"),
    )
    return summary.fillna(0.0)
