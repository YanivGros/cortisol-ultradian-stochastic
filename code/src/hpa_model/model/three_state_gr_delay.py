"""Three-state delayed HPA model with GR feedback."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

from ..data.registry import get_dataset_spec, load_shift_params


def rate_from_half_life(half_life_minutes: float) -> float:
    if half_life_minutes <= 0:
        raise ValueError("half_life_minutes must be positive")
    return math.log(2.0) / half_life_minutes


DEFAULT_A1 = rate_from_half_life(5.0)
DEFAULT_A2 = rate_from_half_life(10.0)
DEFAULT_A3 = rate_from_half_life(15.0)  # cortisol t1/2=15 min (canonical drive-noise fit; data median ~14.6 min)
DEFAULT_B1 = DEFAULT_A1
DEFAULT_B2 = DEFAULT_A2
DEFAULT_B3 = DEFAULT_A3
DEFAULT_KGR = 5.0
DEFAULT_TAU_MIN = 0.0
DEFAULT_X3_FLOOR = 0.01
DEFAULT_HILL_COEFF = 3.0
DEFAULT_INITIAL_STATE = (1.0, 1.0, 1.0)
MIN_POSITIVE_STATE = np.finfo(float).tiny


class Drive(Protocol):
    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        """Sample the drive value at time t."""


@dataclass(frozen=True)
class ConstantDrive:
    level: float = 1.0

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        del t_min, rng
        return self.level


@dataclass(frozen=True)
class SineDrive:
    baseline: float = 1.0
    amplitude: float = 0.0
    phase_min: float = 0.0
    period_min: float = 1440.0

    def base_value(self, t_min: float) -> float:
        angle = 2.0 * math.pi * (t_min - self.phase_min) / self.period_min
        return self.baseline + self.amplitude * math.sin(angle)

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        del rng
        return self.base_value(t_min)


@dataclass(frozen=True)
class SineNoiseDrive:
    baseline: float = 1.0
    amplitude: float = 0.0
    phase_min: float = 0.0
    period_min: float = 1440.0
    epsilon: float = 0.0

    def base_value(self, t_min: float) -> float:
        angle = 2.0 * math.pi * (t_min - self.phase_min) / self.period_min
        return self.baseline + self.amplitude * math.sin(angle)

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        base = self.base_value(t_min)
        noise_factor = 1.0 + self.epsilon * rng.normal()
        return base * noise_factor


@dataclass(frozen=True)
class PulseTrainDrive:
    """Repeating short bursts of input above a constant baseline.

    Within each period of `period_min`, the drive sits at `baseline + amplitude`
    for the first `pulse_width_min` minutes (relative to `phase_min`) and at
    `baseline` for the rest. Useful for modeling discrete activations of CRH
    (meals, stress events, photic stimuli) rather than smooth sinusoidal drive.
    """

    baseline: float = 1.0
    amplitude: float = 1.0
    pulse_width_min: float = 10.0
    period_min: float = 90.0
    phase_min: float = 0.0

    def base_value(self, t_min: float) -> float:
        t_in_cycle = (float(t_min) - self.phase_min) % self.period_min
        if 0.0 <= t_in_cycle < self.pulse_width_min:
            return self.baseline + self.amplitude
        return self.baseline

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        del rng
        return self.base_value(t_min)


@dataclass(frozen=True)
class TwoHarmonicDrive:
    a24: float
    phase24: float
    a12: float
    phase12: float
    baseline: float
    period_min: float = 1440.0
    second_period_min: float = 720.0

    def base_value(self, t_min: float) -> float:
        w24 = 2.0 * math.pi / self.period_min
        w12 = 2.0 * math.pi / self.second_period_min
        return (
            self.baseline
            + self.a24 * math.sin(w24 * t_min + self.phase24)
            + self.a12 * math.sin(w12 * t_min + self.phase12)
        )

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        del rng
        return self.base_value(t_min)


@dataclass(frozen=True)
class TwoHarmonicNoiseDrive:
    a24: float
    phase24: float
    a12: float
    phase12: float
    baseline: float
    epsilon: float = 0.0
    period_min: float = 1440.0
    second_period_min: float = 720.0
    # Noise form for the drive multiplier. "multiplicative" gives 1+εN(0,1)
    # which can go negative for large ε; "lognormal" gives exp(εN−ε²/2)
    # (always positive, mean 1) and is the right choice for biological drives.
    noise_form: str = "multiplicative"

    def base_value(self, t_min: float) -> float:
        w24 = 2.0 * math.pi / self.period_min
        w12 = 2.0 * math.pi / self.second_period_min
        return (
            self.baseline
            + self.a24 * math.sin(w24 * t_min + self.phase24)
            + self.a12 * math.sin(w12 * t_min + self.phase12)
        )

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        base = self.base_value(t_min)
        z = rng.normal()
        if self.noise_form == "lognormal":
            noise_factor = math.exp(self.epsilon * z - 0.5 * self.epsilon * self.epsilon)
        elif self.noise_form == "normal_positive":
            noise_factor = max(0.0, 1.0 + self.epsilon * z)
        elif self.noise_form == "multiplicative_positive":
            noise_factor = 1.0 + self.epsilon * abs(z)
        else:
            noise_factor = 1.0 + self.epsilon * z
        return base * noise_factor


def _normalize_series_id(series_id: object, spec_id_col: str) -> object:
    if spec_id_col == "ID":
        try:
            return int(series_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected integer series id for {spec_id_col}, got {series_id!r}") from exc
    return str(series_id)


def _resolve_shift_row(params: dict[str, Any]) -> dict[str, float]:
    dataset_name = str(params["dataset"])
    variant = str(params.get("variant", "shifted"))
    series_id = params["series_id"]
    shift_params = load_shift_params(dataset_name, variant=variant)
    spec = get_dataset_spec(dataset_name)
    normalized_id = _normalize_series_id(series_id, spec.id_col)
    matches = shift_params.loc[shift_params[spec.id_col] == normalized_id]
    if matches.empty:
        raise KeyError(f"No shift parameters for dataset={dataset_name!r}, {spec.id_col}={normalized_id!r}")
    return {key: float(value) for key, value in matches.iloc[0].to_dict().items() if key != spec.id_col and key != "fallback_mode"}


def _apply_time_shift_to_two_harmonic(row: dict[str, float], shift_min: float) -> dict[str, float]:
    period_min = float(row.get("period_min", 1440.0))
    second_period_min = float(row.get("second_period_min", 720.0))
    w24 = 2.0 * math.pi / period_min
    w12 = 2.0 * math.pi / second_period_min
    return {
        **row,
        "phase24": float((float(row["phase24"]) - w24 * shift_min) % (2.0 * math.pi)),
        "phase12": float((float(row["phase12"]) - w12 * shift_min) % (2.0 * math.pi)),
        "peak24_min": float((float(row.get("peak24_min", 0.0)) + shift_min) % period_min),
        "peak12_min": float((float(row.get("peak12_min", 0.0)) + shift_min) % second_period_min),
        "combined_peak_min": float((float(row.get("combined_peak_min", 0.0)) + shift_min) % period_min),
    }


def _two_harmonic_params(params: dict[str, Any]) -> dict[str, float]:
    if "dataset" in params and "series_id" in params:
        row = _resolve_shift_row(params)
        variant = str(params.get("variant", "shifted"))
        phase_reference = str(params.get("phase_reference", "shifted" if variant == "shifted" else "raw"))
        if phase_reference not in {"raw", "shifted"}:
            raise ValueError(f"Unsupported phase_reference: {phase_reference}")
        if phase_reference == "shifted":
            row = _apply_time_shift_to_two_harmonic(row, float(row.get("applied_shift_min", 0.0)))
        fitted_baseline = max(abs(float(row["c"])), 1e-6)
        baseline = float(params.get("baseline", 1.0))
        amplitude_scale = float(params.get("amplitude_scale", 1.0))
        a24_scale = float(params.get("a24_scale", amplitude_scale))
        a12_scale = float(params.get("a12_scale", amplitude_scale))
        epsilon = float(params.get("epsilon", 0.0))
        period_min = float(params.get("period_min", row.get("period_min", 1440.0)))
        second_period_min = float(params.get("second_period_min", row.get("second_period_min", 720.0)))
        phase24 = float(row["phase24"])
        phase12 = float(row["phase12"])
        if "period_min" in params:
            w24 = 2.0 * math.pi / period_min
            phase24 = float((math.pi / 2.0 - w24 * float(row.get("peak24_min", 0.0))) % (2.0 * math.pi))
        if "second_period_min" in params:
            w12 = 2.0 * math.pi / second_period_min
            phase12 = float((math.pi / 2.0 - w12 * float(row.get("peak12_min", 0.0))) % (2.0 * math.pi))
        return {
            "a24": float(a24_scale * baseline * float(row["a24"]) / fitted_baseline),
            "phase24": phase24,
            "a12": float(a12_scale * baseline * float(row["a12"]) / fitted_baseline),
            "phase12": phase12,
            "baseline": baseline,
            "period_min": period_min,
            "second_period_min": second_period_min,
            "epsilon": epsilon,
        }
    return {
        "a24": float(params.get("a24", 0.0)),
        "phase24": float(params.get("phase24", 0.0)),
        "a12": float(params.get("a12", 0.0)),
        "phase12": float(params.get("phase12", 0.0)),
        "baseline": float(params.get("baseline", params.get("c", 0.0))),
        "period_min": float(params.get("period_min", 1440.0)),
        "second_period_min": float(params.get("second_period_min", 720.0)),
        "epsilon": float(params.get("epsilon", 0.0)),
    }


def build_drive(kind: str, params: dict[str, Any]) -> Drive:
    if kind == "constant":
        return ConstantDrive(
            level=float(params.get("level", params.get("baseline", 1.0))),
        )
    if kind == "sine":
        return SineDrive(
            baseline=float(params.get("baseline", 1.0)),
            amplitude=float(params.get("amplitude", 0.0)),
            phase_min=float(params.get("phase_min", 0.0)),
            period_min=float(params.get("period_min", 1440.0)),
        )
    if kind == "sine_noise":
        return SineNoiseDrive(
            baseline=float(params.get("baseline", 1.0)),
            amplitude=float(params.get("amplitude", 0.0)),
            phase_min=float(params.get("phase_min", 0.0)),
            period_min=float(params.get("period_min", 1440.0)),
            epsilon=float(params.get("epsilon", 0.0)),
        )
    if kind == "pulse_train":
        return PulseTrainDrive(
            baseline=float(params.get("baseline", 1.0)),
            amplitude=float(params.get("amplitude", 1.0)),
            pulse_width_min=float(params.get("pulse_width_min", 10.0)),
            period_min=float(params.get("period_min", 90.0)),
            phase_min=float(params.get("phase_min", 0.0)),
        )
    if kind == "two_harmonic":
        resolved = _two_harmonic_params(params)
        return TwoHarmonicDrive(
            a24=resolved["a24"],
            phase24=resolved["phase24"],
            a12=resolved["a12"],
            phase12=resolved["phase12"],
            baseline=resolved["baseline"],
            period_min=resolved["period_min"],
            second_period_min=resolved["second_period_min"],
        )
    if kind == "two_harmonic_noise":
        resolved = _two_harmonic_params(params)
        return TwoHarmonicNoiseDrive(
            a24=resolved["a24"],
            phase24=resolved["phase24"],
            a12=resolved["a12"],
            phase12=resolved["phase12"],
            baseline=resolved["baseline"],
            epsilon=resolved["epsilon"],
            period_min=resolved["period_min"],
            second_period_min=resolved["second_period_min"],
            noise_form=str(params.get("noise_form", "multiplicative")),
        )
    raise ValueError(f"Unsupported drive kind: {kind}")


@dataclass(frozen=True)
class ThreeStateGRDelayModel:
    """CRH-ACTH-cortisol model with delayed ACTH-to-cortisol coupling."""

    a1: float = DEFAULT_A1
    a2: float = DEFAULT_A2
    a3: float = DEFAULT_A3
    b1: float = DEFAULT_B1
    b2: float = DEFAULT_B2
    b3: float = DEFAULT_B3
    kgr: float = DEFAULT_KGR
    tau_min: float = DEFAULT_TAU_MIN
    x3_floor: float = DEFAULT_X3_FLOOR
    hill_coeff: float = DEFAULT_HILL_COEFF
    initial_state: tuple[float, float, float] = DEFAULT_INITIAL_STATE

    @property
    def state_names(self) -> tuple[str, str, str]:
        return ("x1", "x2", "x3")

    def drift(
        self,
        t_min: float,
        state: np.ndarray,
        u: float,
        *,
        x2_delay: float | None = None,
        x1_secretion_scale: float = 1.0,
        x2_secretion_scale: float = 1.0,
        x3_secretion_scale: float = 1.0,
        x1_secretion_additive: float = 0.0,
        x2_secretion_additive: float = 0.0,
        x3_secretion_additive: float = 0.0,
    ) -> np.ndarray:
        del t_min
        x1_raw, x2_raw, x3_raw = np.asarray(state, dtype=float)
        x1 = float(x1_raw)
        x2 = max(float(x2_raw), 0.0)
        x3 = max(float(x3_raw), 0.0)
        x3_eff = max(x3, self.x3_floor, MIN_POSITIVE_STATE)
        kgr_eff = max(float(self.kgr), self.x3_floor, MIN_POSITIVE_STATE)
        x2_delayed = max(float(x2 if x2_delay is None else x2_delay), 0.0)

        mr = 1.0 / x3_eff
        gr = 1.0 / (1.0 + (x3_eff / kgr_eff) ** self.hill_coeff)

        dx1 = self.b1 * gr * mr * float(u) * float(x1_secretion_scale) + float(x1_secretion_additive) - self.a1 * x1
        dx2 = self.b2 * x1 * gr * float(x2_secretion_scale) + float(x2_secretion_additive) - self.a2 * x2
        dx3 = self.b3 * x2_delayed * float(x3_secretion_scale) + float(x3_secretion_additive) - self.a3 * x3
        if x3 <= 0.0 and dx3 < 0.0:
            dx3 = 0.0
        return np.array([dx1, dx2, dx3], dtype=float)

    def to_params_dict(self) -> dict[str, float | bool | list[float]]:
        return {
            "a1": self.a1,
            "a2": self.a2,
            "a3": self.a3,
            "b1": self.b1,
            "b2": self.b2,
            "b3": self.b3,
            "kgr": self.kgr,
            "tau_min": self.tau_min,
            "x3_floor": self.x3_floor,
            "hill_coeff": self.hill_coeff,
            "initial_state": list(self.initial_state),
        }


def resolve_constant_drive_steady_state(
    model: ThreeStateGRDelayModel,
    drive_level: float,
) -> np.ndarray:
    """Solve the positive fixed point for a constant drive level."""
    drive = max(0.0, float(drive_level))
    production_scale = (
        float(model.b1)
        * float(model.b2)
        * float(model.b3)
        * drive
        / max(float(model.a1) * float(model.a2) * float(model.a3), MIN_POSITIVE_STATE)
    )
    if production_scale <= 0.0:
        return np.zeros(3, dtype=float)

    x3_floor = max(float(model.x3_floor), MIN_POSITIVE_STATE)
    kgr_eff = max(float(model.kgr), x3_floor)
    hill_coeff = float(model.hill_coeff)
    target = math.sqrt(production_scale)

    def gr_for_x3(x3_value: float) -> float:
        return 1.0 / (1.0 + (float(x3_value) / kgr_eff) ** hill_coeff)

    def residual(x3_value: float) -> float:
        return float(x3_value) * (1.0 + (float(x3_value) / kgr_eff) ** hill_coeff) - target

    floor_residual = residual(x3_floor)
    if floor_residual >= 0.0:
        x3 = production_scale * gr_for_x3(x3_floor) ** 2 / x3_floor
    else:
        lower = x3_floor
        upper = max(target, kgr_eff, x3_floor * 2.0, 1.0)
        while residual(upper) < 0.0:
            upper *= 2.0
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            if residual(midpoint) < 0.0:
                lower = midpoint
            else:
                upper = midpoint
        x3 = 0.5 * (lower + upper)

    x3_eff = max(float(x3), x3_floor)
    gr = gr_for_x3(x3_eff)
    x1 = float(model.b1) * gr * drive / max(float(model.a1) * x3_eff, MIN_POSITIVE_STATE)
    x2 = float(model.b2) * x1 * gr / max(float(model.a2), MIN_POSITIVE_STATE)
    x3 = float(model.b3) * x2 / max(float(model.a3), MIN_POSITIVE_STATE)
    return np.array([x1, x2, x3], dtype=float)
