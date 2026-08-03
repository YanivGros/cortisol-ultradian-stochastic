"""Human-scaled Walker, Terry & Lightman (2010) pituitary--adrenal delay model.

Reference: Walker JJ, Terry JR, Lightman SL (2010) "Origin of ultradian
pulsatility in the hypothalamic--pituitary--adrenal axis", Proc R Soc B
277:1627--1633 (doi:10.1098/rspb.2009.2148), electronic supplementary material
eq. (6) and Table 1.

The model is the dimensionless delay differential equation (DDE)

    da/dt = p1 / (1 + p2 r o) - p3 a
    dr/dt = (o r)^2 / (p4 + (o r)^2) + p5 - p6 r
    do/dt = a(t - tau) - o

where ``a`` = ACTH, ``r`` = glucocorticoid-receptor (GR) availability, ``o`` =
cortisol (CORT), and ``tau`` is the ACTH->CORT delay. Dimensionless time relates
to real time by ``t = (ln 2 / CORT_half_life) * T`` (their eq. 10), i.e. one
dimensionless time unit equals ``CORT_half_life / ln 2`` minutes.

**Human scaling (this module).** Walker's published parameters are rodent-derived.
Here we keep the *structure* but rescale the clearance/timescale block from our
own human HPA half-lives:

* ``p3 = CORT_half_life / ACTH_half_life`` (their Table-1 ``p3``); with our
  human half-lives (CORT 15 min, ACTH 20 min) this is ``0.75`` (vs the rat's 7.2).
* the dimensionless->minutes conversion uses CORT_half_life = 15 min, so one time
  unit is ``15 / ln 2`` ~ 21.6 min. Because human CORT clearance is ~2x slower
  than the rat's, the model's natural ultradian period lands near the human
  ~100--135 min range without further tuning.

The GR-submodule dimensionless parameters ``p2, p4, p5, p6`` have no analogue in
our (dynamic-GR-free) model, so they are left free to be fit to the data (their
rodent defaults 15, 0.05, 0.11, 2.9 make useful initial guesses / bounds anchors).

``p1`` is the (CRH) drive onto the pituitary; pass a constant or supply
``drive_fn`` to modulate it by a shared circadian envelope.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

LN2 = math.log(2.0)

# Walker (2010) Table 1 rodent defaults — used only as fit anchors / bounds.
WALKER_P2_RODENT = 15.0
WALKER_P4_RODENT = 0.05
WALKER_P5_RODENT = 0.11
WALKER_P6_RODENT = 2.9


def p3_from_half_lives(cort_half_life_min: float, acth_half_life_min: float) -> float:
    """p3 = CORT half-life / ACTH half-life (= ratio of dimensional decay rates)."""
    return float(cort_half_life_min) / float(acth_half_life_min)


def simulate_walker(
    *,
    p1: float,
    p2: float,
    p4: float,
    p5: float,
    p6: float,
    tau_min: float,
    p3: float = 0.75,
    cort_half_life_min: float = 15.0,
    dt_min: float = 0.25,
    warmup_min: float = 5760.0,
    duration_min: float = 1440.0,
    drive_fn: Callable[[float], float] | None = None,
    initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
    epsilon: float = 0.0,
    noise_form: str = "lognormal",
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the human-scaled Walker (2010) DDE in real-time minutes.

    The dimensionless RHS is multiplied by ``k_od = ln2 / cort_half_life_min`` so
    the system can be integrated directly on a minutes grid (and the delay
    ``tau_min`` is a real-time lag). Fixed-step forward Euler with an explicit
    history buffer for the delayed ACTH term ``a(t - tau)``.

    If ``drive_fn`` is given the effective drive is ``p1 * drive_fn(T)`` (T in
    minutes); ``drive_fn`` should have mean ~1 (e.g. a two-harmonic circadian
    envelope normalised to a unit baseline). Returns a dict with ``time_min``
    (relative to the end of warm-up) and the three state arrays ``a``, ``r``, ``o``
    (``o`` = cortisol).

    **Drive noise (optional).** With ``epsilon > 0`` the drive is multiplied at
    every step by a mean-preserving stochastic factor, mirroring the canonical
    lognormal drive noise of the three-state model (``simulate/engine.py``):
    ``lognormal`` → ``exp(epsilon*z - epsilon**2/2)``, ``multiplicative`` →
    ``1 + epsilon*z``, with ``z ~ N(0,1)`` drawn per step from ``default_rng(seed)``.
    ``epsilon = 0`` reproduces the deterministic integration exactly.
    """
    k_od = LN2 / float(cort_half_life_min)
    dt = float(dt_min)
    n_warm = int(round(float(warmup_min) / dt))
    n_obs = int(round(float(duration_min) / dt))
    n_total = n_warm + n_obs
    lag_steps = int(round(float(tau_min) / dt))

    eps = float(epsilon)
    rng = np.random.default_rng(seed) if eps > 0.0 else None

    a, r, o = (float(initial_state[0]), float(initial_state[1]), float(initial_state[2]))
    a_hist = np.empty(n_total + 1, dtype=float)
    a_hist[0] = a

    out_t = np.empty(n_obs + 1, dtype=float)
    out_a = np.empty(n_obs + 1, dtype=float)
    out_r = np.empty(n_obs + 1, dtype=float)
    out_o = np.empty(n_obs + 1, dtype=float)

    for step in range(n_total + 1):
        t_real = step * dt
        if step >= n_warm:
            j = step - n_warm
            out_t[j] = j * dt
            out_a[j], out_r[j], out_o[j] = a, r, o

        if step == n_total:
            break

        # delayed ACTH a(t - tau): fall back to the initial value before t=0
        a_delayed = a_hist[step - lag_steps] if step - lag_steps >= 0 else a_hist[0]

        base_drive = drive_fn(t_real) if drive_fn is not None else 1.0
        if rng is not None:
            z = rng.normal()
            if noise_form == "lognormal":
                base_drive *= math.exp(eps * z - 0.5 * eps * eps)
            else:  # mean-1 additive-multiplicative ("multiplicative")
                base_drive *= max(0.0, 1.0 + eps * z)
        drive = p1 * base_drive

        # keep states non-negative / non-singular
        o_eff = o if o > 1e-9 else 1e-9
        r_eff = r if r > 1e-9 else 1e-9
        or2 = (o_eff * r_eff) ** 2

        da = k_od * (drive / (1.0 + p2 * r_eff * o_eff) - p3 * a)
        dr = k_od * (or2 / (p4 + or2) + p5 - p6 * r)
        do = k_od * (a_delayed - o)

        a = a + dt * da
        r = r + dt * dr
        o = o + dt * do
        if o < 0.0:
            o = 0.0
        if r < 0.0:
            r = 0.0
        a_hist[step + 1] = a

    return {"time_min": out_t, "a": out_a, "r": out_r, "o": out_o}
