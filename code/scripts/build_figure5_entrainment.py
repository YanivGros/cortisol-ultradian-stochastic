"""Figure 5 — entrainment of the noise-driven HPA oscillator by a pulsatile cue.

Combined multi-panel figure (one file, panels A-D) built on the canonical
drive-noise model (kgr=5, cortisol t1/2=15 min, lognormal drive noise eps=1.707).

The entraining stimulus is a *pulse train* (a brief bolus of width
``--pulse-width-min`` delivered every ``--stim-period-min``), added on top of the
circadian drive before the multiplicative noise — i.e. a discrete cue such as a
scheduled meal, light flash, or pulsatile hormone bolus, NOT a smooth sine.

Panels:
  A  Cortisol dynamics: noise-only (top) vs entrained (bottom); the stimulus
     pulses are drawn as shaded bars so you can see pulses lock to the cue.
  B  Peak raster across realizations (no stimulus vs entrained) with the cue
     times marked.
  C  Phase histogram of peak timing within the stimulus cycle: ~uniform with no
     stimulus (Rayleigh R~0), concentrated when entrained (R high).
  D  Frequency response — phase-locking R vs the period of a weak sine probe.
  E  Resonance — peak-to-trough cortisol amplitude vs probe period; peaks at the
     oscillator's natural ultradian period (dashed line). D-E use a weak sine in
     the feedback-engaged regime so they reflect the oscillator's own tuning.

Usage:
  PYTHONPATH=src python scripts/build_figure5_entrainment.py \
      --fit-dir experiments/runs/fit_cortisol_drive_noise_prom05_prev_dip_v14_cort15_stage1refit \
      --out experiments/runs/fig5_entrainment_pulse
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpa_model.fit.habs_dual_peak_stats import _build_model  # noqa: E402
from hpa_model.model.three_state_gr_delay import ConstantDrive, build_drive  # noqa: E402
from hpa_model.plotting import setup_nature_style  # noqa: E402
from hpa_model.simulate.engine import simulate_trajectory  # noqa: E402

COL_NOISE = "#355070"      # noise-only (blue)
COL_ENTRAIN = "#C44E52"    # entrained (red)
COL_STIM = "#E6A817"       # stimulus cue (amber)


# ── drive: two-harmonic circadian + pulse-train stimulus + lognormal noise ──────
@dataclass(frozen=True)
class CircadianPulseDrive:
    a24: float
    phase24: float
    a12: float
    phase12: float
    baseline: float = 1.0
    period_min: float = 1440.0
    second_period_min: float = 720.0
    # pulse-train stimulus
    stim_amplitude: float = 0.0
    stim_period_min: float = 120.0
    pulse_width_min: float = 10.0
    stim_phase_min: float = 0.0
    # multiplicative drive noise
    drive_epsilon: float = 0.0
    drive_noise_form: str = "lognormal"
    # how the external cue couples to the endogenous noise:
    #   "multiplicative": (circadian + cue) * noise  — cue is randomised by the
    #       endogenous secretion noise (legacy behaviour).
    #   "additive":       circadian * noise + cue     — the cue is exogenous
    #       (injected bolus / light / meal), delivered at a fixed dose and NOT
    #       subject to the body's own multiplicative secretion noise. Gives
    #       uniform peak heights even at high endogenous noise.
    cue_mode: str = "multiplicative"

    def circadian(self, t_min: float) -> float:
        w24 = 2.0 * math.pi / self.period_min
        w12 = 2.0 * math.pi / self.second_period_min
        return (
            self.baseline
            + self.a24 * math.sin(w24 * t_min + self.phase24)
            + self.a12 * math.sin(w12 * t_min + self.phase12)
        )

    def stimulus(self, t_min: float) -> float:
        if self.stim_amplitude <= 0.0:
            return 0.0
        t_in_cycle = (float(t_min) - self.stim_phase_min) % self.stim_period_min
        return self.stim_amplitude if 0.0 <= t_in_cycle < self.pulse_width_min else 0.0

    def _noise_factor(self, rng: np.random.Generator) -> float:
        eps = self.drive_epsilon
        if eps <= 0.0:
            return 1.0
        z = rng.normal()
        if self.drive_noise_form == "lognormal":
            return math.exp(eps * z - 0.5 * eps * eps)
        if self.drive_noise_form == "normal_positive":
            return max(0.0, 1.0 + eps * z)
        return 1.0 + eps * z

    def sample(self, t_min: float, rng: np.random.Generator) -> float:
        endo = self.circadian(t_min)
        cue = self.stimulus(t_min)
        if self.cue_mode == "additive":
            # noise multiplies only the endogenous drive; cue added noise-free
            base = endo * self._noise_factor(rng) + cue
        else:
            base = (endo + cue) * self._noise_factor(rng)
        return max(base, 1e-6)


# ── helpers ─────────────────────────────────────────────────────────────────────
def _load_model_and_params(fit_dir: Path):
    cfg = yaml.safe_load((fit_dir / "artifacts" / "fitted_config.yaml").read_text())
    model = _build_model(cfg)
    dp = cfg["drive"]["params"]
    circ = dict(
        a24=float(dp["a24"]), phase24=float(dp["phase24"]),
        a12=float(dp["a12"]), phase12=float(dp["phase12"]),
        baseline=float(dp.get("baseline", 1.0)),
        period_min=float(dp.get("period_min", 1440.0)),
        second_period_min=float(dp.get("second_period_min", 720.0)),
    )
    drive_eps = float(dp.get("epsilon", 0.0))
    noise_form = str(dp.get("noise_form", "lognormal"))
    return model, circ, drive_eps, noise_form


def _natural_period_min(model, level: float = 1.0) -> float:
    """Deterministic ringing period of the skeleton at a given constant drive."""
    s0 = 1.5 * level
    pert = model.__class__(
        a1=model.a1, a2=model.a2, a3=model.a3,
        b1=model.b1, b2=model.b2, b3=model.b3,
        kgr=model.kgr, tau_min=model.tau_min,
        x3_floor=getattr(model, "x3_floor", 0.01),
        hill_coeff=model.hill_coeff, initial_state=(s0, s0, s0),
    )
    df = simulate_trajectory(
        pert, ConstantDrive(level=level),
        dt_min=0.5, warmup_min=0.0, duration_min=2880.0, seed=0,
    )
    x3 = df["x3"].to_numpy(float)
    t = df["time_min"].to_numpy(float)
    pk, _ = find_peaks(x3)
    if len(pk) >= 3:
        return float(np.median(np.diff(t[pk])))
    return float("nan")


def _freq_response(model, period_min, *, baseline, amplitude, eps, dt_min,
                   warmup_min, duration_min, n_reps, base_seed, min_dist_min):
    """Probe the oscillator with a weak sinusoidal drive of the given period and
    measure (Rayleigh phase-locking R, mean peak-to-trough amplitude).

    Weak secretion noise + a clean sine isolate the system's frequency response
    (resonance), as opposed to the strong-noise pulsatile entrainment in A-C.
    """
    p2t, phases = [], []
    dist = max(1, int(round(min_dist_min / dt_min)))
    for rep in range(n_reps):
        drive = build_drive("sine", {"baseline": baseline, "amplitude": amplitude,
                                     "period_min": period_min})
        df = simulate_trajectory(
            model, drive, dt_min=dt_min, warmup_min=warmup_min,
            duration_min=duration_min, seed=base_seed + rep,
            noise_locations=["x1_secretion"], noise_epsilons={"x1_secretion": eps},
            noise_form="multiplicative",
        )
        x = df["x3"].to_numpy(float)
        t = df["time_min"].to_numpy(float)
        pk, _ = find_peaks(x, distance=dist)
        tr, _ = find_peaks(-x, distance=dist)
        if len(pk) and len(tr):
            p2t.append(float(x[pk].mean() - x[tr].mean()))
        if len(pk):
            phases.extend((2.0 * np.pi * ((t[pk] % period_min) / period_min)).tolist())
    R = float(np.abs(np.mean(np.exp(1j * np.array(phases))))) if phases else float("nan")
    return R, (float(np.mean(p2t)) if p2t else float("nan"))


def _detect_peaks(x3: np.ndarray, dt_min: float, prom_factor: float,
                  min_dist_min: float):
    std = float(np.std(x3))
    prom = prom_factor * std if std > 0 else prom_factor
    dist = max(1, int(round(min_dist_min / dt_min)))
    idx, _ = find_peaks(x3, distance=dist, prominence=prom)
    return idx


def _simulate(drive, model, *, dt_min, warmup_min, duration_min, n_reps,
              base_seed, prom_factor, min_dist_min):
    trajs, peak_t, peak_amp, peak_rep = [], [], [], []
    for rep in range(n_reps):
        df = simulate_trajectory(
            model, drive, dt_min=dt_min, warmup_min=warmup_min,
            duration_min=duration_min, seed=base_seed + rep,
        )
        x3 = df["x3"].to_numpy(float)
        t = df["time_min"].to_numpy(float)
        trajs.append((t, x3))
        idx = _detect_peaks(x3, dt_min, prom_factor, min_dist_min)
        peak_t.extend(t[idx].tolist())
        peak_amp.extend(x3[idx].tolist())
        peak_rep.extend([rep] * len(idx))
    return trajs, (np.array(peak_t), np.array(peak_amp), np.array(peak_rep))


def _rayleigh_R(peak_t: np.ndarray, period_min: float) -> float:
    if peak_t.size == 0:
        return float("nan")
    phase = 2.0 * np.pi * ((peak_t % period_min) / period_min)
    return float(np.abs(np.mean(np.exp(1j * phase))))


# ── main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/"
                    "fit_cortisol_drive_noise_prom05_prev_dip_v14_cort15_stage1refit")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "experiments/runs/fig5_entrainment_pulse")
    ap.add_argument("--stim-amplitude", type=float, default=4.0,
                    help="Demo pulse bolus height added to the drive baseline (~1.0) "
                         "for panels A-C (strong, clearly entraining)")
    # Panel D-E: frequency response (resonance), probed with a weak sine drive in
    # the feedback-engaged regime (high baseline relative to kgr), which is where
    # the GR feedback "spring" gives a sharp resonance at the natural period.
    ap.add_argument("--resonance-baseline", type=float, default=10.0,
                    help="Constant drive baseline for the weak-sine resonance probe")
    ap.add_argument("--resonance-amp-frac", type=float, default=0.15,
                    help="Sine probe amplitude as a fraction of the baseline")
    ap.add_argument("--resonance-eps", type=float, default=0.15,
                    help="Weak x1-secretion noise for the resonance probe")
    ap.add_argument("--stim-period-min", type=float, default=120.0)
    ap.add_argument("--pulse-width-min", type=float, default=10.0)
    ap.add_argument("--cue-mode", choices=["multiplicative", "additive"],
                    default="multiplicative",
                    help="How the external cue couples to the endogenous noise. "
                         "'multiplicative' (default): (circadian+cue)*noise. "
                         "'additive': circadian*noise + cue — exogenous cue not "
                         "randomised by endogenous noise (uniform peak heights).")
    ap.add_argument("--dt-min", type=float, default=1.0)
    ap.add_argument("--warmup-min", type=float, default=1440.0)
    ap.add_argument("--duration-min", type=float, default=1440.0)
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--sweep-duration-min", type=float, default=1440.0)
    ap.add_argument("--sweep-n-reps", type=int, default=120)
    ap.add_argument("--sweep-periods-min", nargs="+", type=float,
                    default=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 135, 150, 180, 220, 270])
    ap.add_argument("--prom-factor", type=float, default=0.1)
    ap.add_argument("--min-distance-min", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_nature_style()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "artifacts").mkdir(parents=True, exist_ok=True)

    model, circ, drive_eps, noise_form = _load_model_and_params(args.fit_dir)
    nat_period = _natural_period_min(model)
    print(f"[model] kgr={model.kgr} cort_t12={math.log(2)/model.a3:.1f}min "
          f"drive_eps={drive_eps:.3f} natural_period={nat_period:.1f}min")

    common = dict(stim_period_min=args.stim_period_min,
                  pulse_width_min=args.pulse_width_min,
                  drive_epsilon=drive_eps, drive_noise_form=noise_form,
                  cue_mode=args.cue_mode, **circ)
    noise_drive = CircadianPulseDrive(stim_amplitude=0.0, **common)
    entr_drive = CircadianPulseDrive(stim_amplitude=args.stim_amplitude, **common)

    sim_kw = dict(dt_min=args.dt_min, warmup_min=args.warmup_min,
                  duration_min=args.duration_min, n_reps=args.n_reps,
                  base_seed=args.seed, prom_factor=args.prom_factor,
                  min_dist_min=args.min_distance_min)
    print(f"[sim] noise-only ({args.n_reps} reps)...")
    noise_trajs, noise_pk = _simulate(noise_drive, model, **sim_kw)
    print(f"[sim] entrained ({args.n_reps} reps, pulse every "
          f"{args.stim_period_min:.0f} min, A={args.stim_amplitude})...")
    entr_trajs, entr_pk = _simulate(entr_drive, model, **sim_kw)

    R_noise = _rayleigh_R(noise_pk[0], args.stim_period_min)
    R_entr = _rayleigh_R(entr_pk[0], args.stim_period_min)
    print(f"[lock] Rayleigh R: no-stim={R_noise:.3f}  entrained={R_entr:.3f}")

    # frequency response (resonance) — weak sine probe, feedback engaged ---------
    res_amp = args.resonance_amp_frac * args.resonance_baseline
    nat_period_res = _natural_period_min(model, level=args.resonance_baseline)
    print(f"[resonance] weak sine probe (baseline={args.resonance_baseline}, "
          f"amp={res_amp:.2f}, eps={args.resonance_eps}); "
          f"natural period @ this drive = {nat_period_res:.1f} min")
    sweep = []
    for i, per in enumerate(args.sweep_periods_min):
        R, p2t = _freq_response(
            model, per, baseline=args.resonance_baseline, amplitude=res_amp,
            eps=args.resonance_eps, dt_min=args.dt_min, warmup_min=args.warmup_min,
            duration_min=args.sweep_duration_min, n_reps=args.sweep_n_reps,
            base_seed=args.seed + 1000 + i * args.sweep_n_reps,
            min_dist_min=args.min_distance_min,
        )
        sweep.append(dict(period_min=per, period_hr=per / 60.0,
                          rayleigh_R=R, peak_to_trough=p2t))
        print(f"   {per:5.0f} min  R={R:.3f}  peak-to-trough={p2t:.3f}")
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(args.out / "artifacts" / "frequency_sweep.csv", index=False)

    # ── figure assembly ────────────────────────────────────────────────────────
    _assemble(args, nat_period_res, noise_trajs, entr_trajs, noise_pk, entr_pk,
              R_noise, R_entr, sweep_df)

    # manifest
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    (args.out / "manifest.json").write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit, "python_version": platform.python_version(),
        "task": "build_figure5_entrainment",
        "fit_dir": str(args.fit_dir),
        "natural_period_resonance_min": nat_period_res,
        "stim_amplitude": args.stim_amplitude,
        "stim_period_min": args.stim_period_min,
        "pulse_width_min": args.pulse_width_min,
        "resonance_baseline": args.resonance_baseline,
        "resonance_amp_frac": args.resonance_amp_frac,
        "resonance_eps": args.resonance_eps,
        "R_no_stim": R_noise, "R_entrained": R_entr,
    }, indent=2))
    print(f"[done] {args.out}")


def _draw_pulses(ax, period_min, width_min, t_max_hr, color, alpha=0.18, label=None):
    k = 0
    first = True
    while k * period_min <= t_max_hr * 60.0:
        x0 = k * period_min / 60.0
        x1 = (k * period_min + width_min) / 60.0
        ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0,
                   label=label if first else None)
        first = False
        k += 1


def _assemble(args, nat_period, noise_trajs, entr_trajs, noise_pk, entr_pk,
              R_noise, R_entr, sweep_df) -> None:
    t_max_hr = args.duration_min / 60.0
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.5, 1.2, 1.05],
        hspace=0.6, wspace=0.34,
        left=0.10, right=0.96, top=0.95, bottom=0.07,
    )

    # Panel A: dynamics (noise-only top, entrained bottom) -----------------------
    gsA = gs[0, :].subgridspec(2, 1, hspace=0.5)
    axA1 = fig.add_subplot(gsA[0])
    axA2 = fig.add_subplot(gsA[1])
    t0, x0 = noise_trajs[0]
    axA1.plot(t0 / 60.0, x0, color=COL_NOISE, lw=1.1)
    axA1.set_title("Circadian drive + noise (no stimulus)", color=COL_NOISE, fontsize=9)
    axA1.set_ylabel("Cortisol (a.u.)")
    axA1.tick_params(labelbottom=False)
    t1, x1 = entr_trajs[0]
    _draw_pulses(axA2, args.stim_period_min, args.pulse_width_min, t_max_hr,
                 COL_STIM, alpha=0.30, label="stimulus pulse")
    axA2.plot(t1 / 60.0, x1, color=COL_ENTRAIN, lw=1.1)
    axA2.set_title(f"+ pulsatile stimulus every {args.stim_period_min:.0f} min",
                   color=COL_ENTRAIN, fontsize=9)
    axA2.set_ylabel("Cortisol (a.u.)")
    axA2.set_xlabel("Time (hours)")
    axA2.legend(loc="upper right", frameon=False, fontsize=7)
    for ax in (axA1, axA2):
        ax.set_xlim(0, t_max_hr)
    axA1.text(-0.10, 1.06, "A", transform=axA1.transAxes, fontweight="bold", fontsize=12)

    # Panel B: peak raster -------------------------------------------------------
    axB = fig.add_subplot(gs[1, 0])
    n_reps = args.n_reps
    _draw_pulses(axB, args.stim_period_min, args.pulse_width_min, t_max_hr,
                 COL_STIM, alpha=0.22)
    # entrained at bottom rows, noise-only at top rows
    et, _, erep = entr_pk
    nt, _, nrep = noise_pk
    axB.scatter(et / 60.0, erep, s=4, color=COL_ENTRAIN, alpha=0.8, lw=0)
    axB.scatter(nt / 60.0, nrep + n_reps + 3, s=4, color=COL_NOISE, alpha=0.8, lw=0)
    axB.axhline(n_reps + 1.5, color="0.6", lw=0.6, ls="--")
    axB.set_xlim(0, t_max_hr)
    axB.set_ylim(-2, 2 * n_reps + 5)
    axB.set_yticks([n_reps / 2, 1.5 * n_reps + 3])
    axB.set_yticklabels(["entrained", "no stim"], rotation=90, va="center")
    axB.set_xlabel("Time (hours)")
    axB.set_title("Peak raster", fontsize=9)
    axB.text(-0.16, 1.06, "B", transform=axB.transAxes, fontweight="bold", fontsize=12)

    # Panel C: phase histogram ---------------------------------------------------
    axC = fig.add_subplot(gs[1, 1])
    per = args.stim_period_min
    nbins = 16
    edges = np.linspace(0, per, nbins + 1)
    for tarr, col, lab, R in (
        (nt, COL_NOISE, "no stim", R_noise),
        (et, COL_ENTRAIN, "entrained", R_entr),
    ):
        ph = tarr % per
        h, _ = np.histogram(ph, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axC.step(np.r_[0, centers, per], np.r_[h[0], h, h[-1]], where="mid",
                 color=col, lw=1.4, label=f"{lab} (R={R:.2f})")
    axC.axvspan(0, args.pulse_width_min, color=COL_STIM, alpha=0.30, lw=0)
    axC.set_xlim(0, per)
    axC.set_ylim(0, axC.get_ylim()[1] * 1.18)
    axC.set_xlabel("Peak phase within cue cycle (min)")
    axC.set_ylabel("Peak density")
    axC.set_title("Phase locking", fontsize=9)
    axC.legend(loc="upper right", frameon=False, fontsize=7)
    axC.text(-0.20, 1.06, "C", transform=axC.transAxes, fontweight="bold", fontsize=12)

    # Panels D & E: frequency response (resonance), weak sine probe --------------
    x = sweep_df["period_hr"].to_numpy()
    # mark the resonance peak (the period that maximises the entrained amplitude)
    peak_idx = int(np.nanargmax(sweep_df["peak_to_trough"].to_numpy()))
    res_hr = float(x[peak_idx])
    res_min = float(sweep_df["period_min"].to_numpy()[peak_idx])

    def _mark_natural(ax):
        ax.axvline(res_hr, color="0.35", ls="--", lw=1.0)
        ax.plot([res_hr], [ax.lines[0].get_ydata()[peak_idx]],
                "o", color="0.25", ms=5, zorder=5)

    # D: phase-locking R
    axD = fig.add_subplot(gs[2, 0])
    axD.plot(x, sweep_df["rayleigh_R"], "o-", color=COL_NOISE, lw=1.6, ms=4)
    _mark_natural(axD)
    axD.set_xlabel("Stimulus period (hours)")
    axD.set_ylabel("Rayleigh R (phase-locking)")
    axD.set_ylim(0, 1.02)
    axD.set_title("Phase-locking vs period", fontsize=9)
    axD.text(-0.20, 1.05, "D", transform=axD.transAxes, fontweight="bold", fontsize=12)

    # E: peak-to-trough amplitude (resonance)
    axE = fig.add_subplot(gs[2, 1])
    axE.plot(x, sweep_df["peak_to_trough"], "o-", color=COL_ENTRAIN, lw=1.6, ms=4)
    axE.fill_between(x, 0, sweep_df["peak_to_trough"], color=COL_ENTRAIN, alpha=0.12)
    _mark_natural(axE)
    axE.text(res_hr, axE.get_ylim()[1] * 0.97,
             f" natural ultradian\n period ~{res_min:.0f} min ({res_hr:.1f} h)",
             fontsize=7, color="0.3", va="top", ha="left")
    axE.set_xlabel("Stimulus period (hours)")
    axE.set_ylabel("Peak-to-trough\namplitude (a.u.)")
    axE.set_ylim(bottom=0)
    axE.set_title("Resonance: response peaks at the natural period", fontsize=9)
    axE.text(-0.22, 1.05, "E", transform=axE.transAxes, fontweight="bold", fontsize=12)

    for ext in ("png", "pdf"):
        fig.savefig(args.out / "figures" / f"figure_5.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
