"""Fit the model to the ultradian-band PSD of cortisol and/or ACTH.

Unlike the peak-statistics fitters, this targets the power spectral density
directly. It fits free parameters (e.g. the secretion rates b1/b2/b3 and the
drive-noise epsilon) so the model's simulated ultradian-band PSD matches the
pooled PSD of one or more subjects, for one or more signals.

Processing (identical for data and every model replicate, per signal):
  1. subtract a 24h + 12h harmonic, then normalize (see psd_mode),
  2. Hann-windowed periodogram, frequency in cycles/hour,
  3. restrict to the ultradian band (default periods 60-120 min).

psd_mode controls normalization / what is identifiable:
  - "zscore":     divide residual by its SD (scale-free shape; noise amplitude
                  is normalized away). Loss = band shape + in-band power fraction.
  - "fractional": divide residual by the MEAN (real, unit-invariant relative
                  fluctuation, CV preserved). Loss = absolute band PSD match.
                  This is the right choice for multi-signal fits (cortisol nmol/L
                  and ACTH pg/mL are not on a common absolute scale).
  - "absolute":   no normalization; the model is scaled into data units by the
                  fitted secretion rates and pinned with a mean-cortisol anchor.
                  Single-signal only (a shared absolute scale is ill-defined
                  across signals with different physical units).

The model is simulated onto each subject's own grid, so its periodogram shares
frequency bins with the data and pools bin-by-bin.
"""

from __future__ import annotations

from functools import partial
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from ..config import dump_yaml
from ..data.registry import get_dataset_spec, load_dataset
from ..model.three_state_gr_delay import ThreeStateGRDelayModel, build_drive
from ..plotting import setup_nature_style
from ..simulate.engine import simulate_trajectory_fit_arrays
from .habs_dual_peak_stats import CONFIG_FREE_PARAM_PATHS, _get_nested, _set_nested

COL_DATA = "#2F5C85"
COL_MODEL = "#C85C3A"

# Signal name -> simulated model state.
SIGNAL_STATE = {"Cortisol": "x3", "ACTH": "x2"}

# Free-param -> config-path map: the shared map plus the secretion rates b1/b2/b3.
FREE_PARAM_PATHS: dict[str, tuple[str, ...]] = {
    **CONFIG_FREE_PARAM_PATHS,
    "b1": ("model", "params", "b1"),
    "b2": ("model", "params", "b2"),
    "b3": ("model", "params", "b3"),
}


# --------------------------------------------------------------------------- #
# PSD helpers (mirrors notebooks/scratch/model_psd_vs_data_ultradian_band.py)  #
# --------------------------------------------------------------------------- #
def _harmonic_residual(t_min: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract a 24h+12h harmonic; return (raw residual, mean of y)."""
    w24 = 2.0 * np.pi / 1440.0
    w12 = 2.0 * np.pi / 720.0
    D = np.column_stack([
        np.ones_like(t_min),
        np.sin(w24 * t_min), np.cos(w24 * t_min),
        np.sin(w12 * t_min), np.cos(w12 * t_min),
    ])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    return resid, float(np.mean(y))


def normalize_residual(t_min: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    """Detrend then normalize per fit mode (see module docstring)."""
    resid, ymean = _harmonic_residual(t_min, y)
    if mode == "zscore":
        sd = resid.std()
        return resid / sd if sd > 0 else resid
    if mode == "fractional":
        return resid / ymean if ymean != 0 else resid
    if mode == "absolute":
        return resid
    raise ValueError(f"Unknown psd_mode: {mode}")


def periodogram_cph(t_min: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Hann-windowed periodogram of z; frequency in cycles/hour."""
    dt = float(np.median(np.diff(t_min)))
    n = len(z)
    tg = t_min[0] + dt * np.arange(n)
    zg = np.interp(tg, t_min, z)
    zg = zg - zg.mean()
    win = np.hanning(n)
    X = np.fft.rfft(zg * win)
    f = np.fft.rfftfreq(n, d=dt) * 60.0
    psd = (np.abs(X) ** 2) / (np.sum(win ** 2))
    return f[1:], psd[1:]


def _frac_shape(psd: np.ndarray, band: np.ndarray) -> tuple[float, np.ndarray]:
    """In-band power fraction and unit-area-over-band shape."""
    total = float(psd.sum())
    in_band = float(psd[band].sum())
    frac = in_band / total if total > 0 else 0.0
    shape = psd[band] / in_band if in_band > 0 else np.zeros(int(band.sum()))
    return frac, shape


def _band_psd_term(psd_data: np.ndarray, psd_model: np.ndarray, band: np.ndarray) -> float:
    """Relative SSE of the band PSD (captures band shape AND magnitude)."""
    pd_b, pm_b = psd_data[band], psd_model[band]
    denom = float(np.sum(pd_b ** 2))
    return float(np.sum((pd_b - pm_b) ** 2) / denom) if denom > 0 else float(np.sum((pd_b - pm_b) ** 2))


def _signal_loss(mode, psd_d, psd_m, band, shape_d, frac_d, shape_w, power_w) -> float:
    if mode == "zscore":
        frac_m, shape_m = _frac_shape(psd_m, band)
        return shape_w * float(np.sum((shape_d - shape_m) ** 2)) + power_w * float((frac_d - frac_m) ** 2)
    return _band_psd_term(psd_d, psd_m, band)


def _build_model(model_params: dict[str, Any]) -> ThreeStateGRDelayModel:
    p = model_params
    return ThreeStateGRDelayModel(
        a1=float(p["a1"]), a2=float(p["a2"]), a3=float(p["a3"]),
        b1=float(p["b1"]), b2=float(p["b2"]), b3=float(p["b3"]),
        kgr=float(p["kgr"]), tau_min=float(p["tau_min"]),
        x3_floor=float(p["x3_floor"]), hill_coeff=float(p["hill_coeff"]),
        initial_state=tuple(float(x) for x in p["initial_state"]),
    )


# --------------------------------------------------------------------------- #
# Forward PSD given a parameter vector                                        #
# --------------------------------------------------------------------------- #
def _model_pooled_psd(ctx: dict[str, Any], theta: np.ndarray):
    """Per-signal pooled model PSD and per-signal per-subject mean."""
    model_params = dict(ctx["model_params"])
    drive_params = dict(ctx["drive_params"])
    noise_epsilons = dict(ctx["noise_epsilons"])
    for idx, name in enumerate(ctx["free_params"]):
        path = FREE_PARAM_PATHS[name]
        if path[:2] == ("model", "params"):
            model_params[path[-1]] = float(theta[idx])
        elif path[:2] == ("drive", "params"):
            drive_params[path[-1]] = float(theta[idx])
        elif path[:2] == ("runtime", "noise_epsilons"):
            noise_epsilons[path[-1]] = float(theta[idx])
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"fit_ultradian_psd cannot map free param '{name}'")

    model = _build_model(model_params)
    mode = ctx["psd_mode"]
    signals = ctx["signals"]
    nf = ctx["psd_len"]
    psds: dict[str, list[np.ndarray]] = {s: [] for s in signals}
    means: dict[str, list[float]] = {s: [] for s in signals}
    for si, subj in enumerate(ctx["subjects"]):
        params = dict(drive_params)
        params["dataset"] = subj["dataset"]
        params["variant"] = subj["variant"]
        params["series_id"] = subj["sid"]
        drive = build_drive(ctx["drive_kind"], params)
        t = subj["t"]
        span = float(t[-1] - t[0])
        subj_means: dict[str, list[float]] = {s: [] for s in signals}
        for rep in range(ctx["n_reps"]):
            out = simulate_trajectory_fit_arrays(
                model, drive,
                dt_min=ctx["dt_min"], warmup_min=ctx["warmup_min"],
                duration_min=span + 60.0,
                seed=ctx["seed"] + 1000 * si + rep,
                noise_locations=ctx["noise_locations"],
                noise_epsilons=noise_epsilons,
                noise_form=ctx["noise_form"],
            )
            tm = out["time_min"] - out["time_min"][0]
            for s in signals:
                ym = np.interp(t - t[0], tm, out[SIGNAL_STATE[s]])
                subj_means[s].append(float(np.mean(ym)))
                z = normalize_residual(t, ym, mode)
                _f, p = periodogram_cph(t, z)
                psds[s].append(p[:nf])
        for s in signals:
            means[s].append(float(np.mean(subj_means[s])))
    pooled = {s: np.mean(psds[s], axis=0) for s in signals}
    mean_arr = {s: np.asarray(means[s]) for s in signals}
    return pooled, mean_arr


def _objective(theta: np.ndarray, ctx: dict[str, Any]) -> float:
    psd_model, means_model = _model_pooled_psd(ctx, theta)
    band = ctx["band"]
    loss = 0.0
    for s in ctx["signals"]:
        loss += ctx["signal_weights"][s] * _signal_loss(
            ctx["psd_mode"], ctx["psd_data"][s], psd_model[s], band,
            ctx["shape_data"][s], ctx["frac_data"][s],
            ctx["shape_weight"], ctx["power_weight"],
        )
    if ctx["psd_mode"] == "absolute":
        for s in ctx["signals"]:
            md = ctx["data_means"][s]
            rel = (means_model[s] - md) / md
            loss += ctx["mean_weight"] * float(np.mean(rel ** 2))
    return float(loss)


# --------------------------------------------------------------------------- #
# Data side                                                                   #
# --------------------------------------------------------------------------- #
def resolve_series_ids(dataset, variant, raw, max_subjects=0) -> list[str]:
    """Expand 'all' to every subject id; otherwise pass through. Optional cap."""
    if raw in ("all", ["all"], None):
        df = load_dataset(dataset, variant)
        idc = get_dataset_spec(dataset).id_col
        ids = [str(s) for s in sorted(df[idc].unique())]
    else:
        ids = [str(s) for s in raw]
    if max_subjects and max_subjects > 0:
        ids = ids[:max_subjects]
    return ids


def _build_subjects(datasets_cfg, signals) -> list[dict]:
    """Unified subject list across one or more datasets.

    Each subject carries its dataset/variant/id, time grid, and per-signal data
    so its model drive auto-loads from the right dataset and the data PSD is
    computed from the matching columns.
    """
    subjects: list[dict] = []
    for d in datasets_cfg:
        name, variant = str(d["name"]), str(d["variant"])
        sids = resolve_series_ids(name, variant, d.get("series_ids", "all"),
                                  int(d.get("max_subjects", 0)))
        df = load_dataset(name, variant)
        spec = get_dataset_spec(name)
        col = {sg.name: sg.column for sg in spec.signals}
        id_str = df[spec.id_col].astype(str)
        for sid in sids:
            sub = df[id_str == str(sid)].sort_values("time_min")
            t = sub["time_min"].to_numpy(float)
            y = {s: sub[col[s]].to_numpy(float) for s in signals}
            subjects.append({"dataset": name, "variant": variant, "sid": sid, "t": t, "y": y})
    return subjects


def _data_psd(subjects, signals, mode):
    """Pooled data PSD per signal, common freq grid, per-signal means, psd_len.

    Records share frequency spacing (all 24 h); mixed sampling gives different
    Nyquist lengths, so pool over the common band by truncating to the minimum.
    """
    raw: dict[str, list[np.ndarray]] = {s: [] for s in signals}
    means: dict[str, list[float]] = {s: [] for s in signals}
    freqs: list[np.ndarray] = []
    for subj in subjects:
        t = subj["t"]
        for s in signals:
            y = subj["y"][s]
            ymean = float(np.nanmean(y))
            means[s].append(ymean)
            z = normalize_residual(t, np.nan_to_num(y, nan=ymean), mode)
            f, p = periodogram_cph(t, z)
            raw[s].append(p)
            if s == signals[0]:
                freqs.append(f)
    nf = min(len(f) for f in freqs)
    freq = freqs[int(np.argmin([len(f) for f in freqs]))][:nf]
    pooled = {s: np.mean([p[:nf] for p in raw[s]], axis=0) for s in signals}
    mean_arr = {s: np.asarray(means[s]) for s in signals}
    return pooled, freq, mean_arr, nf


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def fit_ultradian_psd_from_config(
    config: dict[str, Any],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or logging.getLogger("hpa_model")
    out_dir = Path(out_dir)
    artifacts = out_dir / "artifacts"
    figures = out_dir / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    dataset = config["dataset"]["name"]
    variant = config["dataset"]["variant"]
    fit_cfg = config["fit"]
    free_params = [str(p) for p in fit_cfg["free_params"]]
    psd_mode = str(fit_cfg.get("psd_mode", "zscore"))
    signals = [str(s) for s in fit_cfg.get("signals", ["Cortisol"])]
    sw_cfg = fit_cfg.get("signal_weights", {}) or {}
    signal_weights = {s: float(sw_cfg.get(s, 1.0)) for s in signals}

    # One or more datasets to pool. Fall back to the single config.dataset.
    datasets_cfg = fit_cfg.get("datasets") or [{
        "name": dataset, "variant": variant,
        "series_ids": fit_cfg.get("series_ids", "all"),
        "max_subjects": fit_cfg.get("max_subjects", 0),
    }]
    subjects = _build_subjects(datasets_cfg, signals)

    min_period = float(fit_cfg.get("band_min_period_min", 60.0))
    max_period = float(fit_cfg.get("band_max_period_min", 120.0))
    band_lo, band_hi = 60.0 / max_period, 60.0 / min_period

    psd_data, freq, data_means, psd_len = _data_psd(subjects, signals, psd_mode)
    band = (freq >= band_lo) & (freq <= band_hi)
    frac_data, shape_data = {}, {}
    for s in signals:
        frac_data[s], shape_data[s] = _frac_shape(psd_data[s], band)

    runtime = config.get("runtime", {})
    ctx = {
        "model_params": config["model"]["params"],
        "drive_kind": config["drive"]["kind"],
        "drive_params": config["drive"]["params"],
        "free_params": free_params,
        "subjects": subjects,
        "signals": signals,
        "signal_weights": signal_weights,
        "dt_min": float(config["solver"]["dt_min"]),
        "warmup_min": float(config["solver"]["warmup_min"]),
        "n_reps": int(fit_cfg.get("n_reps", 8)),
        "seed": int(runtime.get("seed", 42)),
        "noise_locations": [str(x) for x in runtime.get("noise_locations", [])],
        "noise_epsilons": {str(k): float(v) for k, v in (runtime.get("noise_epsilons", {}) or {}).items()},
        "noise_form": str(runtime.get("noise_form", "lognormal")),
        "psd_mode": psd_mode,
        "psd_len": psd_len,
        "band": band,
        "psd_data": psd_data,
        "data_means": data_means,
        "frac_data": frac_data,
        "shape_data": shape_data,
        "shape_weight": float(fit_cfg.get("shape_weight", 1.0)),
        "power_weight": float(fit_cfg.get("power_weight", 1.0)),
        "mean_weight": float(fit_cfg.get("mean_weight", 1.0)),
    }

    bounds = [tuple(float(v) for v in fit_cfg["bounds"][p]) for p in free_params]
    opt_cfg = fit_cfg.get("optimizer", {})
    opt_name = str(opt_cfg.get("name", "differential_evolution"))
    workers = int(opt_cfg.get("workers", -1))
    # Start point from the config's current param values (e.g. kgr=5), clipped to bounds.
    x0 = np.array([_get_nested(config, FREE_PARAM_PATHS[n]) for n in free_params], dtype=float)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    log.info("fit_ultradian_psd[%s/%s]: %d subjects (%d datasets), signals=%s, %d reps, "
             "free=%s, x0=%s, band %.0f-%.0f min", psd_mode, opt_name, len(subjects),
             len(datasets_cfg), signals, ctx["n_reps"], free_params, list(x0), min_period, max_period)

    obj = partial(_objective, ctx=ctx)
    if opt_name in ("nelder_mead", "nelder-mead", "local"):
        result = minimize(
            obj, x0, method="Nelder-Mead", bounds=bounds,
            options={"maxiter": int(opt_cfg.get("maxiter", 200)),
                     "xatol": float(opt_cfg.get("xatol", 1e-3)),
                     "fatol": float(opt_cfg.get("fatol", 1e-4))},
        )
    else:
        result = differential_evolution(
            obj, bounds,
            maxiter=int(opt_cfg.get("maxiter", 40)),
            popsize=int(opt_cfg.get("popsize", 12)),
            tol=float(opt_cfg.get("tol", 0.005)),
            seed=int(runtime.get("seed", 42)),
            polish=bool(opt_cfg.get("polish", True)),
            x0=x0,
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
        )
    theta = result.x
    params = {name: float(theta[i]) for i, name in enumerate(free_params)}
    log.info("fit_ultradian_psd: best %s  objective=%.5f", params, float(result.fun))

    # Final metrics at the optimum.
    psd_model, means_model = _model_pooled_psd(ctx, theta)
    f_band = freq[band]
    per_signal = {}
    for s in signals:
        r = float(np.corrcoef(psd_data[s][band], psd_model[s][band])[0, 1])
        per_signal[s] = {
            "inband_pearson_r": r,
            "peak_period_data_min": float(60.0 / f_band[np.argmax(psd_data[s][band])]),
            "peak_period_model_min": float(60.0 / f_band[np.argmax(psd_model[s][band])]),
            "mean_data": float(np.mean(data_means[s])),
            "mean_model": float(np.mean(means_model[s])),
        }

    primary = signals[0]
    summary_row = {
        **params,
        "objective_value": float(result.fun),
        "psd_mode": psd_mode,
        "signals": "+".join(signals),
        "inband_pearson_r": per_signal[primary]["inband_pearson_r"],
        "peak_period_data_min": per_signal[primary]["peak_period_data_min"],
        "peak_period_model_min": per_signal[primary]["peak_period_model_min"],
        "n_subjects": len(subjects),
        "n_datasets": len(datasets_cfg),
        "n_reps": ctx["n_reps"],
    }
    for s in signals:
        for k, v in per_signal[s].items():
            summary_row[f"{s}_{k}"] = v

    # ----- artifacts -----
    fitted_config = _write_fitted_config(config, free_params, theta, artifacts)
    pd.DataFrame([{"param": k, "value": v} for k, v in params.items()]
                 + [{"param": "objective", "value": float(result.fun)}]
                 ).to_csv(artifacts / "fit_params.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(artifacts / "fit_summary.csv", index=False)
    psd_cols = {"freq_cph": freq, "period_min": 60.0 / freq, "in_band": band}
    for s in signals:
        psd_cols[f"psd_data_{s}"] = psd_data[s]
        psd_cols[f"psd_model_{s}"] = psd_model[s]
    pd.DataFrame(psd_cols).to_csv(artifacts / "psd_comparison.csv", index=False)

    # ----- figures -----
    _plot_psd(freq, psd_data, psd_model, band, band_lo, band_hi, signals,
              per_signal, params, psd_mode, figures / "psd_data_vs_model.png")

    return {"summary_row": summary_row, "params": params,
            "per_signal": per_signal, "fitted_config": fitted_config}


def _write_fitted_config(config, free_params, theta, artifacts) -> dict[str, Any]:
    from copy import deepcopy
    fitted = deepcopy(config)
    for i, name in enumerate(free_params):
        _set_nested(fitted, FREE_PARAM_PATHS[name], float(theta[i]))
    (artifacts / "fitted_config.yaml").write_text(dump_yaml(fitted))
    return fitted


def _plot_psd(freq, psd_data, psd_model, band, band_lo, band_hi, signals,
              per_signal, params, mode, path) -> None:
    setup_nature_style()
    nsig = len(signals)
    fig, axes = plt.subplots(nsig, 2, figsize=(9.6, 3.7 * nsig), squeeze=False)
    fb = freq[band]
    ylab = {"zscore": "PSD of z-scored residual",
            "fractional": "PSD of fractional residual",
            "absolute": "PSD of residual"}.get(mode, "PSD")
    for row, s in enumerate(signals):
        axA, axB = axes[row]
        axA.plot(freq, psd_data[s], color=COL_DATA, lw=1.6, label="data")
        axA.plot(freq, psd_model[s], color=COL_MODEL, lw=1.6, ls="--", label="model")
        axA.axvspan(band_lo, band_hi, color="0.9", zorder=0)
        axA.set_xlim(0, 1.6)
        axA.set_xlabel("frequency (cycles/hour)")
        axA.set_ylabel(f"{s}\n{ylab}")
        axA.set_title(f"{s}: full spectrum", fontsize=9)
        axA.legend(frameon=False, fontsize=7)

        if mode == "zscore":
            sd = psd_data[s][band] / np.trapz(psd_data[s][band], fb)
            sm = psd_model[s][band] / np.trapz(psd_model[s][band], fb)
            ylab_b = "normalized band PSD"
        else:
            sd, sm = psd_data[s][band], psd_model[s][band]
            ylab_b = "band PSD"
        axB.plot(fb, sd, color=COL_DATA, lw=1.8, marker="o", ms=3, label="data")
        axB.plot(fb, sm, color=COL_MODEL, lw=1.8, ls="--", marker="s", ms=3, label="model")
        axB.set_xlabel("frequency (cycles/hour)")
        axB.set_ylabel(ylab_b)
        r = per_signal[s]["inband_pearson_r"]
        axB.set_title(f"{s}: ultradian band (r={r:.2f})", fontsize=9)
        secx = axB.secondary_xaxis("top", functions=(
            lambda f: 60.0 / np.where(f <= 0, np.nan, f),
            lambda T: 60.0 / np.where(T <= 0, np.nan, T)))
        secx.set_ticks([120, 90, 70, 60])
        secx.tick_params(labelsize=6.5)
        axB.legend(frameon=False, fontsize=7)
        for ax in (axA, axB):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    pstr = ", ".join(f"{k}={v:.3g}" for k, v in params.items())
    fig.suptitle(f"Ultradian-band PSD: model vs data  ({mode})\n{pstr}", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
