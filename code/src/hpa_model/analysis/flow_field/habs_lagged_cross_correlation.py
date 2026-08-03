"""Lagged ACTH-cortisol cross-correlation analysis on packaged HABS data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...config import dump_yaml
from ...data.registry import get_dataset_spec, load_dataset


@dataclass(frozen=True)
class LaggedCrossCorrelationSettings:
    dataset: str = "habs"
    variant: str = "shifted"
    normalize: str = "per_id_zscore"
    taus_min: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    min_pairs: int = 4


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("hpa_model.habs_lagged_cross_correlation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
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


def _prepare_run_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _normalize_subject_signals(acth: np.ndarray, cortisol: np.ndarray, *, normalize: str) -> tuple[np.ndarray, np.ndarray]:
    acth = np.asarray(acth, dtype=float)
    cortisol = np.asarray(cortisol, dtype=float)
    if normalize == "raw":
        return acth, cortisol
    if normalize == "per_id_zscore":
        return _zscore(acth), _zscore(cortisol)
    raise ValueError(f"Unsupported normalize mode: {normalize}")


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if not np.isfinite(x_std) or not np.isfinite(y_std) or x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_subject_lagged_correlation(
    time_min: np.ndarray,
    acth: np.ndarray,
    cortisol: np.ndarray,
    *,
    taus_min: tuple[float, ...],
    min_pairs: int = 4,
) -> pd.DataFrame:
    time_min = np.asarray(time_min, dtype=float)
    acth = np.asarray(acth, dtype=float)
    cortisol = np.asarray(cortisol, dtype=float)
    if not (time_min.ndim == acth.ndim == cortisol.ndim == 1):
        raise ValueError("time_min, acth, and cortisol must be one-dimensional")
    if not (len(time_min) == len(acth) == len(cortisol)):
        raise ValueError("time_min, acth, and cortisol must have equal length")

    finite = np.isfinite(time_min) & np.isfinite(acth) & np.isfinite(cortisol)
    time_min = time_min[finite]
    acth = acth[finite]
    cortisol = cortisol[finite]
    if time_min.size < max(2, int(min_pairs)):
        return pd.DataFrame(columns=["tau_min", "corr", "n_pairs"])

    order = np.argsort(time_min)
    time_min = time_min[order]
    acth = acth[order]
    cortisol = cortisol[order]

    rows: list[dict[str, float]] = []
    min_time = float(time_min.min())
    max_time = float(time_min.max())
    for tau_min in taus_min:
        tau = float(tau_min)
        query_time = time_min + tau
        valid = (query_time >= min_time) & (query_time <= max_time)
        n_pairs = int(np.sum(valid))
        corr = float("nan")
        if n_pairs >= int(min_pairs):
            cortisol_shifted = np.interp(query_time[valid], time_min, cortisol)
            corr = _pearson_corr(acth[valid], cortisol_shifted)
        rows.append(
            {
                "tau_min": tau,
                "corr": corr,
                "n_pairs": float(n_pairs),
            }
        )
    return pd.DataFrame(rows)


def _build_resolved_config(settings: LaggedCrossCorrelationSettings) -> dict[str, Any]:
    return {
        "task": "plot_habs_lagged_cross_correlation",
        "dataset": {"name": settings.dataset, "variant": settings.variant},
        "normalize": settings.normalize,
        "taus_min": [float(tau) for tau in settings.taus_min],
        "min_pairs": int(settings.min_pairs),
        "correlation": "pearson",
        "definition": "corr(ACTH(t), cortisol(t + tau))",
    }


def _plot_lagged_correlation(summary: pd.DataFrame, *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tau = summary["tau_min"].to_numpy(dtype=float)
    mean_corr = summary["mean_subject_corr"].to_numpy(dtype=float)
    if "subject_corr_sd" in summary:
        sd = summary["subject_corr_sd"].to_numpy(dtype=float)
        ax.fill_between(tau, mean_corr - sd, mean_corr + sd, color="#94d2bd", alpha=0.25, lw=0.0)
    ax.plot(tau, mean_corr, color="#0a9396", marker="o", lw=2.2, label="Mean subject corr")

    finite_mean = np.isfinite(mean_corr)
    if finite_mean.any():
        peak_idx = int(np.nanargmax(mean_corr))
        peak_tau = float(tau[peak_idx])
        ax.axvline(
            peak_tau,
            color="#0a9396",
            ls="--",
            lw=1.4,
            alpha=0.9,
            label=rf"Peak $\tau^*$ = {peak_tau:.1f} min",
        )

    ax.set_xlabel("Cortisol lead tau (min)")
    ax.set_ylabel("Correlation")
    ax.set_title("Lagged corr(ACTH(t), cortisol(t + tau))")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_lagged_cross_correlation(settings: LaggedCrossCorrelationSettings, out_dir: Path) -> dict[str, pd.DataFrame]:
    _prepare_run_dirs(out_dir)
    logger = _setup_logging(out_dir / "logs" / "run.log")

    resolved_config = _build_resolved_config(settings)
    (out_dir / "resolved_config.yaml").write_text(dump_yaml(resolved_config))

    manifest = {
        "task": "plot_habs_lagged_cross_correlation",
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(out_dir.resolve()),
        "config_path": str((out_dir / "resolved_config.yaml").resolve()),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    spec = get_dataset_spec(settings.dataset)
    if spec.signal_names != ("ACTH", "Cortisol"):
        raise ValueError(f"Dataset {settings.dataset!r} does not expose ACTH and Cortisol")

    frame = load_dataset(settings.dataset, settings.variant)
    frame = frame.loc[:, [spec.id_col, spec.time_col, "ACTH", "Cortisol"]].copy()
    frame = frame.dropna(subset=["ACTH", "Cortisol"]).copy()

    subject_rows: list[pd.DataFrame] = []
    pooled_rows: list[dict[str, float]] = []
    for subject_id, subject in frame.groupby(spec.id_col, sort=True):
        subject = subject.sort_values(spec.time_col).copy()
        acth, cortisol = _normalize_subject_signals(
            subject["ACTH"].to_numpy(dtype=float),
            subject["Cortisol"].to_numpy(dtype=float),
            normalize=settings.normalize,
        )
        subject_corr = compute_subject_lagged_correlation(
            subject[spec.time_col].to_numpy(dtype=float),
            acth,
            cortisol,
            taus_min=settings.taus_min,
            min_pairs=settings.min_pairs,
        )
        if subject_corr.empty:
            continue
        subject_corr.insert(0, "ID", subject_id)
        subject_rows.append(subject_corr)

        time_min = subject[spec.time_col].to_numpy(dtype=float)
        min_time = float(np.min(time_min))
        max_time = float(np.max(time_min))
        for tau_min in settings.taus_min:
            tau = float(tau_min)
            query_time = time_min + tau
            valid = (query_time >= min_time) & (query_time <= max_time)
            if int(np.sum(valid)) < int(settings.min_pairs):
                continue
            cortisol_shifted = np.interp(query_time[valid], time_min, cortisol)
            pooled_rows.append(
                pd.DataFrame(
                    {
                        "tau_min": tau,
                        "acth": acth[valid],
                        "cortisol_shifted": cortisol_shifted,
                    }
                )
            )
        logger.info("Processed subject %s", str(subject_id))

    if not subject_rows:
        raise ValueError("No valid subject rows were available for lagged correlation")

    subject_frame = pd.concat(subject_rows, ignore_index=True)

    summary_rows: list[dict[str, float]] = []
    for tau_min, tau_frame in subject_frame.groupby("tau_min", sort=True):
        valid_subject_corr = tau_frame["corr"].to_numpy(dtype=float)
        valid_subject_corr = valid_subject_corr[np.isfinite(valid_subject_corr)]
        pooled_corr = float("nan")
        pooled_n_pairs = 0
        pooled_tau_frames = [frame for frame in pooled_rows if float(frame["tau_min"].iloc[0]) == float(tau_min)]
        if pooled_tau_frames:
            pooled_frame = pd.concat(pooled_tau_frames, ignore_index=True)
            pooled_n_pairs = int(len(pooled_frame))
            pooled_corr = _pearson_corr(
                pooled_frame["acth"].to_numpy(dtype=float),
                pooled_frame["cortisol_shifted"].to_numpy(dtype=float),
            )
        summary_rows.append(
            {
                "tau_min": float(tau_min),
                "mean_subject_corr": float(np.mean(valid_subject_corr)) if valid_subject_corr.size else float("nan"),
                "median_subject_corr": float(np.median(valid_subject_corr)) if valid_subject_corr.size else float("nan"),
                "subject_corr_sd": float(np.std(valid_subject_corr)) if valid_subject_corr.size else float("nan"),
                "n_subjects": int(valid_subject_corr.size),
                "mean_subject_pairs": float(np.mean(tau_frame["n_pairs"].to_numpy(dtype=float))),
                "pooled_corr": pooled_corr,
                "pooled_n_pairs": pooled_n_pairs,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("tau_min").reset_index(drop=True)
    summary.to_csv(out_dir / "artifacts" / "lagged_cross_correlation_summary.csv", index=False)
    subject_frame.to_csv(out_dir / "artifacts" / "lagged_cross_correlation_subjects.csv", index=False)
    _plot_lagged_correlation(summary, output_path=out_dir / "figures" / "lagged_cross_correlation.png")
    _plot_lagged_correlation(summary, output_path=out_dir / "figures" / "lagged_cross_correlation.pdf")

    best_mean = summary.loc[summary["mean_subject_corr"].idxmax()]
    best_pooled = summary.loc[summary["pooled_corr"].idxmax()]
    readme_lines = [
        "# HABS Lagged ACTH-Cortisol Cross-Correlation",
        "",
        f"- Dataset: `{settings.dataset}` ({settings.variant})",
        f"- Normalization: `{settings.normalize}`",
        f"- Correlation definition: `corr(ACTH(t), cortisol(t + tau))`",
        f"- Taus (min): {', '.join(f'{float(tau):g}' for tau in settings.taus_min)}",
        f"- Minimum aligned pairs per subject: {int(settings.min_pairs)}",
        f"- Peak mean subject correlation at tau={float(best_mean['tau_min']):.1f} min with value {float(best_mean['mean_subject_corr']):.6f}",
        f"- Peak pooled correlation at tau={float(best_pooled['tau_min']):.1f} min with value {float(best_pooled['pooled_corr']):.6f}",
        "",
        "Artifacts:",
        "- `artifacts/lagged_cross_correlation_summary.csv`",
        "- `artifacts/lagged_cross_correlation_subjects.csv`",
        "- `figures/lagged_cross_correlation.png`",
        "- `figures/lagged_cross_correlation.pdf`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines))

    logger.info(
        "Completed lagged cross-correlation across %d taus; peak mean subject corr at tau=%.1f min",
        len(summary),
        float(best_mean["tau_min"]),
    )
    return {"summary": summary, "subjects": subject_frame}


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run lagged corr(ACTH(t), cortisol(t + tau)) on HABS data.")
    parser.add_argument("--dataset", default="habs")
    parser.add_argument("--variant", default="shifted")
    parser.add_argument("--normalize", default="per_id_zscore", choices=["per_id_zscore", "raw"])
    parser.add_argument("--taus-min", nargs="+", type=float, default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument(
        "--out",
        default="experiments/runs/plot_habs_lagged_cross_correlation",
        help="Output run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = LaggedCrossCorrelationSettings(
        dataset=str(args.dataset),
        variant=str(args.variant),
        normalize=str(args.normalize),
        taus_min=tuple(float(tau) for tau in args.taus_min),
        min_pairs=int(args.min_pairs),
    )
    run_lagged_cross_correlation(settings, Path(args.out))


if __name__ == "__main__":
    main()
