"""Summary and plotting utilities for run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def summarize_trajectory_frame(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "duration_min": float(frame["time_min"].max()),
        "n_rows": float(len(frame)),
        "x1_mean": float(frame["x1"].mean()),
        "x2_mean": float(frame["x2"].mean()),
        "x3_mean": float(frame["x3"].mean()),
        "x2_peak": float(frame["x2"].max()),
        "x3_peak": float(frame["x3"].max()),
        "u_mean": float(frame["u"].mean()),
    }


def plot_simulation_replicates(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for rep, rep_df in frame.groupby("rep"):
        axes[0].plot(rep_df["time_min"], rep_df["x1"], alpha=0.7, label=f"rep {rep}")
        axes[1].plot(rep_df["time_min"], rep_df["x2"], alpha=0.7)
        axes[2].plot(rep_df["time_min"], rep_df["x3"], alpha=0.7)
        axes[3].plot(rep_df["time_min"], rep_df["u"], alpha=0.7)
    axes[0].set_ylabel("CRH (x1)")
    axes[1].set_ylabel("ACTH (x2)")
    axes[2].set_ylabel("Cortisol (x3)")
    axes[3].set_ylabel("u(t)")
    axes[3].set_xlabel("Time (min)")
    axes[0].legend(loc="upper right")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _fit_summary_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""

    lines: list[str] = []
    if "tau_min" in summary:
        lines.append(f"tau={float(summary['tau_min']):.2f} min")
    if "drive_baseline" in summary:
        lines.append(f"baseline={float(summary['drive_baseline']):.3f}")
    if "drive_amplitude" in summary:
        lines.append(f"amplitude={float(summary['drive_amplitude']):.3f}")
    if "rmse_mean_acth" in summary:
        lines.append(f"ACTH mean RMSE={float(summary['rmse_mean_acth']):.3f}")
    if "rmse_mean_cortisol" in summary:
        lines.append(f"Cort mean RMSE={float(summary['rmse_mean_cortisol']):.3f}")
    if "rmse_cv_acth" in summary:
        lines.append(f"ACTH CV RMSE={float(summary['rmse_cv_acth']):.3f}")
    if "rmse_cv_cortisol" in summary:
        lines.append(f"Cort CV RMSE={float(summary['rmse_cv_cortisol']):.3f}")
    if "loss_mode" in summary:
        lines.append(f"loss={summary['loss_mode']}")
    return " | ".join(lines)


def plot_dual_fit(
    stats: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    summary: dict[str, Any] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    signal_info = [
        ("ACTH", "observed_mean_acth", "observed_std_acth", "sim_mean_acth", "sim_std_acth"),
        (
            "Cortisol",
            "observed_mean_cortisol",
            "observed_std_cortisol",
            "sim_mean_cortisol",
            "sim_std_cortisol",
        ),
    ]

    for ax, (label, obs_mean, obs_std, sim_mean, sim_std) in zip(axes, signal_info):
        x = stats["time_min"] / 60.0
        ax.plot(x, stats[obs_mean], label=f"Observed {label}", color="#1f77b4")
        ax.fill_between(
            x,
            stats[obs_mean] - stats[obs_std],
            stats[obs_mean] + stats[obs_std],
            alpha=0.2,
            color="#1f77b4",
        )
        ax.plot(x, stats[sim_mean], label=f"Simulated {label}", color="#d62728")
        ax.fill_between(
            x,
            stats[sim_mean] - stats[sim_std],
            stats[sim_mean] + stats[sim_std],
            alpha=0.2,
            color="#d62728",
        )
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle(title)
    fit_text = _fit_summary_text(summary)
    if fit_text:
        fig.text(0.5, 0.94, fit_text, ha="center", va="top", fontsize=9)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_run_readme(task: str, highlights: list[str]) -> str:
    lines = [f"# {task}", "", "## Highlights"]
    lines.extend(f"- {line}" for line in highlights)
    lines.append("")
    return "\n".join(lines)
