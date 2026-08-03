"""Shared plotting utilities and paper-ready styles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def setup_nature_style() -> None:
    """Configure a restrained seaborn-based style for manuscript figures."""
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        },
    )


def setup_paper_style() -> None:
    """Backward-compatible alias for the repo-wide default manuscript style."""
    setup_nature_style()


def apply_paper_style(ax: plt.Axes | None = None) -> None:
    """Apply paper-ready cleanups to a specific axis.
    
    Removes top/right spines, disables the grid, and clears the title.
    If ax is None, applies to plt.gca().
    """
    if ax is None:
        ax = plt.gca()
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    # ax.set_title("")
