"""CLI entry-point for peak_amplitude_direct_rayleigh_fit."""

import matplotlib
matplotlib.use("Agg")

from hpa_model.analysis.plotting.peak_amplitude_direct_rayleigh_fit import main

if __name__ == "__main__":
    main()
