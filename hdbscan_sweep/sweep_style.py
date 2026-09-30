"""Colours for the HDBSCAN-sweep paper figures, taken from ALGO_COLORS /
CATEGORICAL in CaloClouds-3/projection/plot_distributions_from_grid.py.
References use their exact projection-plot colour; an m_s line takes the colour
of the trained setting with that m_s where there is one, else an overflow one."""

import matplotlib.pyplot as plt

REF_COLORS = {
    "merge within cell": "#0072b2",  # merge_within_cell
    "merge within regular subcell": "#e69f00",  # merge_within_regular_subcell
}
MS_COLORS = {
    2: "#9a9a92",
    3: "#009e73",  # hdbscan_ms3_mcs10
    5: "#a1215a",
    8: "#cc79a7",  # hdbscan_ms8_mcs40
    12: "#7a21dd",  # hdbscan_ms12_mcs12
    20: "#455a64",
    40: "#56b4e9",
    60: "#0b0b0b",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "axes.edgecolor": "#0b0b0b",
            "grid.color": "#e1e0d9",
        }
    )


# m_s settings left out of the paper figures
EXCLUDE_MS = {40, 60}
