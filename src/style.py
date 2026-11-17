"""Shared chart styling so every figure in the project looks consistent."""
import matplotlib.pyplot as plt

# Colour-blind-safe palette (blue, teal, amber, red, purple, grey)
PALETTE = ["#2563eb", "#0d9488", "#d97706", "#dc2626", "#7c3aed", "#6b7280"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d1d5db",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelcolor": "#374151",
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
        "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
