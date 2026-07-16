"""Shared plotting style. Imports nothing but matplotlib, so BOTH environments
(eztao side and jax side) can use it -- which is the point: AA_STYLE lived in
LightCurves.py and was copy-pasted into the Stage 1 notebook, and two copies of a
style dict drift.

A&A/ApJ conventions: ticks inward on all four sides, minor ticks on, no legend
frame, serif to match the LaTeX body text.
"""
AA_STYLE = {
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 130,
}


def useAA():
    """Apply the style. Call once per notebook."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(AA_STYLE)
    return AA_STYLE
