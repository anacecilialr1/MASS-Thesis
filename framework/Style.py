"""Shared plotting style. Imports nothing but matplotlib, so BOTH environments
(eztao side and jax side) can use it 
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
