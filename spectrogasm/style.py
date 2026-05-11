"""Plot styling for scientific-poster quality figures.

Two anchor colors define the palette::

    PRIMARY   = "#870047"   # deep magenta — main signal, fitted curves
    SECONDARY = "#ff6aa7"   # bright pink  — points, secondary series

The remaining swatches are linear interpolations of those two anchors
(plus near-black and near-white extremes) so every figure stays on the
same hue family. ``apply_style()`` sets matplotlib ``rcParams`` to large
poster-friendly defaults; call it once at import time.

A diverging colormap ``CMAP`` and a sequential ``CMAP_SEQ`` are also
exported for spectra, residual maps, etc.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb


# --- Anchor colors --------------------------------------------------------

PRIMARY = "#870047"
SECONDARY = "#ff6aa7"

# Hand-picked extensions of the anchors. Dark = near-black with a magenta
# tint; light = washed pink for fills/error bands; ink = body text.
DARK = "#3a0020"
LIGHT = "#ffc2da"
INK = "#1a1014"
MUTED = "#7a5060"

# Ordered palette used by the matplotlib color cycle. PRIMARY first so the
# main data series picks it up by default; SECONDARY second for overlays.
PALETTE: list[str] = [PRIMARY, SECONDARY, DARK, MUTED, LIGHT]


# --- Continuous colormaps -------------------------------------------------

# Sequential: light pink -> primary -> dark. Use for monotonic quantities
# (intensity, time-since-first-night, etc.).
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "spectrogasm_seq", [LIGHT, SECONDARY, PRIMARY, DARK], N=256
)

# Diverging: dark <- primary <- white -> secondary -> light. Use for
# signed residuals.
CMAP = LinearSegmentedColormap.from_list(
    "spectrogasm_div", [DARK, PRIMARY, "#ffffff", SECONDARY, LIGHT], N=256
)


# --- Style application ----------------------------------------------------

_RC_POSTER: dict[str, object] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 220,
    "figure.dpi": 110,

    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.labelweight": "semibold",
    "axes.linewidth": 1.4,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "axes.titlesize": 16,
    "axes.labelsize": 14,

    "grid.color": "#dcc6d2",
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.6,

    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "lines.linewidth": 1.8,
    "lines.markersize": 6,

    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#dcc6d2",
    "legend.fontsize": 11,

    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,

    "figure.titlesize": 18,
    "figure.titleweight": "bold",
}


def apply_style() -> None:
    """Install the poster style as global matplotlib defaults."""
    mpl.rcParams.update(_RC_POSTER)


def style_axes(ax: plt.Axes) -> plt.Axes:
    """Per-axes polish: drop top/right spines, color the remaining ones."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)
    return ax


def with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    """Return ``color`` as an RGBA tuple with the given alpha."""
    r, g, b = to_rgb(color)
    return (r, g, b, alpha)


# Apply on import so any downstream ``import matplotlib.pyplot`` gets the
# poster defaults without having to call ``apply_style`` manually.
apply_style()
