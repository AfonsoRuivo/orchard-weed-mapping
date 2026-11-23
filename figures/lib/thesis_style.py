"""Shared look for the Chapter 7 figures (see the figure plan).

Import this first:  from thesis_style import *
 - 15 cm wide figures, serif 8 pt text / 7 pt ticks, white background, vector PDF
 - ONE colour dictionary used by every figure (never hard-code a colour elsewhere)
 - helpers: panel labels, scale bar, north arrow, camera presets for the 3D views
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

CM = 1 / 2.54
FIG_W = 15 * CM                      # included at 0.9 * textwidth

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["FreeSans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "FreeSans", "mathtext.it": "FreeSans:italic", "mathtext.bf": "FreeSans:bold",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

# -- one colour dictionary for the whole chapter ------------------------------
# Wong / Okabe-Ito colour-blind-safe hues.
COL = {
    "trunk": "#8B5A2B",      # brown
    "weed": "#5B8C2A",       # olive green
    "path": "#D4A62A",       # muted gold
    "path_halo": "#FFFFFF",
    "liosam": "#CC79A7",     # reddish purple
    "kiss": "#0072B2",       # blue
    "dlio": "#D55E00",       # vermillion
    "yolo": "#334155",       # dark slate  } YOLO vs colour mask: always these two,
    "colour": "#00A6A6",     # teal        } disjoint from every other hue above
    "ground": "#DDD8CC",     # light ground plane
    "contour": "#9C8F78",
    "grey": "#8A8A8A",
    "gnss": "#444444",
}
# heights above ground: cividis without its near-black end, so low vegetation does not dominate
HEIGHT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "cividis_t", mpl.colormaps["cividis"](np.linspace(0.28, 1.0, 256)))

# -- camera presets (orthographic oblique, written down so they can be reproduced)
# az: yaw of the row-aligned frame [deg]; el: elevation above the horizon [deg].
CAMERAS = {
    "f1a_block": dict(az=14.0, el=38.0),
    "f1b_alley": dict(az=8.0, el=11.0),
}


def panel_label(ax, s, x=0.0, y=1.0, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom", **kw)


def scale_bar(ax, length, label, xy=(0.04, 0.05), color="k", lw=1.6, fs=7):
    """Scale bar in DATA units, positioned in axes fraction."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs = x0 + xy[0] * (x1 - x0); ys = y0 + xy[1] * (y1 - y0)
    ax.plot([xs, xs + length], [ys, ys], color=color, lw=lw, solid_capstyle="butt", zorder=20)
    ax.text(xs + length / 2, ys + 0.012 * (y1 - y0), label, ha="center", va="bottom",
            fontsize=fs, color=color, zorder=20,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def north_arrow(ax, dx, dy, xy=(0.93, 0.80), size=0.09, color="k"):
    """Arrow in axes fraction pointing along (dx, dy) in SCREEN directions."""
    n = np.hypot(dx, dy); dx, dy = dx / n, dy / n
    ax.add_patch(FancyArrowPatch(xy, (xy[0] + size * dx, xy[1] + size * dy),
                                 transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, zorder=20))
    ax.text(xy[0] + 1.35 * size * dx, xy[1] + 1.35 * size * dy, "N", transform=ax.transAxes,
            ha="center", va="center", fontsize=7, color=color, zorder=20)


def project(P, az, el):
    """Orthographic oblique projection of row-aligned points P (N,3) = (u,v,z).
    Returns screen X, screen Y and a depth key (draw large depth first)."""
    a, e = np.radians(az), np.radians(el)
    u = P[:, 0] * np.cos(a) - P[:, 1] * np.sin(a)
    v = P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a)
    X = u
    Y = P[:, 2] * np.cos(e) + v * np.sin(e)
    depth = v * np.cos(e) - P[:, 2] * np.sin(e)
    return X, Y, depth


def save(fig, name, outdir=None):
    from pathlib import Path
    outdir = Path(outdir or Path(__file__).resolve().parent.parent)
    fig.savefig(outdir / f"{name}.pdf", dpi=400)
    fig.savefig(outdir / f"{name}.png", dpi=200)
