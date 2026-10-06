"""Shared figure style + saving (every figure -> PDF + PNG, stamped with its generating notebook)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# categorical slots, fixed order (validated palette; see dataviz reference)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CHANNEL_COLORS = dict(zip(["dapi", "bnd", "r18s", "smavim"], CATEGORICAL))
CHANNEL_LABELS = {"dapi": "DAPI", "bnd": "ATP1A1/CD45/E-Cad", "r18s": "18S rRNA", "smavim": "αSMA/Vimentin"}
SEG_SHORT = {
    "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)": "boundary",
    "Segmented by interior stain (18S)": "interior (18S)",
    "Segmented by nucleus expansion of 5.0µm": "nucleus exp.",
}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#104281", "#5598e7", "#f0efec", "#ec835a", "#b8302f"])
# microscopy pseudo-colour for image composites (not a chart encoding)
IMG_RGB = {"dapi": (0.2, 0.4, 1.0), "bnd": (1.0, 0.45, 0.1), "r18s": (0.1, 0.9, 0.5), "smavim": (1.0, 0.1, 0.8)}


def style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.edgecolor": "#888888", "xtick.color": "#555555", "ytick.color": "#555555",
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def save_fig(fig, name: str, outdir: str | Path, source: str):
    """Save as PDF + PNG with a small footer naming the generating notebook/script."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.002, f"source: {source}", ha="right", va="bottom", fontsize=6, color="#999999")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    return outdir / f"{name}.png"


def composite(img: np.ndarray, channels, lo=None, hi=None) -> np.ndarray:
    """(C,H,W) -> RGB additive composite, per-channel lo/hi clipping."""
    rgb = np.zeros((*img.shape[1:], 3))
    for i, ch in enumerate(channels):
        a, b = (lo[i] if lo is not None else np.percentile(img[i], 1),
                hi[i] if hi is not None else np.percentile(img[i], 99.5))
        x = np.clip((img[i] - a) / max(b - a, 1e-6), 0, 1)
        rgb += x[..., None] * np.array(IMG_RGB[ch])
    return np.clip(rgb, 0, 1)
