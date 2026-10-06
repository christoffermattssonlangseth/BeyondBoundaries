"""Phase 2: tissue mask, cell-free background / autofluorescence, edge distance, normalisation.

Works on a low-resolution pyramid level (default 3 = 1.7 um/px), which is enough for slowly
varying background and for tissue geometry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology

from .io import XeniumBundle

INTENSITY_STATS = ("mean", "p50", "p90", "p99")


def section_background(b: XeniumBundle, level: int = 3, cell_margin_um: float = 5.0,
                       smooth_um: float = 25.0, tissue_smooth_um: float = 10.0,
                       min_tissue_um2: float = 2e4) -> dict:
    """Low-res maps for one section.

    tissue   : smoothed DAPI+18S > Otsu (log), holes filled, small specks removed
    cellfree : tissue pixels further than cell_margin_um from any segmented cell
    bg_<ch>  : local background = Gaussian-weighted mean of cell-free pixels (normalised convolution)
    edge_um  : distance to tissue edge
    """
    f = 2 ** level
    px = b.pixel_size * f
    imgs = {ch: b.read_level(ch, level).astype(np.float32) for ch in b.channel_files}
    shape = imgs["dapi"].shape
    sig = np.log1p(ndi.gaussian_filter(imgs["dapi"] + imgs["r18s"], tissue_smooth_um / px))
    tissue = sig > filters.threshold_otsu(sig)
    tissue = ndi.binary_fill_holes(tissue)
    tissue = morphology.remove_small_objects(tissue, int(min_tissue_um2 / px ** 2))
    cells = b.cell_mask_lowres(f)[:shape[0], :shape[1]]
    near_cell = ndi.binary_dilation(cells, iterations=max(1, int(round(cell_margin_um / px))))
    cellfree = tissue & ~near_cell
    w = ndi.gaussian_filter(cellfree.astype(np.float32), smooth_um / px)
    out = {"px_um": px, "tissue": tissue, "cellfree": cellfree, "cells": cells,
           "edge_um": ndi.distance_transform_edt(tissue) * px, "img": imgs}
    for ch, im in imgs.items():
        num = ndi.gaussian_filter(np.where(cellfree, im, 0), smooth_um / px)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"bg_{ch}"] = np.where(w > 1e-3, num / w, np.nan)
        out[f"bg_global_{ch}"] = float(np.median(im[cellfree])) if cellfree.any() else np.nan
    return out


def sample_at_cells(maps: dict, x_um: np.ndarray, y_um: np.ndarray, channels) -> pd.DataFrame:
    """Per-cell covariates from the low-res maps (nearest pixel); NaN local bg -> section median."""
    px = maps["px_um"]
    H, W = maps["tissue"].shape
    iy = np.clip((y_um / px).astype(int), 0, H - 1)
    ix = np.clip((x_um / px).astype(int), 0, W - 1)
    d = {"edge_um": maps["edge_um"][iy, ix], "in_tissue": maps["tissue"][iy, ix]}
    for ch in channels:
        v = maps[f"bg_{ch}"][iy, ix]
        d[f"{ch}_bg_local"] = np.where(np.isfinite(v), v, maps[f"bg_global_{ch}"])
        d[f"{ch}_bg_global"] = np.full(len(iy), maps[f"bg_global_{ch}"])
    return pd.DataFrame(d)


def cellfree_by_region(maps: dict, cell_xy_um: np.ndarray, cell_region: np.ndarray, channels,
                       max_dist_um: float = 50.0) -> pd.DataFrame:
    """Cell-free pixel intensities labelled by the region of the nearest annotated cell
    (e.g. Global_anatomical_region WM vs GM) -> long table for autofluorescence comparison."""
    from scipy.spatial import cKDTree

    px = maps["px_um"]
    yy, xx = np.nonzero(maps["cellfree"])
    dist, j = cKDTree(cell_xy_um).query(np.c_[xx * px, yy * px], distance_upper_bound=max_dist_um)
    ok = np.isfinite(dist)
    df = pd.DataFrame({"region": np.asarray(cell_region)[j[ok]]})
    for ch in channels:
        df[ch] = maps["img"][ch][yy[ok], xx[ok]]
    return df


def normalise(feats: pd.DataFrame, channels, ref_mask: np.ndarray, bg: str = "local") -> pd.DataFrame:
    """Background-subtract compartment intensities and scale per section.

    x_norm = (x - bg) / s_section, s_section = median(cell_mean - bg) over reference cells
    (ref_mask, e.g. physiological-niche cells) of that section. Sums: (sum - bg * npx) / s.
    Ratio-type features (radial, polarity, texture, morphology) are left unchanged.
    """
    out = feats.copy()
    for ch in channels:
        b = feats[f"{ch}_bg_{bg}"]
        for sec, idx in feats.groupby("section_id").groups.items():
            m = feats.index.isin(idx)
            ref = m & ref_mask
            s = np.nanmedian((feats.loc[ref, f"{ch}_cell_mean"] - b[ref]).values) if ref.any() else np.nan
            for comp in ("cell", "nuc", "cyto", "rim", "ring", "terr"):
                for st in INTENSITY_STATS:
                    c = f"{ch}_{comp}_{st}"
                    out.loc[m, c] = (feats.loc[m, c] - b[m]) / s
                c = f"{ch}_{comp}_sum"
                if c in feats:
                    out.loc[m, c] = (feats.loc[m, c] - b[m] * feats.loc[m, f"{comp}_npx"]) / s
            out.loc[m, f"{ch}_scale"] = s
    return out
