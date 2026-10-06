"""Per-cell image features computed on one image window, all channels at once.

Compartments (per cell label L):
  cell   all pixels of L
  nuc    nucleus pixels of L (union if several nuclei)
  cyto   cell minus nucleus
  rim    cell pixels within rim_um of the cell edge (inner membrane band)
  ring   non-cell pixels within ring_um outside the cell, assigned to the nearest cell
  terr   non-cell pixels within territory_um, assigned to the nearest cell (local extracellular
         context: neuropil, vessel wall, debris)
Every grouped operation is a bincount / lexsort over pixels, no per-cell Python loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels, find_boundaries

COMPARTMENTS = ("cell", "nuc", "cyto", "rim", "ring", "terr")
REPORT_Q = (50, 90, 99)
_Q = (1, 10, 50, 90, 99)  # 1/10/99 also used internally (texture quantisation, polarity floor)


@dataclass
class FeatureParams:
    pixel_size: float = 0.2125
    rim_um: float = 1.0
    ring_um: float = 2.0
    territory_um: float = 10.0
    radial_bins: int = 5
    glcm_levels: int = 16
    glcm_distance_um: float = 0.5
    # texture is computed where the stain lives: chromatin for DAPI, whole cell otherwise
    glcm_compartment: dict = field(default_factory=lambda: {"dapi": "nuc"})

    def px(self, um: float) -> int:
        return max(1, int(round(um / self.pixel_size)))


def grouped_stats(g: np.ndarray, v: np.ndarray, n: int, qs=_Q) -> dict[str, np.ndarray]:
    """g: (P,) group index 1..n, v: (P,C). -> count (n,), sum/mean/q<k> (n,C); NaN for empty groups."""
    C = v.shape[1]
    cnt = np.bincount(g, minlength=n + 1)[1:]
    out = {"count": cnt}
    out["sum"] = np.stack([np.bincount(g, v[:, c], minlength=n + 1)[1:] for c in range(C)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mean"] = out["sum"] / cnt[:, None]
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    empty = cnt == 0
    for q in qs:
        out[f"q{q}"] = np.full((n, C), np.nan)
    if len(g) == 0:
        return out
    for c in range(C):
        sv = v[np.lexsort((v[:, c], g)), c]
        for q in qs:
            pos = starts + q / 100 * np.maximum(cnt - 1, 0)
            lo = np.minimum(np.floor(pos).astype(np.int64), len(sv) - 1)
            hi = np.minimum(lo + 1, starts + cnt - 1).clip(0, len(sv) - 1)
            val = sv[lo] + (sv[hi] - sv[lo]) * (pos - np.floor(pos))
            val[empty] = np.nan
            out[f"q{q}"][:, c] = val
    return out


def _shift_pairs(a: np.ndarray, dy: int, dx: int):
    H, W = a.shape
    return (a[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)],
            a[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)])


def glcm_features(q: np.ndarray, g: np.ndarray, n: int, L: int, d: int) -> dict[str, np.ndarray]:
    """Symmetric grey-level co-occurrence over 4 directions at distance d, pooled per group.
    q: (H,W) levels 0..L-1; g: (H,W) group 1..n (0 = ignore). Pairs only within the same group."""
    acc = np.zeros((n + 1) * L * L)
    for dy, dx in ((0, d), (d, 0), (d, d), (d, -d)):
        ga, gb = _shift_pairs(g, dy, dx)
        qa, qb = _shift_pairs(q, dy, dx)
        ok = (ga == gb) & (ga > 0)
        ga, qa, qb = ga[ok].astype(np.int64), qa[ok].astype(np.int64), qb[ok].astype(np.int64)
        acc += np.bincount(ga * L * L + qa * L + qb, minlength=acc.size)
        acc += np.bincount(ga * L * L + qb * L + qa, minlength=acc.size)
    P = acc.reshape(n + 1, L, L)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        P = P / P.sum((1, 2), keepdims=True)
        i = np.arange(L)[:, None]
        j = np.arange(L)[None, :]
        mu = (P * i).sum((1, 2))
        var = (P * (i - mu[:, None, None]) ** 2).sum((1, 2))
        logP = np.where(P > 0, np.log2(np.where(P > 0, P, 1)), 0)
        return {
            "contrast": (P * (i - j) ** 2).sum((1, 2)),
            "homogeneity": (P / (1 + (i - j) ** 2)).sum((1, 2)),
            "asm": (P ** 2).sum((1, 2)),
            "entropy": -(P * logP).sum((1, 2)),
            "correlation": (P * (i - mu[:, None, None]) * (j - mu[:, None, None])).sum((1, 2)) / var,
        }


def window_features(img: np.ndarray, cell: np.ndarray, nuc: np.ndarray, labels, channels,
                    p: FeatureParams, interior_edges=(False, False, False, False),
                    valid: np.ndarray | None = None) -> pd.DataFrame:
    """Features for the cells `labels` (cell-mask label ids) inside one window.

    img (C,H,W) float; cell (H,W) cell labels; nuc (H,W) nucleus pixels carrying their *cell* label.
    interior_edges: (top, bottom, left, right) True where the window edge is not the image edge;
    cells touching such an edge are flagged `truncated` (the tiling halo should make this rare).
    valid: (H,W) bool, False for zero-padding outside the image (excluded from ring/territory).
    """
    labels = np.asarray(labels, np.int64)
    n, C = len(labels), img.shape[0]
    lut = np.zeros(max(int(cell.max()), int(labels.max(initial=0))) + 1, np.int64)
    lut[labels] = np.arange(1, n + 1)
    loc = lut[cell]                                   # local index 1..n, 0 = background/other cells
    nuc = np.where(nuc == cell, nuc, 0)
    in_cell, in_nuc = cell > 0, nuc > 0

    # --- geometry maps -------------------------------------------------------------------
    # pad so the array border counts as a cell edge (identical whether a cell is cut by the image
    # edge or by a zero-padded window)
    edge = find_boundaries(np.pad(cell, 1), mode="inner")[1:-1, 1:-1]
    d_edge = ndi.distance_transform_edt(~edge)
    ring_lab = expand_labels(cell, p.px(p.ring_um))
    terr_lab = expand_labels(cell, p.px(p.territory_um))
    masks = {
        "cell": loc > 0,
        "nuc": (loc > 0) & in_nuc,
        "cyto": (loc > 0) & ~in_nuc,
        "rim": (loc > 0) & (d_edge <= p.px(p.rim_um) - 1),
    }
    gmaps = {k: loc for k in masks}
    outside = in_cell if valid is None else in_cell | ~valid
    gmaps["ring"] = lut[np.where(outside, 0, ring_lab)]
    gmaps["terr"] = lut[np.where(outside, 0, terr_lab)]
    masks["ring"] = gmaps["ring"] > 0
    masks["terr"] = gmaps["terr"] > 0

    cols: dict[str, np.ndarray] = {}
    stats = {}
    for comp in COMPARTMENTS:
        m = masks[comp]
        g = gmaps[comp][m]
        s = grouped_stats(g, img[:, m].T, n)
        stats[comp] = s
        cols[f"{comp}_npx"] = s["count"]
        for ci, ch in enumerate(channels):
            cols[f"{ch}_{comp}_mean"] = s["mean"][:, ci]
            for q in REPORT_Q:
                cols[f"{ch}_{comp}_p{q}"] = s[f"q{q}"][:, ci]
            if comp in ("cell", "nuc", "cyto", "rim"):
                cols[f"{ch}_{comp}_sum"] = s["sum"][:, ci]

    # --- radial profile: bin 0 = cell centre ... bin B-1 = cell edge, normalised to cell mean
    m = masks["cell"]
    g = loc[m]
    d = d_edge[m]
    dmax = np.zeros(n + 1)
    np.maximum.at(dmax, g, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 1 - d / dmax[g]
    B = p.radial_bins
    b = np.minimum((np.nan_to_num(r) * B).astype(np.int64), B - 1)
    gb = g * B + b
    cnt_b = np.bincount(gb, minlength=(n + 1) * B).reshape(n + 1, B)[1:]
    for ci, ch in enumerate(channels):
        sum_b = np.bincount(gb, img[ci][m], minlength=(n + 1) * B).reshape(n + 1, B)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = sum_b / cnt_b / stats["cell"]["mean"][:, ci:ci + 1]
        for k in range(B):
            cols[f"{ch}_radial_b{k}"] = prof[:, k]

    # --- polarity: intensity-weighted centroid vs nucleus / geometric centroid (/ eq. radius)
    yy, xx = np.nonzero(m)
    cnt = stats["cell"]["count"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cy = np.bincount(g, yy, minlength=n + 1)[1:] / cnt
        cx = np.bincount(g, xx, minlength=n + 1)[1:] / cnt
        mn = masks["nuc"]
        gn = loc[mn]
        ny_, nx_ = np.nonzero(mn)
        ncnt = np.bincount(gn, minlength=n + 1)[1:].astype(float)
        ny = np.bincount(gn, ny_, minlength=n + 1)[1:] / ncnt
        nx = np.bincount(gn, nx_, minlength=n + 1)[1:] / ncnt
        R = np.sqrt(cnt / np.pi)
        for ci, ch in enumerate(channels):
            w = np.maximum(img[ci][m] - stats["cell"]["q10"][g - 1, ci], 0)
            sw = np.bincount(g, w, minlength=n + 1)[1:]
            wy = np.bincount(g, w * yy, minlength=n + 1)[1:] / sw
            wx = np.bincount(g, w * xx, minlength=n + 1)[1:] / sw
            cols[f"{ch}_polarity_shape"] = np.hypot(wy - cy, wx - cx) / R
            cols[f"{ch}_polarity_nuc"] = np.hypot(wy - ny, wx - nx) / R
            cols[f"{ch}_polarity_dx_um"] = (wx - cx) * p.pixel_size
            cols[f"{ch}_polarity_dy_um"] = (wy - cy) * p.pixel_size
        cols["nuc_offset"] = np.hypot(ny - cy, nx - cx) / R

    # --- texture (Haralick on per-cell 1-99% rescaled levels) -------------------------------
    L, dist = p.glcm_levels, p.px(p.glcm_distance_um)
    for ci, ch in enumerate(channels):
        comp = p.glcm_compartment.get(ch, "cell")
        mc = masks[comp]
        gq = np.where(mc, loc, 0)
        lo = stats[comp]["q1"][:, ci]
        hi = stats[comp]["q99"][:, ci]
        span = np.where(hi > lo, hi - lo, np.inf)
        q = np.zeros(cell.shape, np.int64)
        gi = gq[mc] - 1
        q[mc] = np.clip(((img[ci][mc] - lo[gi]) / span[gi] * L).astype(np.int64), 0, L - 1)
        for k, v in glcm_features(q, gq, n, L, dist).items():
            cols[f"{ch}_glcm_{k}"] = v

    # --- morphology ---------------------------------------------------------------------
    df = pd.DataFrame(cols, index=pd.Index(labels, name="label"))
    px2 = p.pixel_size ** 2
    props = ["label", "area", "perimeter", "eccentricity", "solidity",
             "axis_major_length", "axis_minor_length", "orientation"]
    cp = pd.DataFrame(regionprops_table(np.where(loc > 0, cell, 0), properties=props)).set_index("label")
    npr = pd.DataFrame(regionprops_table(np.where(masks["nuc"], cell, 0),
                                         properties=props[:5])).set_index("label")
    morph = pd.DataFrame(index=df.index)
    morph["morph_cell_area_um2"] = cp["area"] * px2
    morph["morph_cell_perimeter_um"] = cp["perimeter"] * p.pixel_size
    morph["morph_cell_eccentricity"] = cp["eccentricity"]
    morph["morph_cell_solidity"] = cp["solidity"]
    morph["morph_cell_major_um"] = cp["axis_major_length"] * p.pixel_size
    morph["morph_cell_minor_um"] = cp["axis_minor_length"] * p.pixel_size
    morph["morph_cell_orientation"] = cp["orientation"]
    morph["morph_cell_circularity"] = 4 * np.pi * cp["area"] / cp["perimeter"].clip(lower=1) ** 2
    morph["morph_nuc_area_um2"] = npr["area"] * px2
    morph["morph_nuc_perimeter_um"] = npr["perimeter"] * p.pixel_size
    morph["morph_nuc_eccentricity"] = npr["eccentricity"]
    morph["morph_nuc_solidity"] = npr["solidity"]
    morph["morph_nuc_cell_ratio"] = morph["morph_nuc_area_um2"] / morph["morph_cell_area_um2"]
    df = pd.concat([morph, df], axis=1)

    # --- truncation flag ----------------------------------------------------------------
    edges = [cell[0], cell[-1], cell[:, 0], cell[:, -1]]
    cut = np.unique(np.concatenate([e for e, inner in zip(edges, interior_edges) if inner] or [[]]))
    df["truncated"] = df.index.isin(cut.astype(np.int64))
    df["centroid_y_px"] = cy
    df["centroid_x_px"] = cx
    return df
