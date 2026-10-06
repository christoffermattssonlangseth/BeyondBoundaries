"""Tile-wise, parallel per-cell feature extraction for one Xenium section.

A cell is computed in the window whose core contains its centroid; each window is read with a halo
(default 256 px = 54 um, > largest cell) so the cell, its ring and its territory are complete.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureParams, window_features
from .io import XeniumBundle

CELL_COLS = ["cell_id", "x_centroid", "y_centroid", "segmentation_method", "transcript_counts",
             "n_genes", "cell_area", "nucleus_area", "nucleus_count"]


def plan_windows(cells: pd.DataFrame, pixel_size: float, tile: int) -> dict[tuple[int, int], np.ndarray]:
    ty = (cells.y_centroid.values / pixel_size // tile).astype(int)
    tx = (cells.x_centroid.values / pixel_size // tile).astype(int)
    out: dict[tuple[int, int], list] = {}
    for key, lab in zip(zip(ty, tx), cells.label.values):
        out.setdefault(key, []).append(lab)
    return {k: np.asarray(v, np.int64) for k, v in sorted(out.items())}


def run_window(read, shape, ty, tx, labels, tile, halo, channels, params: FeatureParams) -> pd.DataFrame:
    H, W = shape
    y0, x0 = ty * tile - halo, tx * tile - halo
    h = w = tile + 2 * halo
    img, cell, nuc = read(y0, x0, h, w)
    interior = (y0 > 0, y0 + h < H, x0 > 0, x0 + w < W)
    valid = np.zeros((h, w), bool)
    valid[max(0, -y0):H - y0, max(0, -x0):W - x0] = True
    df = window_features(img, cell, nuc, labels, channels, params, interior, valid)
    df["centroid_y_px"] += y0
    df["centroid_x_px"] += x0
    return df


_BUNDLE: XeniumBundle | None = None


def _init_worker(path: str):
    global _BUNDLE
    _BUNDLE = XeniumBundle(path)


def _worker(task):
    ty, tx, labels, tile, halo, params = task
    t0 = time.time()
    df = run_window(_BUNDLE.read_window, _BUNDLE.shape, ty, tx, labels, tile, halo,
                    list(_BUNDLE.channel_files), params)
    return df, time.time() - t0


def extract_section(bundle_path: str | Path, out_path: str | Path, params: FeatureParams,
                    tile: int = 2048, halo: int = 256, workers: int = 4, max_windows: int | None = None,
                    log=print) -> pd.DataFrame:
    b = XeniumBundle(bundle_path)
    params.pixel_size = b.pixel_size
    cells = b.cells()
    windows = plan_windows(cells, b.pixel_size, tile)
    tasks = [(ty, tx, lab, tile, halo, params) for (ty, tx), lab in windows.items()]
    if max_windows:
        tasks = tasks[:max_windows]
    log(f"{b.section_id}: {len(cells)} cells, {len(tasks)} windows, {workers} workers")
    t0, parts, wtimes = time.time(), [], []
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                             initializer=_init_worker, initargs=(str(b.path),)) as ex:
        for i, (df, dt) in enumerate(ex.map(_worker, tasks)):
            parts.append(df)
            wtimes.append(dt)
            if (i + 1) % max(1, len(tasks) // 10) == 0:
                log(f"  {i + 1}/{len(tasks)} windows, {time.time() - t0:.0f}s elapsed")
    feats = pd.concat(parts)
    if feats.index.duplicated().any():
        raise RuntimeError("cell computed in more than one window")
    meta = cells.set_index("label")[CELL_COLS]
    out = meta.join(feats, how="inner").reset_index()
    out.insert(0, "section_id", b.section_id)
    out = out.set_index("cell_id")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    log(f"{b.section_id}: wrote {len(out)} cells x {out.shape[1]} cols in {time.time() - t0:.0f}s "
        f"(window median {np.median(wtimes):.1f}s, truncated {out.truncated.mean():.4f})")
    return out
