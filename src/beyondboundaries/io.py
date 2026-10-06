"""Readers for Xenium (XOA >= 4) output bundles.

Verified on run5/run6 (XOA 4.0.1, see RESULTS.md step 0):
- morphology_focus/ch000N_*.ome.tif: one uint16 plane per file (multi-file OME, JPEG2000 tiles);
  the OME header maps FirstC -> FileName, channel names come from <Channel Name=...>.
- cells.zarr.zip: masks/1 = cell labels (label L <-> cells.parquet row L-1),
  masks/0 = nucleus labels in their own index (nucleus L -> polygon_sets/0/cell_index[L-1] -> cell L'-1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
import zarr

# OME channel name -> short feature prefix
CHANNEL_KEYS = {
    "DAPI": "dapi",
    "ATP1A1/CD45/E-Cadherin": "bnd",     # boundary stain
    "18S": "r18s",                       # interior RNA stain
    "alphaSMA/Vimentin": "smavim",       # interior protein stain
}


def decode_cell_id(prefix: int, suffix: int) -> str:
    """Xenium cell_id: 8 hex digits of the uint32 prefix shifted to 'a'..'p', plus '-suffix'."""
    return "".join(chr(ord("a") + int(c, 16)) for c in f"{prefix:08x}") + f"-{suffix}"


def section_id(bundle: Path) -> str:
    """'output-XETG00045__0088858__C2_G1_Mid__...' in run5 -> 'run5_C2_G1_Mid_0088858' (matches annotation)."""
    _, slide, region = bundle.name.split("__")[:3]
    run = re.search(r"_(run\d+)$", bundle.parent.name).group(1)
    return f"{run}_{region}_{slide}"


def find_bundles(cfg: dict) -> dict[str, Path]:
    """In-scope bundles from config: {section_id: bundle path}, excluding cfg['exclude_regions'] globs."""
    import fnmatch
    out = {}
    for run_dir in cfg["runs"].values():
        for b in sorted(Path(run_dir).glob("output-*")):
            if not any(fnmatch.fnmatch(b.name.split("__")[2], pat) for pat in cfg["exclude_regions"]):
                out[section_id(b)] = b
    return out


class TiledPlane:
    """Random-access window reads from one tiled TIFF plane without zarr (tifffile>=2025 needs zarr 3)."""

    def __init__(self, path: Path):
        self.tif = tifffile.TiffFile(path)
        self.page = self.tif.pages[0]
        self.shape = self.page.shape
        self.th, self.tw = self.page.tilelength, self.page.tilewidth
        self.ntx = -(-self.shape[1] // self.tw)

    def read(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        p, fh = self.page, self.tif.filehandle
        out = np.zeros((h, w), p.dtype)
        ys, ye = max(y0, 0), min(y0 + h, self.shape[0])
        xs, xe = max(x0, 0), min(x0 + w, self.shape[1])
        for ty in range(ys // self.th, (ye - 1) // self.th + 1):
            for tx in range(xs // self.tw, (xe - 1) // self.tw + 1):
                i = ty * self.ntx + tx
                fh.seek(p.dataoffsets[i])
                tile = p.decode(fh.read(p.databytecounts[i]), i)[0].reshape(self.th, self.tw)
                Y, X = ty * self.th, tx * self.tw
                a, b = max(Y, ys), min(Y + self.th, ye)
                c, d = max(X, xs), min(X + self.tw, xe)
                out[a - y0:b - y0, c - x0:d - x0] = tile[a - Y:b - Y, c - X:d - X]
        return out

    def close(self):
        self.tif.close()


class XeniumBundle:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.experiment = json.loads((self.path / "experiment.xenium").read_text())
        self.pixel_size = float(self.experiment["pixel_size"])
        self.section_id = section_id(self.path)
        self.channel_files = self._channel_files()
        self._planes = None
        self._seg = None

    # ---- images -------------------------------------------------------------------------
    def _channel_files(self) -> dict[str, Path]:
        focus = self.path / "morphology_focus"
        files = sorted(focus.glob("*.ome.tif"))
        if not files:
            raise FileNotFoundError(f"no morphology_focus images in {self.path}")
        with tifffile.TiffFile(files[0]) as t:
            ome = t.ome_metadata
        names = re.findall(r'<Channel[^>]*Name="([^"]*)"', ome)
        out = {}
        for attrs, fname in re.findall(r'<TiffData([^>]*)>\s*<UUID[^>]*FileName="([^"]*)"', ome):
            c = int(re.search(r'FirstC="(\d+)"', attrs).group(1))
            out[CHANNEL_KEYS[names[c]]] = focus / fname
        if set(out) != set(CHANNEL_KEYS.values()):
            raise ValueError(f"unexpected channels {names} in {focus}")
        return {k: out[k] for k in CHANNEL_KEYS.values()}  # fixed order

    @property
    def planes(self) -> dict[str, TiledPlane]:
        if self._planes is None:
            self._planes = {k: TiledPlane(p) for k, p in self.channel_files.items()}
        return self._planes

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.planes.values())).shape

    def read_level(self, channel: str, level: int) -> np.ndarray:
        """Whole plane at pyramid level `level` (2**level downsampled; SubIFDs, OME parsing off)."""
        with tifffile.TiffFile(self.channel_files[channel], is_ome=False) as t:
            return t.series[0].levels[level].asarray()

    def transcript_density(self, factor: int, min_qv: float = 20) -> np.ndarray:
        """Decoded gene transcripts (qv >= min_qv) counted per pixel of a 2**level grid (factor = 2**level)."""
        import pyarrow.parquet as pq

        H, W = self.shape
        out = np.zeros((-(-H // factor), -(-W // factor)), np.int64)
        f = pq.ParquetFile(self.path / "transcripts.parquet")
        step = self.pixel_size * factor
        for i in range(f.num_row_groups):
            t = f.read_row_group(i, columns=["x_location", "y_location", "qv", "is_gene"])
            ok = (t["qv"].to_numpy() >= min_qv) & t["is_gene"].to_numpy()
            iy = np.clip((t["y_location"].to_numpy()[ok] / step).astype(int), 0, out.shape[0] - 1)
            ix = np.clip((t["x_location"].to_numpy()[ok] / step).astype(int), 0, out.shape[1] - 1)
            np.add.at(out, (iy, ix), 1)
        return out

    def cell_mask_lowres(self, factor: int) -> np.ndarray:
        """Boolean 'any cell' mask downsampled by `factor` (max-pool), read in row strips."""
        cell_m, _, _ = self._open_seg()
        H, W = cell_m.shape
        out = np.zeros((-(-H // factor), -(-W // factor)), bool)
        step = factor * max(1, 4096 // factor)
        for y in range(0, H, step):
            strip = cell_m[y:y + step] > 0
            h = strip.shape[0]
            pad = np.zeros((-(-h // factor) * factor, out.shape[1] * factor), bool)
            pad[:h, :W] = strip
            out[y // factor:y // factor + pad.shape[0] // factor] = \
                pad.reshape(pad.shape[0] // factor, factor, -1, factor).any((1, 3))
        return out

    # ---- segmentation -------------------------------------------------------------------
    def _open_seg(self):
        if self._seg is None:
            root = zarr.open_group(zarr.ZipStore(str(self.path / "cells.zarr.zip"), mode="r"), mode="r")
            ci = root["polygon_sets"]["0"]["cell_index"][:].astype(np.int64)
            nuc_to_cell = np.concatenate([[0], ci + 1]).astype(np.uint32)
            self._seg = (root["masks"]["1"], root["masks"]["0"], nuc_to_cell)
        return self._seg

    def read_window(self, y0: int, x0: int, h: int, w: int):
        """-> img (C,h,w) float32 in CHANNEL_KEYS order, cell labels (h,w), nucleus pixels as cell labels (h,w)."""
        cell_m, nuc_m, nuc_to_cell = self._open_seg()
        H, W = self.shape
        ys, ye, xs, xe = max(y0, 0), min(y0 + h, H), max(x0, 0), min(x0 + w, W)
        cell = np.zeros((h, w), np.uint32)
        nuc = np.zeros((h, w), np.uint32)
        cell[ys - y0:ye - y0, xs - x0:xe - x0] = cell_m[ys:ye, xs:xe]
        nuc[ys - y0:ye - y0, xs - x0:xe - x0] = nuc_to_cell[nuc_m[ys:ye, xs:xe]]
        img = np.stack([p.read(y0, x0, h, w) for p in self.planes.values()]).astype(np.float32)
        return img, cell, nuc

    # ---- tables -------------------------------------------------------------------------
    def cells(self) -> pd.DataFrame:
        """cells.parquet + mask label + gene count; checks the label <-> cell_id mapping."""
        c = pd.read_parquet(self.path / "cells.parquet")
        root = zarr.open_group(zarr.ZipStore(str(self.path / "cells.zarr.zip"), mode="r"), mode="r")
        ids = root["cell_id"][:]
        dec = np.array([decode_cell_id(int(p), int(s)) for p, s in ids])
        if not (dec == c.cell_id.values).all():
            raise ValueError(f"{self.path.name}: cells.zarr cell_id order != cells.parquet")
        c["label"] = np.arange(1, len(c) + 1, dtype=np.uint32)
        c["n_genes"] = self.gene_counts().reindex(c.cell_id).values
        return c

    def gene_counts(self) -> pd.Series:
        """Number of detected genes (Gene Expression features only) per cell."""
        with h5py.File(self.path / "cell_feature_matrix.h5", "r") as f:
            m = f["matrix"]
            is_gene = m["features/feature_type"].asstr()[:] == "Gene Expression"
            indptr, indices = m["indptr"][:], m["indices"][:]
            barcodes = m["barcodes"].asstr()[:]
        col = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        n = np.bincount(col[is_gene[indices]], minlength=len(barcodes))
        return pd.Series(n, index=barcodes, name="n_genes")
