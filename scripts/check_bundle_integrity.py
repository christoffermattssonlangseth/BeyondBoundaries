"""Integrity check of copied Xenium bundles (local disk, no source needed).

Per bundle: CRC of every cells.zarr.zip entry, full read of cells.parquet and cell_feature_matrix.h5, and decode of
every full-resolution tile of the four morphology_focus images. Catches files damaged in transit (one cells.zarr.zip
from the run1-3 copy had the right size but corrupt content). Prints one line per bundle; 'BAD' lines list problems.

usage: python scripts/check_bundle_integrity.py [--runs run1 run2 run3] [--workers 6]
"""
import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import pandas as pd
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beyondboundaries.io import XeniumBundle, find_bundles  # noqa: E402


def check(path: Path) -> tuple[str, list[str]]:
    bad = []
    try:
        z = zipfile.ZipFile(path / "cells.zarr.zip")
        if (e := z.testzip()) is not None:
            bad.append(f"cells.zarr.zip CRC error in {e}")
    except Exception as e:  # noqa: BLE001
        bad.append(f"cells.zarr.zip: {e!r}")
    for f, fn in [("cells.parquet", lambda p: pd.read_parquet(p)),
                  ("cell_feature_matrix.h5", lambda p: [h5py.File(p)["matrix"][k][:] for k in ("data", "indices", "indptr")])]:
        try:
            fn(path / f)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{f}: {e!r}")
    try:
        for ch, p in XeniumBundle(path).channel_files.items():
            with tifffile.TiffFile(p) as t:
                page, fh = t.pages[0], t.filehandle
                for i, (off, n) in enumerate(zip(page.dataoffsets, page.databytecounts)):
                    try:
                        fh.seek(off)
                        page.decode(fh.read(n), i)
                    except Exception as e:  # noqa: BLE001
                        bad.append(f"{p.name} tile {i}: {e!r}")
                        break
    except Exception as e:  # noqa: BLE001
        bad.append(f"images: {e!r}")
    return path.name, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["run1", "run2", "run3"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["runs"] = {r: p for r, p in cfg["runs"].items() if r in a.runs}
    bundles = find_bundles(cfg)
    n_bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed({ex.submit(check, p): s for s, p in bundles.items()}):
            name, bad = fut.result()
            n_bad += bool(bad)
            print(("BAD " if bad else "ok  ") + name + ("" if not bad else ": " + " | ".join(bad)), flush=True)
    print(f"{len(bundles)} bundles checked, {n_bad} with problems")


if __name__ == "__main__":
    main()
