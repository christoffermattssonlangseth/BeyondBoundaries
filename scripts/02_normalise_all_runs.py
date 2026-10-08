"""Background maps + normalisation for all five runs -> data/features_norm_all.parquet

Same method as notebooks/02_normalisation_qc (local background from cell-free tissue; per-section scale = p90 of raw
cell means of physiological-niche cells). Runs 1-3 were copied without transcripts.parquet: the autofluorescence
proxy (QC only) is skipped for them; normalisation does not use it. data/features_norm.parquet (runs 5/6, used by
notebooks 02-10) is left untouched; this script checks that it reproduces it for runs 5/6.

usage: python scripts/02_normalise_all_runs.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beyondboundaries import background as bgm  # noqa: E402
from beyondboundaries import data  # noqa: E402
from beyondboundaries.io import XeniumBundle, find_bundles  # noqa: E402

H5AD_ALL = "data/RRMAP2_all_runs.h5ad"
CACHE = ROOT / "data" / "qc"
CH = data.CHANNELS


def main():
    t0 = time.time()
    cfg = data.load_config()
    for sec, path in find_bundles(cfg).items():
        out = CACHE / f"{sec}_covariates.parquet"
        if out.exists():
            continue
        m = bgm.section_background(XeniumBundle(path))
        xy = pd.read_parquet(ROOT / cfg["features_dir"] / f"{sec}.parquet", columns=["x_centroid", "y_centroid"])
        cov = bgm.sample_at_cells(m, xy.x_centroid.values, xy.y_centroid.values, CH)
        cov.index = sec + ":" + xy.index
        cov.to_parquet(out)
        print(f"{sec}: background maps done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    cov = pd.concat(pd.read_parquet(p) for p in sorted(CACHE.glob("*_covariates.parquet")))
    f = data.load_features(cfg, h5ad=H5AD_ALL).join(cov, how="left")
    ref = (f.Curated_niche_state == "Physiological").values
    fn = bgm.normalise(f, CH, ref, bg="local", group="section_id")
    fn["run"] = fn.section_id.str.split("_").str[0]
    fn.drop(columns=["in_tissue"]).to_parquet(ROOT / "data" / "features_norm_all.parquet")
    print(f"wrote {fn.shape} | cells per run: {fn.run.value_counts().sort_index().to_dict()}", flush=True)

    # reproduce check: runs 5/6 must equal notebook 02's table
    old = pd.read_parquet(ROOT / "data" / "features_norm.parquet", columns=[f"{c}_cell_mean" for c in CH])
    new = fn.loc[old.index.intersection(fn.index), old.columns]
    diff = (new - old.loc[new.index]).abs().max()
    print(f"runs 5/6 vs features_norm.parquet: {len(new)}/{len(old)} cells, max |diff| per channel {diff.round(6).to_dict()}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
