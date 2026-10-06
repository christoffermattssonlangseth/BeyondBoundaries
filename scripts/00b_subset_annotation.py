"""Cut the run5/run6 cells (in-scope sections) out of the annotated RRMAP2 object.

Keeps obs (annotation, niches, design), raw counts as X, var and obsm['spatial'];
index becomes the Xenium cell_id prefixed by section ("<sample_id>:<cell_id>", unchanged).

usage: python scripts/00b_subset_annotation.py <annotated.h5ad> <out.h5ad>
"""
import sys

import anndata as ad
import h5py
import numpy as np
from anndata.experimental import read_elem, sparse_dataset

src, dst = sys.argv[1:3]
with h5py.File(src, "r") as f:   # element-wise read: skips uns (old anndata can't parse it)
    obs = read_elem(f["obs"])
    var = read_elem(f["var"])
    keep = np.flatnonzero(obs["run_id"].isin(["run5", "run6"]).values)
    print(f"{len(keep)} / {len(obs)} cells in run5/run6")
    X = sparse_dataset(f["layers/counts"])[keep[0]:keep[-1] + 1][keep - keep[0]]
    spatial = f["obsm/spatial"][keep[0]:keep[-1] + 1][keep - keep[0]]
obs = obs.iloc[keep].drop(columns=[c for c in obs if c.startswith(("CellCharter_", "leiden_", "polygon_"))])
for c in obs.select_dtypes("category"):
    obs[c] = obs[c].cat.remove_unused_categories()
out = ad.AnnData(X=X, obs=obs, var=var, obsm={"spatial": spatial})
out.write_h5ad(dst, compression="gzip")
print("wrote", dst, out)
