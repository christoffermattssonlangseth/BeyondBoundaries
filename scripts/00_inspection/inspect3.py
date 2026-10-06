import sys
from pathlib import Path
import numpy as np, pandas as pd, tifffile, zarr, h5py

exec(open(Path(__file__).with_name("inspect2.py")).read().split("# 1) what each file")[0])  # imports + read_window
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

B = Path(sys.argv[1]); px = 0.2125
cells = pd.read_parquet(B / "cells.parquet")
root = zarr.open_group(zarr.ZipStore(str(B / "cells.zarr.zip"), mode="r"), mode="r")
ps0 = root["polygon_sets"]["0"]; ci = ps0["cell_index"][:]; meth0 = ps0["method"][:]
print("nucleus polygons:", len(ci), "unique cell_index:", len(np.unique(ci)), "max cell_index:", ci.max(), "N cells:", len(cells))
print("nucleus methods:", np.unique(meth0, return_counts=True))
print("nucleus_count dist:", cells.nucleus_count.value_counts().sort_index().to_dict())
print(cells.groupby("segmentation_method")[["cell_area","nucleus_area","transcript_counts"]].median().round(1).to_string())
print("cells w/o nucleus by method:\n", cells.assign(nonuc=cells.nucleus_count==0).groupby("segmentation_method").nonuc.mean().round(3).to_string())

cx, cy = cells.x_centroid.median() / px, cells.y_centroid.median() / px
y0, x0 = int(cy) - 512, int(cx) - 512
cm = root["masks"]["1"][y0:y0+1024, x0:x0+1024]; nm = root["masks"]["0"][y0:y0+1024, x0:x0+1024]
# hypothesis: nucleus label L -> polygon row L-1 -> cell_index -> cell label cell_index+1
both = nm > 0
pred_cell = ci[nm[both].astype(np.int64) - 1] + 1
print("nucleus label->polygon_sets/0/cell_index->cell label agrees with cell mask:", (pred_cell == cm[both]).mean().round(4))

files = sorted((B / "morphology_focus").glob("ch*.ome.tif"))
fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
for a, f in zip(ax, files):
    im = read_window(f, y0, x0, 1024, 1024).astype(float)
    a.imshow(np.clip(im / np.percentile(im, 99.5), 0, 1), cmap="gray")
    a.contour(find_boundaries(cm, mode="inner"), levels=[0.5], colors="c", linewidths=0.3)
    a.contour(find_boundaries(nm, mode="inner"), levels=[0.5], colors="m", linewidths=0.3)
    a.set_title(f.name, fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110)

# annotated object overlap
sid = sys.argv[3]  # e.g. run5_C2_G1_Mid_0088858
with h5py.File(sys.argv[4], "r") as f:  # annotated RRMAP2 h5ad
    idx = f["obs"][f["obs"].attrs["_index"]].asstr()[:]
mine = np.array([i.split(":", 1)[1] for i in idx if i.startswith(sid + ":")])
print(f"\nAnnData cells for {sid}: {len(mine)} / bundle {len(cells)}; all in bundle: {np.isin(mine, cells.cell_id).all()}")
kept = cells.cell_id.isin(mine)
print("filtered-out cells by method:\n", cells[~kept].segmentation_method.value_counts().to_string())
print("median transcripts kept vs dropped:", cells[kept].transcript_counts.median(), cells[~kept].transcript_counts.median())
