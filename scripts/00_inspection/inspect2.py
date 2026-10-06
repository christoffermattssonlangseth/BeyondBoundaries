import sys, re
from pathlib import Path
import numpy as np, pandas as pd, tifffile, zarr

B = Path(sys.argv[1])
mf = B / "morphology_focus"
files = sorted(mf.glob("ch*.ome.tif"))
def read_window(path, y0, x0, h, w):
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]; th, tw = p.tilelength, p.tilewidth
        ntx = -(-p.imagewidth // tw)
        out = np.zeros((h, w), p.dtype)
        fh = t.filehandle
        for ty in range(y0 // th, (y0 + h - 1) // th + 1):
            for tx in range(x0 // tw, (x0 + w - 1) // tw + 1):
                i = ty * ntx + tx
                fh.seek(p.dataoffsets[i]); data = fh.read(p.databytecounts[i])
                tile = p.decode(data, i)[0].reshape(th, tw)
                Y, X = ty * th, tx * tw
                ys, xs = max(Y, y0), max(X, x0); ye, xe = min(Y + th, y0 + h), min(X + tw, x0 + w)
                out[ys-y0:ye-y0, xs-x0:xe-x0] = tile[ys-Y:ye-Y, xs-X:xe-X]
        return out
# 1) what each file physically holds
for f in files:
    with tifffile.TiffFile(f) as t:
        p = t.pages[0]
        print(f.name, "pages:", len(t.pages), "shape:", p.shape, "subifds:", len(p.subifds or []),
              "desc-UUID:", re.findall(r'<UUID[^>]*FileName="([^"]*)"', t.ome_metadata or "")[:4] if f == files[0] else "")
# crop from tissue centre & compare channels
cells = pd.read_parquet(B / "cells.parquet")
px = 0.2125
cx, cy = cells.x_centroid.median() / px, cells.y_centroid.median() / px
y0, x0 = int(cy) - 512, int(cx) - 512
crops = {}
for f in files:
    crops[f.name] = read_window(f, y0, x0, 1024, 1024)
    a = crops[f.name]; print(f"  {f.name}: mean={a.mean():.1f} p50={np.percentile(a,50):.0f} p99={np.percentile(a,99):.0f} max={a.max()}")
names = list(crops)
C = np.corrcoef([crops[n].ravel().astype(float) for n in names])
print("pixel corr between files:\n", pd.DataFrame(C, index=names, columns=[n[:10] for n in names]).round(2))

# 2) masks <-> cell_id
root = zarr.open_group(zarr.ZipStore(str(B / "cells.zarr.zip"), mode="r"), mode="r")
ids = root["cell_id"][:]
def decode(prefix, suffix):
    h = f"{prefix:08x}"
    return "".join(chr(ord("a") + int(c, 16)) for c in h) + f"-{suffix}"
dec = np.array([decode(p, s) for p, s in ids])
print("\ndecoded cell_id == cells.parquet order:", (dec == cells.cell_id.values).all(), dec[:3])
print("homogeneous_transform:\n", root["masks"]["homogeneous_transform"][:])
cm = root["masks"]["1"]; nm = root["masks"]["0"]
cell_crop = cm[y0:y0+1024, x0:x0+1024]; nuc_crop = nm[y0:y0+1024, x0:x0+1024]
# label at centroid pixel -> should equal index+1
sub = cells[(cells.x_centroid/px > x0+50) & (cells.x_centroid/px < x0+974) & (cells.y_centroid/px > y0+50) & (cells.y_centroid/px < y0+974)]
lab = cell_crop[(sub.y_centroid/px - y0).astype(int), (sub.x_centroid/px - x0).astype(int)]
print(f"cells in crop: {len(sub)}; centroid label == row_index+1: {(lab == sub.index.values + 1).mean():.3f}; label==0: {(lab==0).mean():.3f}")
# area check
labs, cnt = np.unique(cell_crop[cell_crop > 0], return_counts=True)
inner = sub.index.values + 1
m = np.isin(labs, inner)
a_mask = pd.Series(cnt[m] * px * px, index=labs[m] - 1)
print("area mask vs cells.parquet (median ratio):", np.median(a_mask / cells.loc[a_mask.index, "cell_area"]).round(3))
# nucleus labels: same id space as cells?
nl = np.unique(nuc_crop[nuc_crop > 0])
print("nucleus labels in crop:", len(nl), "subset of cell labels:", np.isin(nl, labs).mean().round(3))
both = (nuc_crop > 0)
print("nucleus px whose label == cell label at same px:", (nuc_crop[both] == cell_crop[both]).mean().round(3))
print("nucleus px outside any cell:", (cell_crop[both] == 0).mean().round(3))
print("cell_summary cols? attrs:", dict(root["cell_summary"].attrs))
