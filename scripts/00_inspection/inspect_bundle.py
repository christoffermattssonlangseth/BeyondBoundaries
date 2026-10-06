import sys, json, re, zipfile
from pathlib import Path
import numpy as np, pandas as pd, tifffile, zarr

B = Path(sys.argv[1])
print("BUNDLE", B.name)
mf = B / "morphology_focus"
for f in sorted(mf.glob("*.ome.tif")):
    with tifffile.TiffFile(f) as t:
        s = t.series[0]
        ome = t.ome_metadata or ""
        names = re.findall(r'<Channel[^>]*Name="([^"]*)"', ome)
        px = re.findall(r'PhysicalSizeX="([^"]*)"', ome)[:1]
        print(f" {f.name}: shape={s.shape} axes={s.axes} dtype={s.dtype} levels={len(s.levels)} "
              f"chan_names={names} px={px} tile={t.pages[0].tilewidth}x{t.pages[0].tilelength} comp={t.pages[0].compression.name}")
with tifffile.TiffFile(B / "morphology.ome.tif") as t:
    s = t.series[0]; ome = t.ome_metadata
    print(" morphology.ome.tif:", s.shape, s.axes, s.dtype, "chan:", re.findall(r'<Channel[^>]*Name="([^"]*)"', ome))

cells = pd.read_parquet(B / "cells.parquet")
print("\ncells.parquet cols:", list(cells.columns))
print(cells.head(3).to_string())
if "segmentation_method" in cells:
    print(cells.segmentation_method.value_counts().to_string())

print("\ncells.zarr.zip tree:")
store = zarr.ZipStore(str(B / "cells.zarr.zip"), mode="r")
root = zarr.open_group(store, mode="r")
print(root.tree())
print("root attrs:", dict(root.attrs))
for k in root.group_keys():
    print(k, dict(root[k].attrs))
masks = root["masks"]
for k in masks.array_keys():
    a = masks[k]; print(" mask", k, a.shape, a.dtype, a.chunks)
for k in root.array_keys():
    a = root[k]; print(" arr", k, a.shape, a.dtype)
