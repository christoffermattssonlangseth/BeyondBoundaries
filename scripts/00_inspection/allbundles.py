import json, re, sys
from pathlib import Path
import pandas as pd, numpy as np, tifffile
obs = pd.read_parquet(sys.argv[1])
rows = []
for run in ["/Volumes/moldiassd/20260506__111857__GoncaloTing_5kMouse_run5", "/Volumes/moldiassd/20260506__112634__GoncaloTing_5kMouse_run6"]:
    for B in sorted(Path(run).glob("output-*")):
        e = json.load(open(B / "experiment.xenium"))
        _, slide, region = B.name.split("__")[:3]
        sid = f"{'run5' if 'run5' in run else 'run6'}_{region}_{slide}"
        files = sorted(p.name for p in (B / "morphology_focus").glob("ch*.ome.tif"))
        with tifffile.TiffFile(B / "morphology_focus" / files[0]) as t:
            names = re.findall(r'<Channel[^>]*Name="([^"]*)"', t.ome_metadata); shp = t.pages[0].shape
        c = pd.read_parquet(B / "cells.parquet", columns=["cell_id", "segmentation_method", "nucleus_count"])
        sm = c.segmentation_method.str.extract(r"by (\w+)")[0].value_counts(normalize=True)
        o = obs[obs.sample_id.astype(str) == sid]
        rows.append(dict(sample_id=sid, xoa=e["analysis_sw_version"], fmt=f'{e["major_version"]}.{e["minor_version"]}', px=e["pixel_size"],
            prep=e["preservation_method"], n_files=len(files), chan_ok=names == ['DAPI', 'ATP1A1/CD45/E-Cadherin', '18S', 'alphaSMA/Vimentin'],
            shape=shp, n_cells=len(c), in_adata=len(o), adata_in_bundle=o.cell_id.isin(c.cell_id).mean() if len(o) else np.nan,
            interior=round(sm.get("interior", 0), 3), boundary=round(sm.get("boundary", 0), 3), nucexp=round(sm.get("nucleus", 0), 3),
            no_nuc=round((c.nucleus_count == 0).mean(), 3)))
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df.to_string(index=False))
df.to_csv(sys.argv[2], index=False)
