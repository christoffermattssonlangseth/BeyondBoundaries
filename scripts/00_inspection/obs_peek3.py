import sys, h5py, numpy as np, pandas as pd
p = sys.argv[1]
def col(obs, k):
    g = obs[k]
    if isinstance(g, h5py.Group):
        cats = g["categories"].asstr()[:]; codes = g["codes"][:]
        return pd.Categorical.from_codes(codes, cats)
    v = g[:]; return v.astype(str) if v.dtype.kind in "OS" else v
with h5py.File(p, "r") as f:
    obs = f["obs"]; ik = obs.attrs["_index"]
    idx = obs[ik].asstr()[:]; print("n_obs", len(idx), "index key", ik, idx[:2])
    skip = ("CellCharter_", "leiden_", "source_", "output_", "run_dir", "xenium_output", "polygon", "codeword", "control_")
    for k in obs.keys():
        if k == ik or k.startswith(skip): continue
        g = obs[k]
        if isinstance(g, h5py.Group) and "categories" in g:
            pass
        else: pass
    print("obsm", list(f["obsm"].keys()), "layers", list(f["layers"].keys()) if "layers" in f else None, "X", f["X"].attrs.get("encoding-type") if isinstance(f["X"], h5py.Group) else f["X"].shape)
    print("var n", f["var"][f["var"].attrs["_index"]].shape)
    df = pd.DataFrame({k: col(obs, k) for k in ["run_id", "sample_id", "condition", "model", "stage", "region"] if k in obs})
    ann = [k for k in obs.keys() if "anno" in k.lower() or "region" in k.lower()]
    for k in ann: df[k] = col(obs, k)
df56 = df[df.run_id.isin(["run5", "run6"])].copy()
df56["cell_id"]=[i.split(":",1)[1] for i in idx[df.run_id.isin(["run5","run6"]).values]]
df56.to_parquet(sys.argv[2])
print("\nrun5/6 cells:", len(df56), "sections:", df56.sample_id.nunique())
print(df56.groupby(["run_id", "sample_id"], observed=True).agg(n=("condition", "size"), cond=("condition", lambda s: ",".join(sorted(set(map(str, s))))), model=("model", lambda s: ",".join(sorted(set(map(str, s))))), stage=("stage", lambda s: ",".join(sorted(set(map(str, s))))), region=("region", lambda s: ",".join(sorted(set(map(str, s)))))).to_string())
