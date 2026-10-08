"""Where each transcript sits relative to the stain-based cell outlines (runs 5/6; input to notebook 23).

For every decoded gene transcript (qv >= 20) of the 18 run-5/6 sections:
  - in a cell (Xenium assignment) or not; in its nucleus or not (`overlaps_nucleus`);
  - depth inside the cell: distance to the cell's outline (µm, from the mask at 2x downsampling, 0.425 µm);
  - outside cells: the nearest cell (outline) and the distance to it.

Writes, per section, to data/subcellular/:
  {sid}.npz      annotated cells (rows = `obs_names`) x panel genes (h5ad var order):
                 `total`, `nuclear` (sparse in-cell counts), `halo` (cells x owner groups: transcripts outside cells
                 whose nearest cell is this one, within HALO_UM), `halo_all`, `halo_area_um2` (free area that is
                 nearest to this cell within HALO_UM), `owners` (group names)
  {sid}_points.parquet   every transcript of the FOCUS genes, with its compartment, depth, nearest cell and distance
Plus gene_owner.csv: each gene's owner cell type (the type with >= 50 % of the summed per-type mean expression in
runs 5/6, else 'shared'), used to label what the RNA outside cells is made of.

usage: python scripts/04_transcript_compartments.py [--jobs 2] [--sections run5_C2_G1_Bot_0088858 ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beyondboundaries.io import XeniumBundle, find_bundles  # noqa: E402

OUT = ROOT / "data" / "subcellular"
F = 2            # mask downsampling for the distance maps
HALO_UM = 10.0   # transcripts outside cells count towards the nearest cell up to this distance
POINT_UM = 30.0  # focus-gene transcripts outside cells are kept up to this distance from a cell
MIN_QV = 20
FOCUS = {
    "myelin (transported)": ["Mbp", "Mag", "Mog", "Cnp", "Opalin", "Enpp6", "Cldn11", "Mal"],
    "oligo soma (control)": ["Sox10", "Olig1", "Olig2", "Nkx6-2", "Myrf", "St18"],
    "axon / neuron": ["Nefl", "Nefm", "Nefh", "Snap25", "Gap43", "Camk2a", "Syt1", "Stmn2", "Rbfox3"],
    "phagocyte": ["Cd68", "Trem2", "Lpl", "Cd36", "Gpnmb", "Axl"],
    "astrocyte": ["Gfap", "Aqp4", "Slc1a2", "Slc1a3"],
    "nuclear stress": ["Neat1", "Hspa1a", "Hspa1b", "Ddit3", "Fos", "Jun", "Atf4", "Hsph1"],
}


def gene_owners(a, obs) -> pd.Series:
    X = a.layers["counts"]
    types = obs.Anno_L1_curated.astype(str).to_numpy()
    means = {}
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        if len(idx) >= 200:
            means[t] = np.asarray(X[idx].mean(0)).ravel()
    M = pd.DataFrame(means, index=a.var_names)
    share = M.div(M.sum(1).replace(0, np.nan), axis=0)
    return pd.Series(np.where(share.max(1) >= 0.5, share.idxmax(1), "shared"), index=a.var_names, name="owner")


def mask_lowres(b: XeniumBundle) -> np.ndarray:
    cell_m, _, _ = b._open_seg()
    H = cell_m.shape[0]
    step = 4096  # multiple of F so the strips subsample on the same grid
    return np.concatenate([cell_m[y:y + step][::F, ::F] for y in range(0, H, step)])


def run_section(sid: str, path: Path, sec_obs: pd.DataFrame, genes: pd.Index, owner: pd.Series):
    if (OUT / f"{sid}.npz").exists():
        return sid, "cached"
    b = XeniumBundle(path)
    um = b.pixel_size * F
    cells = b.cells()[["cell_id", "label"]]
    lab2row = np.full(len(cells) + 1, -1, np.int64)
    row_of = pd.Series(np.arange(len(sec_obs)), index=sec_obs.cell_id.astype(str).to_numpy())
    hit = cells.cell_id.isin(row_of.index).to_numpy()
    lab2row[cells.label.to_numpy()[hit]] = row_of[cells.cell_id[hit]].to_numpy()

    # distance maps on the 2x-downsampled label mask
    L = mask_lowres(b)
    edge = np.zeros(L.shape, bool)
    edge[:, :-1] |= L[:, :-1] != L[:, 1:]
    edge[:, 1:] |= L[:, :-1] != L[:, 1:]
    edge[:-1] |= L[:-1] != L[1:]
    edge[1:] |= L[:-1] != L[1:]
    depth = ndimage.distance_transform_edt((L > 0) & ~edge).astype(np.float32)
    del edge
    dist, ind = ndimage.distance_transform_edt(L == 0, return_indices=True)
    dist = dist.astype(np.float32)
    near = L[ind[0], ind[1]]
    del ind
    free = (L == 0) & (dist * um <= HALO_UM)
    nr = lab2row[near[free]]
    halo_area = np.bincount(nr[nr >= 0], minlength=len(sec_obs)) * um * um
    del free, nr

    # transcripts
    t = pq.read_table(path / "transcripts.parquet",
                      columns=["cell_id", "overlaps_nucleus", "feature_name", "x_location", "y_location", "qv", "is_gene"],
                      read_dictionary=["cell_id", "feature_name"])
    t = t.filter(pc.and_(pc.greater_equal(t["qv"], MIN_QV), t["is_gene"])).unify_dictionaries().combine_chunks()
    cid, gname = t["cell_id"].chunk(0), t["feature_name"].chunk(0)
    gi = pd.Series(np.arange(len(genes)), index=genes)
    gcode = gi.reindex(gname.dictionary.to_pylist()).fillna(-1).astype(np.int64).to_numpy()[gname.indices.to_numpy()]
    lab_of = pd.Series(cells.label.to_numpy(), index=cells.cell_id.to_numpy())
    clab = lab_of.reindex(cid.dictionary.to_pylist()).fillna(0).astype(np.int64).to_numpy()[cid.indices.to_numpy()]
    nuc = t["overlaps_nucleus"].to_numpy().astype(bool)
    x, y = t["x_location"].to_numpy(), t["y_location"].to_numpy()
    del t
    iy = np.clip((y / um).astype(np.int64), 0, L.shape[0] - 1)
    ix = np.clip((x / um).astype(np.int64), 0, L.shape[1] - 1)
    ok = gcode >= 0
    row = np.where(clab > 0, lab2row[clab], -1)
    inside = clab > 0
    d_out = dist[iy, ix] * um
    near_row = np.where(~inside, lab2row[near[iy, ix]], -1)
    dep = np.where(inside, depth[iy, ix] * um, np.nan).astype(np.float32)
    del dist, near, depth, L

    n, G = len(sec_obs), len(genes)
    m = ok & (row >= 0)
    total = sp.coo_matrix((np.ones(m.sum(), np.int32), (row[m], gcode[m])), shape=(n, G)).tocsr()
    m2 = m & nuc
    nuclear = sp.coo_matrix((np.ones(m2.sum(), np.int32), (row[m2], gcode[m2])), shape=(n, G)).tocsr()
    groups = np.array(sorted(owner.unique()))
    gown = pd.Series(np.arange(len(groups)), index=groups)[owner.to_numpy()].to_numpy()
    h = ok & (near_row >= 0) & (d_out <= HALO_UM)
    halo = np.zeros((n, len(groups)), np.int32)
    np.add.at(halo, (near_row[h], gown[gcode[h]]), 1)
    np.savez_compressed(OUT / f"{sid}.npz", obs_names=sec_obs.index.to_numpy().astype(str), owners=groups,
                        total_data=total.data, total_indices=total.indices, total_indptr=total.indptr,
                        nuclear_data=nuclear.data, nuclear_indices=nuclear.indices, nuclear_indptr=nuclear.indptr,
                        halo=halo, halo_all=halo.sum(1), halo_area_um2=halo_area.astype(np.float32))

    focus = [g for gs in FOCUS.values() for g in gs if g in gi.index]
    fm = np.isin(gcode, gi[focus].to_numpy()) & ((row >= 0) | ((near_row >= 0) & (d_out <= POINT_UM)))
    pts = pd.DataFrame({
        "x": x[fm].astype(np.float32), "y": y[fm].astype(np.float32),
        "gene": pd.Categorical(genes[gcode[fm]], categories=focus),
        "in_cell": inside[fm], "nucleus": nuc[fm] & inside[fm],
        "cell": np.where(row[fm] >= 0, sec_obs.index.to_numpy()[np.maximum(row[fm], 0)], None),
        "depth_um": dep[fm],
        "near_cell": np.where(near_row[fm] >= 0, sec_obs.index.to_numpy()[np.maximum(near_row[fm], 0)], None),
        "dist_um": np.where(inside[fm], 0, d_out[fm]).astype(np.float32),
    })
    pts.to_parquet(OUT / f"{sid}_points.parquet")
    return sid, f"{n} cells, {int(ok.sum()):,} transcripts, {total.sum() / max(ok.sum(), 1):.0%} in annotated cells"


def main():
    import anndata as ad
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--sections", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["runs"] = {k: v for k, v in cfg["runs"].items() if k in ("run5", "run6")}
    bundles = find_bundles(cfg)
    a = ad.read_h5ad(ROOT / "data" / "RRMAP2_all_runs.h5ad")
    keep = a.obs.run_id.isin(["run5", "run6"]).to_numpy()
    a = a[keep]
    obs = a.obs[["sample_id", "cell_id", "Anno_L1_curated"]].copy()
    if (OUT / "gene_owner.csv").exists():
        owner = pd.read_csv(OUT / "gene_owner.csv", index_col=0).owner.reindex(a.var_names).fillna("shared")
    else:
        owner = gene_owners(a, obs)
        owner.to_csv(OUT / "gene_owner.csv")
    print(owner.value_counts().to_string(), flush=True)
    genes = a.var_names.copy()
    del a
    sids = [s for s in (args.sections or sorted(obs.sample_id.astype(str).unique())) if s in bundles]
    jobs = (delayed(run_section)(s, bundles[s], obs[obs.sample_id.astype(str) == s], genes, owner) for s in sids)
    for sid, msg in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(jobs):
        print(sid, msg, flush=True)


if __name__ == "__main__":
    main()
