"""Where each transcript sits relative to the cell outlines, human MS/Baló sections (input to h04).

Reuses `scripts/04_transcript_compartments.py` (mouse) unchanged, with human gene names for the focus sets, the human
cell labels (`human/scripts/01_label_clusters.py`) and output under human/data/subcellular/ (analysis Mac only, never
git). Same outputs per section: {sid}.npz (in-cell and nuclear counts, halo counts and area) and
{sid}_points.parquet (every focus-gene transcript with compartment, depth, nearest cell and distance).

usage: python human/scripts/02_transcript_compartments.py   (sections run one after another: the reused module is
loaded from a path, which worker processes cannot import)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "04_transcript_compartments.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)
from beyondboundaries.io import find_bundles  # noqa: E402

tc.OUT = ROOT / "human" / "data" / "subcellular"
tc.FOCUS = {
    "myelin (transported)": ["MBP", "MAG", "MOG", "CNP", "OPALIN", "ENPP6", "CLDN11", "MAL", "MOBP"],
    "oligo soma (control)": ["SOX10", "OLIG1", "OLIG2", "NKX6-2", "MYRF", "ST18"],
    "axon / neuron": ["NEFL", "NEFM", "NEFH", "SNAP25", "GAP43", "CAMK2A", "SYT1", "STMN2", "RBFOX3"],
    "phagocyte": ["CD68", "TREM2", "LPL", "CD36", "GPNMB", "AXL", "SPP1", "APOC1"],
    "astrocyte": ["GFAP", "AQP4", "SLC1A2", "SLC1A3"],
    "nuclear stress": ["NEAT1", "HSPA1A", "HSPA1B", "DDIT3", "FOS", "JUN", "ATF4", "HSPH1"],
}


def main():
    argparse.ArgumentParser().parse_args()
    tc.OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "human" / "config.yaml"))
    bundles = find_bundles(cfg)
    a = ad.read_h5ad(cfg["annotation"]["h5ad"])
    ct = pd.read_parquet(ROOT / "human" / "data" / "cell_types.parquet")
    obs = ct[["sample_id", "cell_id"]].copy()
    obs["Anno_L1_curated"] = ct.cell_type
    obs = obs[ct.cell_type != "low quality"]
    if (tc.OUT / "gene_owner.csv").exists():
        owner = pd.read_csv(tc.OUT / "gene_owner.csv", index_col=0).owner.reindex(a.var_names).fillna("shared")
    else:
        owner = tc.gene_owners(a[obs.index], obs)
        owner.to_csv(tc.OUT / "gene_owner.csv")
    genes = a.var_names.copy()
    del a
    sid_of = {sid.split("_")[1]: sid for sid in bundles}
    for smp in sorted(obs.sample_id.unique()):
        sid, msg = tc.run_section(sid_of[smp], bundles[sid_of[smp]], obs[obs.sample_id == smp], genes, owner)
        print(sid, msg, flush=True)


if __name__ == "__main__":
    main()
