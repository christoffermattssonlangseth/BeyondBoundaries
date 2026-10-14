"""Marker-based cell-type labels for the Leiden clusters of the human MS/Baló object (it has clusters only).

Per Leiden cluster (resolution given by --leiden): mean of z-scored log-normalised expression of each marker set; the
cluster gets the best-scoring type if it beats the runner-up by --margin, else 'mixed: A/B'. Writes (not for git):
  human/data/cell_types.parquet   per cell: sample_id, cluster, cell_type, x/y centroid
and, for review (figures may be committed):
  human/results/01_labels/cluster_scores.csv, marker_dotplot.png

usage: python human/scripts/01_label_clusters.py [--leiden leiden_1.5] [--margin 0.3]
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
MARKERS = {
    "Oligodendrocyte": ["MBP", "PLP1", "MOBP", "MOG", "MAG", "ST18", "OPALIN", "CNP"],
    "OPC": ["PDGFRA", "CSPG4", "VCAN", "PCDH15", "OLIG1"],
    "Astrocyte": ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "GJA1", "SOX9"],
    "Microglia": ["P2RY12", "CX3CR1", "TMEM119", "CSF1R", "C1QA", "C1QB"],
    "Macrophage": ["CD163", "MRC1", "LYZ", "CD14", "CD68", "F13A1"],
    "T cell": ["CD3E", "CD3D", "CD2", "CD8A", "IL7R", "CD247"],
    "B / plasma cell": ["MS4A1", "CD79A", "JCHAIN", "MZB1", "IGHG1", "CD19"],
    "Neuron": ["RBFOX3", "SNAP25", "SYT1", "NEFL", "STMN2", "GAP43"],
    "Endothelial": ["CLDN5", "PECAM1", "VWF", "FLT1", "ABCB1"],
    "Pericyte / VSMC": ["ACTA2", "MYH11", "PDGFRB", "RGS5", "TAGLN"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Ependymal": ["FOXJ1", "CFAP43", "DNAH9", "RSPH1"],
    "Schwann": ["MPZ", "PMP22", "PRX", "NCMAP"],
}
# Manual labels for leiden_1.5 clusters the marker scores get wrong or call "mixed", decided from each cluster's top
# genes (mean log-expression in cluster minus rest; checked 2026-10-14). The fibroblast collagen genes are not on the
# panel, so PDGFRA alone cannot separate fibroblasts from OPCs.
MANUAL = {
    "20": "Fibroblast",            # VIM, TNXB, PI16, FMOD, CXCL12, LTBP2, C4A/B (meningeal / perivascular)
    "21": "Fibroblast",            # VIM, TNXB, NBL1, COL18A1, FMOD, LTBP2, CXCL12
    "22": "OPC",                   # PTPRZ1, TNR, BCAN, SOX10, PDGFRA, OLIG2, CSPG4
    "28": "OPC",                   # PTPRZ1, BCAN, TNR, OLIG1/2, SOX8 (Balo sections)
    "13": "Lymphocyte (T/B/plasma)",  # CXCR4, CD3E, CD2, CD8A, CD79A, MS4A1, XBP1, MZB1
    "7": "Myeloid (SPP1+ activated)",  # SPP1, FTL, APOC1, CCL4, CD74 (Balo lesions)
    "12": "Myeloid",               # CSF1R, CD163, LYZ, CD14
    "19": "Myeloid",               # CSF1R, CD163, LYZ, CD14, CD68
    "24": "Myeloid",               # CD163, MRC1, LYZ, CD14, CD68, F13A1 (macrophage)
    "1": "Myeloid",                # P2RY12, CX3CR1, TMEM119, CSF1R (microglia)
    "14": "Astrocyte (reactive)",  # GFAP, SERPINA3, CD44, C4A/B, APOE
    "25": "Perivascular (mixed)",  # COL4A1, CLDN5, CD163, CXCL12, ADAMTS1
    "27": "Neuron (with oligodendrocyte signal)",  # SNAP25, SYP, SLC17A7 + CNP, OPALIN (Balo only)
    "9": "low quality",            # median 104 transcripts; MT-CO1, MCL1, NR4A1, CDKN1A (stress)
    "11": "low quality",           # median 132 transcripts; MT-CO1/2 with GFAP + MBP + CD74 (mixed)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leiden", default="leiden_1.5")
    ap.add_argument("--margin", type=float, default=0.3)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "human" / "config.yaml"))
    out_fig = ROOT / "human" / "results" / "01_labels"
    out_fig.mkdir(parents=True, exist_ok=True)
    (ROOT / "human" / "data").mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(cfg["annotation"]["h5ad"])
    print(a)
    genes = {k: [g for g in v if g in a.var_names] for k, v in MARKERS.items()}
    print({k: len(v) for k, v in genes.items()})
    allg = sorted({g for v in genes.values() for g in v})
    X = a[:, allg].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if X.max() > 50:                                     # counts -> log-normalised
        tot = np.asarray(a.layers["counts"].sum(1)).ravel() if "counts" in a.layers else X.sum(1)
        X = np.log1p(X / np.maximum(tot, 1)[:, None] * 1e4)
    Z = pd.DataFrame((X - X.mean(0)) / (X.std(0) + 1e-9), columns=allg, index=a.obs_names)
    cl = a.obs[args.leiden].astype(str)
    S = pd.DataFrame({k: Z[v].mean(1) for k, v in genes.items() if v}).groupby(cl.values).mean()
    best, second = S.idxmax(1), S.apply(lambda r: r.nlargest(2).index[1], axis=1)
    gap = S.max(1) - S.apply(lambda r: r.nlargest(2).iloc[1], axis=1)
    lab = pd.Series(np.where(gap >= args.margin, best, "mixed: " + best + "/" + second), index=S.index)
    S["marker_call"] = lab
    if args.leiden == "leiden_1.5":
        lab = lab.rename(index=str).copy()
        for c, v in MANUAL.items():
            lab[c] = v
    S["label"], S["margin"], S["cells"] = lab.reindex(S.index), gap, cl.value_counts().reindex(S.index)
    S = S.sort_values("label")
    S.round(3).to_csv(out_fig / "cluster_scores.csv")
    print(S[["label", "margin", "cells"]].to_string())
    cells = a.obs[["sample_id", "x_centroid", "y_centroid", "segmentation_method"]].copy()
    if "cell_id" in a.obs:
        cells["cell_id"] = a.obs.cell_id.astype(str)
    cells["cluster"] = cl
    cells["cell_type"] = cl.map(lab)
    cells.to_parquet(ROOT / "human" / "data" / "cell_types.parquet")
    print(cells.groupby(["sample_id", "cell_type"], observed=True).size().unstack(0).fillna(0).astype(int).to_string())
    # dot plot: marker mean (z) per cluster, ordered by label
    order = S.index.tolist()
    M = Z.groupby(cl.values).mean().reindex(order)
    frac = (pd.DataFrame(X > 0, columns=allg, index=a.obs_names)).groupby(cl.values).mean().reindex(order)
    gorder = [g for v in genes.values() for g in v if g in M.columns]
    gorder = list(dict.fromkeys(gorder))
    fig, ax = plt.subplots(figsize=(0.28 * len(gorder) + 3, 0.28 * len(order) + 2))
    yy, xx = np.meshgrid(range(len(order)), range(len(gorder)), indexing="ij")
    ax.scatter(xx.ravel(), yy.ravel(), s=80 * frac[gorder].to_numpy().ravel(), c=M[gorder].to_numpy().ravel(),
               cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(gorder)), gorder, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), [f"{c}: {S.loc[c, 'label']} ({S.loc[c, 'cells']:,})" for c in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"{args.leiden}: marker expression (colour = mean z, size = fraction expressing)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig / "marker_dotplot.png", dpi=130)


if __name__ == "__main__":
    main()
