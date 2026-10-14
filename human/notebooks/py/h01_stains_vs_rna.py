# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (bb)
#     language: python
#     name: bb
# ---

# %% [markdown]
# # h01 — Human MS / Baló: do the segmentation stains report their proteins?
#
# In mouse the boundary channel (ATP1A1 + CD45 + E-cadherin) behaved as ATP1A1 only: CD45 was undetectable, so the
# kit's immune-cell marker was lost. The human panel carries the RNA for every protein in the stains (*PTPRC* = CD45,
# *ATP1A1*, *CDH1* = E-cadherin, *VIM*; *ACTA2* is not on it), so here each stain can be checked against its own gene,
# cell by cell:
# 1. **CD45.** Are immune cells (lymphocytes, myeloid cells; high *PTPRC*) brighter at their membrane in the boundary
#    channel than (a) other cell types and (b) their own non-immune neighbours at the same spot? Does membrane
#    brightness rise with *PTPRC* within immune cells?
# 2. **ATP1A1, E-cadherin, vimentin.** Per cell type, does the channel follow the gene's RNA?
# 3. **Seeing it.** Random immune cells next to random oligodendrocytes, boundary channel and DAPI.
#
# Data: 4 sections (MS spinal cord S07-256, S13-019; Baló Balo-1, Balo-2), one run. Cell types: marker + manual
# labels (`human/scripts/01_label_clusters.py`); "low quality" cells excluded. Image features: the mouse pipeline
# (`scripts/01_extract_features.py --config human/config.yaml`), raw intensities, compared **within section**.
# Policy: no per-cell data leaves the analysis Mac; figures and summary tables only.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parents[1] if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from skimage.segmentation import find_boundaries

from beyondboundaries import plotting
from beyondboundaries.io import XeniumBundle, find_bundles

plotting.style()
SRC = "human/notebooks/h01_stains_vs_rna.ipynb"
OUT = ROOT / "human" / "results" / "h01_stains_vs_rna"
OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(open(ROOT / "human" / "config.yaml"))
IMMUNE = ["Lymphocyte (T/B/plasma)", "Myeloid", "Myeloid (SPP1+ activated)"]
GENES = ["PTPRC", "ATP1A1", "CDH1", "VIM", "GFAP", "CD3E", "CD68", "P2RY12", "CD163"]
SAMPLES = ["S07-256", "S13-019", "Balo-1", "Balo-2"]

# %%
a = ad.read_h5ad(CFG["annotation"]["h5ad"], backed="r")
ct = pd.read_parquet(ROOT / "human" / "data" / "cell_types.parquet")
cnt = a[:, GENES].to_memory().layers["counts"]
cnt = pd.DataFrame(cnt.toarray() if hasattr(cnt, "toarray") else cnt, index=a.obs_names, columns=GENES)
cells = ct.join(cnt).join(a.obs[["total_counts"]])
a.file.close()
cells = cells[cells.cell_type != "low quality"]
bundles = find_bundles(CFG)
sec_of = {sid.split("_")[1]: sid for sid in bundles}
feats = []
for smp, sid in sec_of.items():
    f = pd.read_parquet(ROOT / CFG["features_dir"] / f"{sid}.parquet")
    if "cell_id" not in f.columns:
        f = f.reset_index()
    f["sample_id"] = smp
    feats.append(f.reset_index(drop=True))
feats = pd.concat(feats)
d = cells.reset_index().merge(feats.drop(columns=["x_centroid", "y_centroid", "segmentation_method"]),
                              on=["sample_id", "cell_id"], how="inner").set_index("index")
for ch in ["bnd", "smavim", "r18s", "dapi"]:
    d[f"{ch}_rim_contrast"] = np.log((d[f"{ch}_rim_mean"] + 1) / (d[f"{ch}_ring_mean"] + 1))
    g = d.groupby("sample_id")[f"{ch}_cell_mean"]
    d[f"{ch}_z"] = (d[f"{ch}_cell_mean"] - g.transform("median")) / (
        (d[f"{ch}_cell_mean"] - g.transform("median")).abs().groupby(d.sample_id).transform("median") * 1.4826)
d["immune"] = d.cell_type.isin(IMMUNE)
for gname in GENES:
    d[f"{gname}_per1k"] = d[gname] / d.total_counts.clip(lower=1) * 1000
print(f"{len(d):,} cells with features;", d.groupby("sample_id").size().to_dict())
pd.crosstab(d.cell_type, d.sample_id)

# %% [markdown]
# ## 1. CD45: are immune cells brighter at the membrane in the boundary channel?
# Two readouts per cell: **membrane contrast** = log(rim mean / ring mean), the cell's outline against the 2 µm just
# outside it (CD45 sits in the membrane, neuropil ATP1A1 around it), and the **cell mean** (robust z within the
# section). Per section and cell type: median. Positive membrane contrast in immune cells, but not in their
# neighbours, would be CD45.

# %%
tab = d.groupby(["cell_type", "sample_id"]).agg(cells=("bnd_rim_contrast", "size"),
                                                membrane_contrast=("bnd_rim_contrast", "median"),
                                                cell_mean_z=("bnd_z", "median"),
                                                PTPRC_per1k=("PTPRC_per1k", "mean")).round(3)
tab = tab[tab.cells >= 30]
tab.to_csv(OUT / "boundary_channel_by_celltype.csv")
tab.unstack("sample_id")["membrane_contrast"].sort_values("S13-019", ascending=False)

# %%
order = d.groupby("cell_type").bnd_rim_contrast.median().sort_values().index
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for ax, col, lab in [(axs[0], "bnd_rim_contrast", "membrane contrast log(rim / ring)"),
                     (axs[1], "bnd_z", "cell mean (robust z within section)")]:
    data = [d.loc[d.cell_type == t, col].clip(*d[col].quantile([0.01, 0.99])) for t in order]
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor("#e74c3c" if t in IMMUNE else "#bbbbbb")
    ax.set_yticks(range(1, len(order) + 1), order, fontsize=8)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(lab)
axs[0].set_title("boundary channel (ATP1A1/CD45/E-cad) per cell type; red = immune")
fig.tight_layout()
plotting.save_fig(fig, "boundary_channel_by_celltype", OUT, SRC)

# %% [markdown]
# **Paired with the neighbour.** Each immune cell against its nearest non-immune cell of the same section (within
# 30 µm), so neuropil and tissue brightness are shared: the immune − neighbour difference in membrane contrast and in
# cell mean, per section (median, share of pairs where the immune cell is brighter, Wilcoxon). Control: the same for
# oligodendrocytes paired with their nearest non-oligodendrocyte.

# %%
def paired(d, focal, label):
    rows = []
    for smp, g in d.groupby("sample_id"):
        F, O = g[focal(g)], g[~focal(g)]
        if len(F) < 30:
            continue
        dist, idx = cKDTree(O[["x_centroid", "y_centroid"]].to_numpy()).query(F[["x_centroid", "y_centroid"]].to_numpy())
        ok = dist <= 30
        for col, nm in [("bnd_rim_contrast", "membrane contrast"), ("bnd_z", "cell mean z")]:
            diff = F[col].to_numpy()[ok] - O[col].to_numpy()[idx[ok]]
            rows.append(dict(focal=label, sample=smp, readout=nm, pairs=int(ok.sum()), median_diff=np.median(diff),
                             share_brighter=(diff > 0).mean(), p=wilcoxon(diff).pvalue))
    return rows


pr = pd.DataFrame(paired(d, lambda g: g.cell_type == "Lymphocyte (T/B/plasma)", "lymphocytes")
                  + paired(d, lambda g: g.cell_type.isin(["Myeloid", "Myeloid (SPP1+ activated)"]), "myeloid")
                  + paired(d, lambda g: g.cell_type == "Oligodendrocyte", "oligodendrocytes (control)"))
pr.round(4).to_csv(OUT / "immune_vs_neighbour.csv", index=False)
pr.round(3)

# %% [markdown]
# **Dose–response within immune cells.** If the boundary channel carries CD45, membrane contrast should rise with
# *PTPRC* RNA (per 1,000 transcripts) across immune cells of the same section. Spearman per section and immune type;
# control: the same against *ATP1A1* RNA.

# %%
rows = []
for smp, g in d.groupby("sample_id"):
    for t in IMMUNE:
        s = g[g.cell_type == t]
        if len(s) < 50:
            continue
        for gene in ["PTPRC", "ATP1A1"]:
            r = spearmanr(s[f"{gene}_per1k"], s.bnd_rim_contrast)
            rows.append(dict(sample=smp, cell_type=t, gene=gene, cells=len(s), rho=r.statistic, p=r.pvalue))
dose = pd.DataFrame(rows)
dose.round(4).to_csv(OUT / "ptprc_dose_response.csv", index=False)
dose.round(3)

# %% [markdown]
# ## 2. Each stain against its own gene's RNA, per cell type
# Per section, the cell-type median of the channel (robust z) against the cell-type mean RNA (per 1,000
# transcripts) of the matching gene: boundary channel vs *ATP1A1*, *PTPRC* and *CDH1*; αSMA/vimentin channel vs *VIM*.
# Spearman over cell types (≥ 30 cells), per section.

# %%
rows, pts = [], []
for smp, g in d.groupby("sample_id"):
    agg = g.groupby("cell_type").agg(n=("bnd_z", "size"), bnd=("bnd_z", "median"), smavim=("smavim_z", "median"),
                                     **{gn: (f"{gn}_per1k", "mean") for gn in ["ATP1A1", "PTPRC", "CDH1", "VIM"]})
    agg = agg[agg.n >= 30]
    for ch, gene in [("bnd", "ATP1A1"), ("bnd", "PTPRC"), ("bnd", "CDH1"), ("smavim", "VIM")]:
        r = spearmanr(agg[ch], agg[gene])
        rows.append(dict(sample=smp, channel=ch, gene=gene, cell_types=len(agg), rho=r.statistic, p=r.pvalue))
    pts.append(agg.assign(sample=smp))
st = pd.DataFrame(rows)
st.round(3).to_csv(OUT / "stain_vs_gene_by_celltype.csv", index=False)
pts = pd.concat(pts)
st.pivot(index=["channel", "gene"], columns="sample", values="rho").round(2)

# %%
fig, axs = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, (ch, gene, lab) in zip(axs, [("bnd", "ATP1A1", "boundary channel"), ("bnd", "PTPRC", "boundary channel"),
                                     ("bnd", "CDH1", "boundary channel"), ("smavim", "VIM", "αSMA/vimentin channel")]):
    for k, smp in enumerate(SAMPLES):
        p_ = pts[pts["sample"] == smp]
        ax.scatter(p_[gene], p_[ch], s=25, color=plotting.CATEGORICAL[k], label=smp)
        if smp == "S13-019":
            for t, r in p_.iterrows():
                ax.annotate(t.split(" (")[0], (r[gene], r[ch]), fontsize=6, alpha=0.7)
    rr = st[(st.channel == ch) & (st.gene == gene)].rho
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(f"{gene} RNA per 1,000 transcripts (cell-type mean)")
    ax.set_ylabel(f"{lab} (cell-type median z)")
    ax.set_title(f"{lab} vs {gene}: ρ {rr.min():+.2f} to {rr.max():+.2f} over sections", fontsize=9)
axs[0].legend(fontsize=7)
fig.tight_layout()
plotting.save_fig(fig, "stain_vs_gene_by_celltype", OUT, SRC)

# %% [markdown]
# ## 3. Seeing it
# Random lymphocytes, random myeloid cells and random oligodendrocytes (one per row block, 6 each, seed 0) from the
# two MS sections: the boundary channel (top of each pair) and DAPI, 30 µm crops, the focal cell outlined in cyan,
# other outlines grey. Same display range per section (1st–99.5th percentile of tissue pixels).

# %%
_B, _R = {}, {}


def bundle(smp):
    if smp not in _B:
        _B[smp] = XeniumBundle(bundles[sec_of[smp]])
    return _B[smp]


def drange(smp, ch):
    if (smp, ch) not in _R:
        im = bundle(smp).read_level(ch, 3).astype(np.float32)
        v = im[im > 0]
        _R[(smp, ch)] = (np.percentile(v, 1), np.percentile(v, 99.5))
    return _R[(smp, ch)]


def crop(ax, r, ch_i, ch, w_um=30, title=""):
    b = bundle(r.sample_id)
    px = b.pixel_size
    y0, x0 = int(r.centroid_y_px - w_um / 2 / px), int(r.centroid_x_px - w_um / 2 / px)
    img, lab, _ = b.read_window(y0, x0, int(w_um / px), int(w_um / px))
    lo, hi = drange(r.sample_id, ch)
    ax.imshow(np.clip((img[ch_i] - lo) / (hi - lo), 0, 1), cmap="magma" if ch == "bnd" else "gray")
    e = find_boundaries(lab, mode="inner")
    ov = np.zeros((*lab.shape, 4))
    ov[e] = (0.6, 0.6, 0.6, 0.4)
    ov[e & (lab == r.label)] = (0.2, 0.9, 1, 1)
    ax.imshow(ov)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


rng = np.random.default_rng(0)
groups = [("lymphocyte", d.cell_type == "Lymphocyte (T/B/plasma)"), ("myeloid", d.cell_type == "Myeloid"),
          ("oligodendrocyte", d.cell_type == "Oligodendrocyte")]
fig, axs = plt.subplots(6, 6, figsize=(14, 14.5))
for gi, (nm, m) in enumerate(groups):
    s = d[m & d.sample_id.isin(["S07-256", "S13-019"])]
    pick = s.iloc[rng.choice(len(s), 6, replace=False)]
    for j, (_, r) in enumerate(pick.iterrows()):
        crop(axs[2 * gi, j], r, 1, "bnd", title=f"{nm} · {r.sample_id} · PTPRC {int(r.PTPRC)} · contrast {r.bnd_rim_contrast:+.2f}")
        crop(axs[2 * gi + 1, j], r, 0, "dapi", title="DAPI")
fig.suptitle("boundary channel (magma) and DAPI: random lymphocytes, myeloid cells and oligodendrocytes (MS sections); "
             "focal cell cyan", fontsize=9)
fig.tight_layout()
plotting.save_fig(fig, "gallery_immune_vs_oligo", OUT, SRC)

# %% [markdown]
# ## Findings
# (filled in after the run)
