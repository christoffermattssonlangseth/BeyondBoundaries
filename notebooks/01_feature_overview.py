# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python (bb)
#     language: python
#     name: bb
# ---

# %% [markdown]
# # 01 — Per-cell image features: what was extracted, and does it look right?
#
# Phase 1 output (`scripts/01_extract_features.py` → `data/features/<section>.parquet`): for every cell in the
# 18 annotated spinal-cord sections (run5/run6) and each of the 4 multimodal-segmentation channels:
#
# | family | what | why it could carry biology |
# |---|---|---|
# | intensity × compartment | mean / p50 / p90 / p99 (+ integrated) in **nucleus, cytoplasm, 1 µm rim, 2 µm outer ring, 10 µm territory** | protein/RNA abundance and *where* it sits; ring/territory = extracellular context (neuropil, vessel wall) |
# | radial profile | 5 bins centre→edge, ÷ cell mean | perinuclear vs peripheral distribution |
# | polarity | intensity-weighted centroid offset (vs shape centroid / nucleus), direction | leukocyte polarisation, migration |
# | texture | Haralick on per-cell rescaled levels (DAPI on chromatin only) | chromatin condensation, punctate vs diffuse |
# | morphology | cell/nucleus area, shape, nucleus:cell ratio, nucleus offset | cell size, process-bearing vs round |
#
# This notebook checks coverage and technical concordance, shows where each stain lives, and gives a first look
# at cells. Intensities here are **raw** (no background correction — that is notebook 02).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from beyondboundaries import data, plotting
from beyondboundaries.io import XeniumBundle, find_bundles

plotting.style()
SRC = "notebooks/01_feature_overview.ipynb"
OUT = ROOT / "results" / "01_features"
OUT.mkdir(parents=True, exist_ok=True)
cfg = data.load_config()
CH = data.CHANNELS
SEG = plotting.SEG_SHORT

# %%
f = data.load_features(cfg, annotated_only=False)
f["seg"] = f.segmentation_method.map(SEG)
f["annotated"] = f.Anno_L1_curated.notna()
print(f.shape)

# %% [markdown]
# ## Coverage
# Every Xenium cell gets features; the annotated object kept a subset (QC filtering upstream).

# %%
cov = f.groupby("section_id").agg(cells=("annotated", "size"), annotated=("annotated", "sum"),
                                   truncated=("truncated", "sum"))
cov["frac_annotated"] = (cov.annotated / cov.cells).round(3)
cov.loc["total"] = cov.sum(numeric_only=True)
cov.loc["total", "frac_annotated"] = round(cov.loc["total", "annotated"] / cov.loc["total", "cells"], 3)
cov.to_csv(OUT / "coverage.csv")
cov

# %% [markdown]
# Cells not in the annotation: are they different (low transcript QC failures)?

# %%
f.groupby("annotated")[["transcript_counts", "n_genes", "morph_cell_area_um2"]].median()

# %% [markdown]
# ## Technical concordance with Xenium's own tables

# %%
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
sub = f.sample(20000, random_state=0)
ax[0].scatter(sub.cell_area, sub.morph_cell_area_um2, s=1, c=plotting.CATEGORICAL[0], alpha=0.3, rasterized=True)
ax[0].set(xlabel="Xenium cell_area (µm²)", ylabel="mask area (µm²)", title="cell area")
m = sub.nucleus_count == 1
ax[1].scatter(sub.nucleus_area[m], sub.morph_nuc_area_um2[m], s=1, c=plotting.CATEGORICAL[0], alpha=0.3, rasterized=True)
ax[1].set(xlabel="Xenium nucleus_area (µm²)", ylabel="mask area (µm²)", title="nucleus area (1 nucleus)")
err = np.hypot(f.centroid_x_px * 0.2125 - f.x_centroid, f.centroid_y_px * 0.2125 - f.y_centroid)
ax[2].hist(err, bins=50, color=plotting.CATEGORICAL[0])
ax[2].set(xlabel="centroid difference (µm)", ylabel="cells", title="centroid")
fig.tight_layout()
plotting.save_fig(fig, "concordance", OUT, SRC)
print("max centroid diff µm:", err.max().round(3), "| area ratio range:",
      (f.morph_cell_area_um2 / f.cell_area).agg(["min", "max"]).round(4).tolist())

# %% [markdown]
# ## Feature catalogue and missingness
# NaNs are structural: no nucleus → nuclear features NaN; cells fully surrounded by other cells → no ring.

# %%
fam = data.feature_families(f.columns)
cat = pd.DataFrame({"family": fam, "nan_frac": f[fam.index].isna().mean()})
cat.groupby("family").agg(n_features=("nan_frac", "size"), max_nan_frac=("nan_frac", "max")).round(3)

# %%
print("cells without nucleus:", (f.nucleus_count == 0).mean().round(4),
      "| by segmentation method:", f.groupby("seg").nucleus_count.apply(lambda s: (s == 0).mean()).round(3).to_dict())
print("cells without outer ring:", f.ring_npx.eq(0).mean().round(4))

# %% [markdown]
# ## Where does each stain live?
# Median compartment intensity relative to the whole-cell mean (raw, per cell, then median over cells).
# 18S defined ~95 % of the masks, so its cell-vs-ring contrast is partly by construction.

# %%
comps = ["nuc", "cyto", "rim", "ring", "terr"]
rel = pd.DataFrame({ch: [(f[f"{ch}_{c}_mean"] / f[f"{ch}_cell_mean"]).median() for c in comps] for ch in CH},
                   index=comps)
fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
for ax, ch in zip(axs, CH):
    ax.bar(comps, rel[ch], color=plotting.CHANNEL_COLORS[ch], width=0.6)
    ax.axhline(1, color="#888888", lw=0.8, ls="--")
    ax.set_title(plotting.CHANNEL_LABELS[ch])
axs[0].set_ylabel("compartment / cell mean (median)")
fig.tight_layout()
plotting.save_fig(fig, "compartment_profile", OUT, SRC)
rel.round(2)

# %% [markdown]
# ## Raw intensity per section and segmentation method
# Section-to-section offsets here are what notebook 02 removes.

# %%
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
secs = sorted(f.section_id.unique())
segs = ["interior (18S)", "boundary", "nucleus exp."]
for ax, ch in zip(axs, CH):
    for k, s in enumerate(segs):
        vals = [np.log10(f.loc[(f.section_id == sec) & (f.seg == s), f"{ch}_cell_mean"].clip(lower=1)) for sec in secs]
        bp = ax.boxplot(vals, positions=np.arange(len(secs)) + (k - 1) * 0.27, widths=0.22, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1))
        for b in bp["boxes"]:
            b.set(facecolor=plotting.CATEGORICAL[k], edgecolor="none")
    ax.set_ylabel(f"log10 {plotting.CHANNEL_LABELS[ch]}\ncell mean")
axs[-1].set_xticks(range(len(secs)), [s.replace("_00", " ") for s in secs], rotation=60, ha="right", fontsize=7)
axs[0].legend([plt.Rectangle((0, 0), 1, 1, fc=plotting.CATEGORICAL[k]) for k in range(3)], segs, ncol=3, loc="upper right")
fig.tight_layout()
plotting.save_fig(fig, "raw_intensity_by_section_segmethod", OUT, SRC)

# %% [markdown]
# ## First look at biology (raw, unnormalised): channel means by cell type
# z-scored across cell types per feature; a teaser for the Phase 3 sanity atlas.

# %%
a = f[f.annotated & ~f.Anno_L1_curated.isin(["Doublet", "T_B_doublet"])]
feat = [f"{ch}_{c}_mean" for ch in CH for c in ("nuc", "cyto", "rim", "terr")] + \
       ["morph_cell_area_um2", "morph_nuc_cell_ratio"]
med = a.groupby("Anno_L1_curated", observed=True)[feat].median()
n = a.Anno_L1_curated.value_counts()
med = med.loc[n[n >= 200].index]
z = (med - med.mean()) / med.std()
fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(z.values, cmap=plotting.DIV, vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(feat)), feat, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(z)), [f"{i} (n={n[i]:,})" for i in z.index], fontsize=8)
fig.colorbar(im, ax=ax, label="z across cell types", shrink=0.6)
fig.tight_layout()
plotting.save_fig(fig, "celltype_raw_heatmap", OUT, SRC)

# %% [markdown]
# ## Cell gallery
# Random cells per type from one section; composite DAPI (blue) / ATP1A1-CD45-ECad (orange) / 18S (green) /
# αSMA-Vim (magenta), cell outline white, nucleus outline grey. 30 µm crops, identical contrast across tiles.

# %%
bundles = find_bundles(cfg)
GAL_SEC = "run5_C2_G1_Mid_0088858"
b = XeniumBundle(bundles[GAL_SEC])
g = a[a.section_id == GAL_SEC]
types = [t for t in ["Neuron", "Oligodendrocyte", "OPC", "Astrocyte", "Myeloid", "T cell", "B cell", "DC",
                     "Endothelial", "VSMC", "Fibroblast", "Ependymal cell", "Schwann cell"] if (g.Anno_L1_curated == t).sum() >= 6]
half = int(15 / 0.2125)
# common contrast from section-level raw percentiles
lo = [np.percentile(a[f"{ch}_terr_p50"].dropna(), 5) for ch in CH]
hi = [np.percentile(a[f"{ch}_cell_p99"].dropna(), 99) for ch in CH]
fig, axs = plt.subplots(len(types), 6, figsize=(9, 1.55 * len(types)))
for r, t in enumerate(types):
    cells = g[g.Anno_L1_curated == t].sample(6, random_state=1)
    for c, (cid, row) in enumerate(cells.iterrows()):
        y, x = int(row.centroid_y_px), int(row.centroid_x_px)
        img, cm, nm = b.read_window(y - half, x - half, 2 * half, 2 * half)
        rgb = plotting.composite(img, CH, lo, hi)
        lab = cm[half, half]
        rgb[find_boundaries(cm == lab, mode="inner")] = 1
        rgb[find_boundaries(nm == lab, mode="inner")] = 0.6
        axs[r, c].imshow(rgb)
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, c].set_title(row.Anno_L2, fontsize=6)
    axs[r, 0].set_ylabel(t, fontsize=8)
fig.tight_layout(h_pad=0.3, w_pad=0.2)
plotting.save_fig(fig, "cell_gallery", OUT, SRC)

# %% [markdown]
# ## Summary
# Numbers referenced in RESULTS.md come from the tables above (coverage, concordance, compartment profile).
