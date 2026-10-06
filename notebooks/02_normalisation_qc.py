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
# # 02 — Background, autofluorescence, tissue edge, normalisation, QC
#
# Per section, from pyramid level 3 (1.7 µm/px):
# - **tissue** mask (smoothed DAPI+18S, Otsu), **distance to tissue edge** per cell
# - **cell-free** tissue: > 5 µm from any segmented cell
# - **autofluorescence (AF) proxy**: cell-free *and* low transcript density (< 25th percentile of tissue) —
#   DAPI-negative, transcript-poor pixels; split by the anatomical region of the nearest annotated cell (WM vs GM)
# - **local background** per channel: Gaussian-weighted (σ 25 µm) mean of cell-free pixels, sampled at each cell
#
# Normalisation (intensity features only): `(x − local background) / s_section`, where `s_section` is the median
# background-subtracted cell mean of **physiological-niche** cells in that section (`Curated_niche_state`), so
# lesion-rich sections are not scaled towards their lesion content. Ratio-type features (radial, polarity,
# texture, morphology) are left as they are.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import block_reduce

from beyondboundaries import background as bgm
from beyondboundaries import data, plotting
from beyondboundaries.io import XeniumBundle, find_bundles

plotting.style()
SRC = "notebooks/02_normalisation_qc.ipynb"
OUT = ROOT / "results" / "02_normalisation_qc"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "qc"
CACHE.mkdir(parents=True, exist_ok=True)
cfg = data.load_config()
CH = data.CHANNELS
bundles = find_bundles(cfg)
obs = data.load_obs(cfg)
WM = ["WM", "WM_Meningeal"]
GM = ["GM", "DorsalHorn", "VentralHorn"]

# %% [markdown]
# ## Per-section maps (cached in `data/qc/`)

# %%
for sec, path in bundles.items():
    if (CACHE / f"{sec}_covariates.parquet").exists():
        continue
    b = XeniumBundle(path)
    m = bgm.section_background(b)
    feats = pd.read_parquet(ROOT / cfg["features_dir"] / f"{sec}.parquet",
                            columns=["x_centroid", "y_centroid"])
    cov = bgm.sample_at_cells(m, feats.x_centroid.values, feats.y_centroid.values, CH)
    cov.index = sec + ":" + feats.index
    cov.to_parquet(CACHE / f"{sec}_covariates.parquet")
    o = obs[obs.sample_id == sec]
    xy = np.c_[o.index.map(feats.x_centroid.rename(lambda i: sec + ":" + i)),
               o.index.map(feats.y_centroid.rename(lambda i: sec + ":" + i))]
    for which in ("af", "cellfree"):
        px = bgm.pixels_by_region(m, xy, o.Global_anatomical_region.astype(str).values, CH, which=which)
        px.groupby("region")[CH].agg(["median", "size"]).to_parquet(CACHE / f"{sec}_{which}_by_region.parquet")
    thumbs = {ch: block_reduce(m["img"][ch], 4, np.mean) for ch in CH}
    masks = {k: block_reduce(m[k], 4, np.max) for k in ("tissue", "cellfree", "af")}
    np.savez_compressed(CACHE / f"{sec}_thumbs.npz", **thumbs, **masks,
                        **{f"bg_{ch}": block_reduce(np.nan_to_num(m[f"bg_{ch}"]), 4, np.mean) for ch in CH})
    print(sec, "done", flush=True)

# %% [markdown]
# ## Tissue / cell-free / AF masks — one section as an example

# %%
EX = "run5_C2_G1_Mid_0088858"
t = np.load(CACHE / f"{EX}_thumbs.npz")
fig, axs = plt.subplots(3, 1, figsize=(12, 5.5))
axs[0].imshow(np.log1p(t["dapi"] + t["r18s"]), cmap="gray"); axs[0].set_title("DAPI + 18S (log), level 5")
over = np.zeros((*t["tissue"].shape, 3))
over[t["tissue"]] = (0.85, 0.85, 0.85)
over[t["cellfree"]] = (0.53, 0.71, 0.94)
over[t["af"]] = (0.92, 0.41, 0.20)
axs[1].imshow(over); axs[1].set_title("tissue (grey) · cell-free (blue) · AF proxy: cell-free & transcript-poor (orange)")
axs[2].imshow(t["bg_bnd"], cmap=plotting.SEQ); axs[2].set_title("local background, ATP1A1/CD45/E-Cad")
for a in axs:
    a.axis("off")
fig.tight_layout()
plotting.save_fig(fig, "masks_example", OUT, SRC)

# %% [markdown]
# ## Autofluorescence: white vs grey matter
# Median intensity of AF-proxy pixels (cell-free, DAPI-negative, transcript-poor), per section and region.
# If myelin autofluorescence matters, WM should be brighter than GM in the channels that are *not* expected
# to be extracellular.

# %%
rows = []
for sec in bundles:
    for which in ("af", "cellfree"):
        d = pd.read_parquet(CACHE / f"{sec}_{which}_by_region.parquet")
        for reg in d.index:
            for ch in CH:
                rows.append(dict(section=sec, which=which, region=reg, channel=ch,
                                 median=d.loc[reg, (ch, "median")], npx=d.loc[reg, (ch, "size")]))
af = pd.DataFrame(rows)
af["class"] = np.select([af.region.isin(WM), af.region.isin(GM)], ["WM", "GM"], "other")
af_wg = (af[(af.which == "af") & (af["class"] != "other")]
         .groupby(["section", "channel", "class"])
         .apply(lambda g: np.average(g["median"], weights=g.npx), include_groups=False).unstack("class"))
af_wg["WM/GM"] = af_wg.WM / af_wg.GM
af_wg.to_csv(OUT / "af_wm_vs_gm_by_section.csv")
af_sum = af_wg.groupby("channel")[["GM", "WM", "WM/GM"]].median().loc[CH].round(2)
af_sum

# %%
fig, axs = plt.subplots(1, 4, figsize=(11, 3))
for ax, ch in zip(axs, CH):
    d = af_wg.xs(ch, level="channel")
    ax.plot([0, 1], [d.GM, d.WM], color="#bbbbbb", lw=0.8)
    ax.scatter(np.zeros(len(d)), d.GM, s=12, color=plotting.CATEGORICAL[0], zorder=3, label="GM")
    ax.scatter(np.ones(len(d)), d.WM, s=12, color=plotting.CATEGORICAL[1], zorder=3, label="WM")
    ax.set_xticks([0, 1], ["GM", "WM"]); ax.set_xlim(-0.4, 1.4)
    ax.set_title(f"{plotting.CHANNEL_LABELS[ch]}\nWM/GM median {af_sum.loc[ch, 'WM/GM']:.2f}")
axs[0].set_ylabel("AF-proxy pixel intensity (raw)")
fig.tight_layout()
plotting.save_fig(fig, "af_wm_vs_gm", OUT, SRC)

# %% [markdown]
# How large is background relative to cell signal? Ratio of local background to the cell's own mean
# (median over cells), by region of the cell. Values near 1 mean the feature is mostly background.

# %%
cov = pd.concat(pd.read_parquet(p) for p in sorted(CACHE.glob("*_covariates.parquet")))
f = data.load_features(cfg)
f = f.join(cov, how="left")
f["seg"] = f.segmentation_method.map(plotting.SEG_SHORT)
f["region_class"] = np.select([f.Global_anatomical_region.isin(WM), f.Global_anatomical_region.isin(GM)],
                              ["WM", "GM"], "other")
bg_rel = pd.DataFrame({ch: (f[f"{ch}_bg_local"] / f[f"{ch}_cell_mean"]).groupby(f.region_class).median()
                       for ch in CH}).round(2)
bg_rel

# %% [markdown]
# ## Distance to tissue edge
# Edge cells (meninges, roots, cut surfaces) get edge artefacts and different extracellular context.

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 2]})
axs[0].hist(f.edge_um.clip(upper=600), bins=60, color=plotting.CATEGORICAL[0])
axs[0].set(xlabel="distance to tissue edge (µm, clipped 600)", ylabel="cells")
e = f.groupby("Anno_L1_curated", observed=True).edge_um.median().sort_values()
e = e[f.Anno_L1_curated.value_counts().reindex(e.index) >= 200]
axs[1].barh(e.index, e.values, color=plotting.CATEGORICAL[0], height=0.6)
axs[1].set(xlabel="median distance to tissue edge (µm)")
fig.tight_layout()
plotting.save_fig(fig, "edge_distance", OUT, SRC)
print("cells within 20 µm of edge:", (f.edge_um < 20).mean().round(3), "| outside tissue mask:", (~f.in_tissue.astype(bool)).mean().round(4))

# %% [markdown]
# ## Normalisation

# %%
ref = (f.Curated_niche_state == "Physiological").values
print("reference (physiological-niche) cells per section:")
print(f[ref].groupby("section_id").size().describe().round(0).to_dict())
fn = bgm.normalise(f, CH, ref, bg="local")
scales = fn.groupby("section_id")[[f"{ch}_scale" for ch in CH]].first().round(1)
scales

# %%
oli = (f.Anno_L1_curated == "Oligodendrocyte") & ref
secs = sorted(f.section_id.unique())
fig, axs = plt.subplots(2, 4, figsize=(13, 5.5), sharey="row")
for j, ch in enumerate(CH):
    for i, (frame, lab) in enumerate([(f, "raw cell mean"), (fn, "normalised cell mean")]):
        vals = [frame.loc[oli & (f.section_id == s), f"{ch}_cell_mean"] for s in secs]
        bp = axs[i, j].boxplot(vals, showfliers=False, patch_artist=True, widths=0.6,
                               medianprops=dict(color="black", lw=1))
        for bx in bp["boxes"]:
            bx.set(facecolor=plotting.CHANNEL_COLORS[ch], edgecolor="none")
        axs[i, j].set_xticks([])
        if j == 0:
            axs[i, j].set_ylabel(lab)
    axs[0, j].set_title(plotting.CHANNEL_LABELS[ch])
fig.suptitle("Physiological-niche oligodendrocytes, one box per section (18): before vs after normalisation", y=1.0)
fig.tight_layout()
plotting.save_fig(fig, "normalisation_oligos", OUT, SRC)

# %%
def cv_between_sections(frame):
    med = frame[oli].groupby(f.section_id[oli])[[f"{ch}_cell_mean" for ch in CH]].median()
    return (med.std() / med.mean().abs()).round(3)

pd.DataFrame({"raw": cv_between_sections(f), "normalised": cv_between_sections(fn)})

# %% [markdown]
# ## QC after normalisation: by segmentation method
# Interior (18S)-segmented cells dominate; boundary and nucleus-expansion cells are few but must be checked
# separately because the stains defined the masks.

# %%
segs = ["interior (18S)", "boundary", "nucleus exp."]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, ch in zip(axs, CH):
    for k, s in enumerate(segs):
        v = fn.loc[fn.seg == s, f"{ch}_cell_mean"].dropna()
        lo, hi = np.nanpercentile(fn[f"{ch}_cell_mean"], [0.5, 99.5])
        ax.hist(v.clip(lo, hi), bins=80, density=True, histtype="step", lw=1.5, color=plotting.CATEGORICAL[k],
                label=f"{s} (n={len(v):,})")
    ax.set_title(plotting.CHANNEL_LABELS[ch]); ax.set_xlabel("normalised cell mean")
axs[0].set_ylabel("density"); axs[-1].legend(fontsize=7, loc="upper right")
fig.tight_layout()
plotting.save_fig(fig, "normalised_by_segmethod", OUT, SRC)
pd.crosstab(fn.Anno_L1_curated, fn.seg, normalize="index").round(3)

# %% [markdown]
# ## Spatial heatmaps of each channel (level 5, 6.8 µm/px)
# Same contrast per channel across sections (1st–99.5th percentile of tissue pixels pooled over sections).

# %%
th = {s: np.load(CACHE / f"{s}_thumbs.npz") for s in secs}
for ch in CH:
    pool = np.concatenate([th[s][ch][th[s]["tissue"]] for s in secs])
    lo, hi = np.percentile(pool, [1, 99.5])
    fig, axs = plt.subplots(6, 3, figsize=(14, 12))
    for ax, s in zip(axs.ravel(), secs):
        im = np.where(th[s]["tissue"], th[s][ch], np.nan)
        ax.imshow(im, cmap=plotting.SEQ, vmin=lo, vmax=hi)
        ax.set_title(s.replace("_00", " "), fontsize=8); ax.axis("off")
    fig.suptitle(f"{plotting.CHANNEL_LABELS[ch]} (raw, common scale)")
    fig.tight_layout()
    plotting.save_fig(fig, f"spatial_{ch}", OUT, SRC)
    plt.show()

# %% [markdown]
# ## Save the normalised table
# `data/features_norm.parquet`: annotated cells, normalised intensities + all other features + covariates
# (edge distance, local/global background, scales) + annotation columns.

# %%
fn.drop(columns=["in_tissue"]).to_parquet(ROOT / "data" / "features_norm.parquet")
print(fn.shape)
