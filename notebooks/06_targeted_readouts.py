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
# # 06 — Targeted readouts (Phase 5)
#
# Three of the four stain targets — *Vim*, *Acta2*, *Atp1a1* — are **not on the 5K panel**, so for them the images
# are the only per-cell measurement (CD45/*Ptprc* is on the panel, but the protein was undetectable, notebook 03).
#
# 1. **Astrocyte reactivity: protein vs RNA.** Protein score from αSMA/Vim (vimentin in astrocytes) vs a
#    transcriptional reactivity score; where and when do they disagree?
# 2. **Leukocyte polarity, perivascular vs parenchymal** — magnitude and direction relative to the nearest vessel.
# 3. **18S rRNA per cell type, lesion vs physiological** — with and without adjusting for transcript density
#    (is it ribosome biology or RNA quality?).
# 4. **Neuropil-loss index** — ATP1A1 in each cell's extracellular territory relative to the same section's
#    physiological tissue of the same class (WM/GM): a tissue-damage readout the transcriptome cannot give.
#
# All per-animal statistics: one value per animal, Wilcoxon signed-rank across animals.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from skimage.segmentation import find_boundaries

from beyondboundaries import data, plotting
from beyondboundaries import orthogonality as orth
from beyondboundaries.io import XeniumBundle, find_bundles

plotting.style()
SRC = "notebooks/06_targeted_readouts.ipynb"
OUT = ROOT / "results" / "06_targeted_readouts"
OUT.mkdir(parents=True, exist_ok=True)
CH = data.CHANNELS
cfg = data.load_config()

fa = pd.read_parquet(ROOT / "data" / "features_norm.parquet")
fa = fa[fa.segmentation_method == "Segmented by interior stain (18S)"].copy()
fa["lesion"] = fa.Curated_niche_state.astype(str).str.startswith("Lesion")
fa["phys"] = fa.Curated_niche_state.astype(str) == "Physiological"
fa["lesion_dist_um"] = orth.lesion_distance(fa, fa.lesion.values)
fa["region_class"] = np.select([fa.Global_anatomical_region.isin(["WM", "WM_Meningeal"]),
                                fa.Global_anatomical_region.isin(["GM", "DorsalHorn", "VentralHorn"])], ["WM", "GM"], "other")
STAGE_ORDER = ["CFA", "NONSYMPTOM", "OS1", "MILD16", "PEAK1", "PEAK2", "PEAK3", "SEVERE16", "SEVERE30",
               "REMISSION1", "REMISSION2"]
adata = ad.read_h5ad(ROOT / cfg["annotation"]["h5ad"])


def section_z(s: pd.Series, by: pd.Series) -> pd.Series:
    """Robust z within section (median / MAD) — removes staining batch, keeps within-section biology."""
    med = s.groupby(by, observed=True).transform("median")
    mad = (s - med).abs().groupby(by, observed=True).transform("median") * 1.4826
    return (s - med) / mad.replace(0, np.nan)


def per_animal_paired(df, value, a_mask, b_mask, min_n=20):
    rows = []
    for an, g in df.groupby("sample_name", observed=True):
        A, B = g.loc[a_mask[g.index], value], g.loc[b_mask[g.index], value]
        if len(A) >= min_n and len(B) >= min_n:
            rows.append(dict(animal=an, a=A.median(), b=B.median()))
    d = pd.DataFrame(rows)
    if len(d) < 5:
        return d, np.nan
    return d, wilcoxon(d.a - d.b).pvalue

# %% [markdown]
# ## 1. Astrocyte reactivity — protein (vimentin channel) vs RNA

# %%
REACTIVE = ["C3", "Gfap", "Serpina3n", "Cd44", "Osmr", "Timp1", "Socs3", "Serping1", "H2-D1", "Psmb8", "Gbp2"]
HOMEO = ["Slc1a3", "Kcnj10", "Slc6a11", "Fgfr3", "Gjb6", "Aldoc", "Aldh1l1"]
ast = fa[fa.Anno_L1_curated == "Astrocyte"].copy()
Xln = orth.lognorm(adata[ast.index].X)
gi = {g: i for i, g in enumerate(adata.var_names)}


def gene_z(genes):
    M = Xln[:, [gi[g] for g in genes]].toarray()
    return ((M - M.mean(0)) / M.std(0)).mean(1)


ast["rna_reactive"] = gene_z(REACTIVE)
ast["rna_homeostatic"] = gene_z(HOMEO)
ast["rna_score"] = section_z(pd.Series(ast.rna_reactive - ast.rna_homeostatic, index=ast.index), ast.section_id)
# protein score: vimentin-channel intensity (cell mean, cytoplasmic p90) up, ATP1A1 rim down (spec)
ast["prot_score"] = (section_z(ast.smavim_cell_mean, ast.section_id) + section_z(ast.smavim_cyto_p90, ast.section_id)
                     - section_z(ast.bnd_rim_mean, ast.section_id)) / 3
r_all = spearmanr(ast.prot_score, ast.rna_score, nan_policy="omit").statistic
per_sec = ast.groupby("section_id").apply(lambda g: spearmanr(g.prot_score, g.rna_score, nan_policy="omit").statistic,
                                          include_groups=False)
print(f"protein vs RNA reactivity, Spearman ρ = {r_all:.3f} (per-section median {per_sec.median():.3f}, "
      f"range {per_sec.min():.2f}–{per_sec.max():.2f}); n = {len(ast):,}")
gene_rho = pd.Series({g: spearmanr(Xln[:, gi[g]].toarray().ravel(), ast.prot_score, nan_policy="omit").statistic
                      for g in REACTIVE + HOMEO}).sort_values()
gene_rho.round(3)

# %% [markdown]
# Quadrants (top/bottom 25 % within section of each score): concordant reactive, concordant quiet,
# **protein-only** (vimentin-high, RNA-quiet) and **RNA-only**. Where are they, and in which disease stages?

# %%
q = lambda s: s.groupby(ast.section_id).transform(lambda x: x.rank(pct=True))
pr, rr = q(ast.prot_score), q(ast.rna_score)
ast["quadrant"] = np.select([(pr > .75) & (rr > .75), (pr < .25) & (rr < .25), (pr > .75) & (rr < .25),
                             (pr < .25) & (rr > .75)],
                            ["both high", "both low", "protein-only", "RNA-only"], "middle")
QUADS = ["both high", "protein-only", "RNA-only", "both low"]
qt = ast[ast.quadrant != "middle"].groupby("quadrant").agg(
    n=("section_id", "size"), lesion_frac=("lesion", "mean"), WM_frac=("region_class", lambda s: (s == "WM").mean()),
    lesion_dist_median=("lesion_dist_um", "median"), edge_um_median=("edge_um", "median"),
    area_median=("morph_cell_area_um2", "median"), transcripts_median=("transcript_counts", "median")).loc[QUADS]
qt.to_csv(OUT / "astro_quadrants.csv")
qt.round(3)

# %%
st = ast[ast.quadrant != "middle"].groupby(["stage", "quadrant"], observed=True).size().unstack(fill_value=0)
st = st.div(st.sum(1), axis=0).reindex([s for s in STAGE_ORDER if s in st.index])[QUADS]
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
hb = axs[0].hexbin(ast.rna_score.clip(-4, 4), ast.prot_score.clip(-4, 4), gridsize=60, cmap=plotting.SEQ, bins="log")
axs[0].set(xlabel="RNA reactivity (within-section z)", ylabel="protein (vimentin-channel) score",
           title=f"astrocytes, ρ = {r_all:.2f}")
bottom = np.zeros(len(st))
for k, qd in enumerate(QUADS):
    axs[1].bar(st.index, st[qd], bottom=bottom, color=plotting.CATEGORICAL[k], label=qd, width=0.7)
    bottom += st[qd].values
axs[1].set_ylabel("share of extreme-quadrant astrocytes"); axs[1].tick_params(axis="x", rotation=45)
axs[1].legend(fontsize=7, ncol=2, loc="upper right")
fig.tight_layout()
plotting.save_fig(fig, "astro_protein_vs_rna", OUT, SRC)
st.round(3)

# %% [markdown]
# Per animal: protein-only share (of extreme-quadrant astrocytes) vs RNA-only share, by disease phase.
# Hypothesis: vimentin protein outlasts the reactive transcriptional programme → protein-only enriched in remission.

# %%
pa = ast[ast.quadrant != "middle"].groupby("sample_name", observed=True).agg(
    stage=("stage", "first"), score=("score_sacrifice", "first"), model=("model", "first"),
    protein_only=("quadrant", lambda s: (s == "protein-only").mean()), rna_only=("quadrant", lambda s: (s == "RNA-only").mean()),
    n=("quadrant", "size"))
pa = pa[pa.n >= 50]
pa["phase"] = np.select([pa.stage.astype(str).str.startswith("REMISSION"),
                         pa.stage.astype(str).str.contains("PEAK|SEVERE|MILD|OS1")], ["remission", "active"], "pre/none")
pa["protein_only_minus_rna_only"] = pa.protein_only - pa.rna_only
print(pa.groupby("phase")[["protein_only", "rna_only", "protein_only_minus_rna_only"]].median().round(3))
from scipy.stats import mannwhitneyu
a_, r_ = pa.loc[pa.phase == "active", "protein_only_minus_rna_only"], pa.loc[pa.phase == "remission", "protein_only_minus_rna_only"]
print(f"remission vs active (protein-only − RNA-only): MWU p = {mannwhitneyu(r_, a_).pvalue:.3g} (n={len(r_)} vs {len(a_)})")
pa.to_csv(OUT / "astro_quadrants_per_animal.csv")

# %% [markdown]
# Gallery: protein-only vs RNA-only astrocytes (one section), same contrast.

# %%
bundles = find_bundles(cfg)
GAL = ast[ast.quadrant.isin(["protein-only", "RNA-only", "both high"])].section_id.value_counts().index[0]
b = XeniumBundle(bundles[GAL])
half = int(15 / 0.2125)
lo = [0.0] * len(CH)
raw = lambda d, ch, st_: d[f"{ch}_{st_}"] * d[f"{ch}_scale"] + d[f"{ch}_bg_local"]
hi = [np.percentile(raw(ast, ch, "cell_p99").dropna(), 99) for ch in CH]
fig, axs = plt.subplots(3, 6, figsize=(9, 5))
for r, qd in enumerate(["both high", "protein-only", "RNA-only"]):
    cells = ast[(ast.section_id == GAL) & (ast.quadrant == qd)]
    cells = cells.sample(min(6, len(cells)), random_state=0)
    for c, (cid, row) in enumerate(cells.iterrows()):
        y, x = int(row.centroid_y_px), int(row.centroid_x_px)
        img, cm, nm = b.read_window(y - half, x - half, 2 * half, 2 * half)
        rgb = plotting.composite(img, CH, lo, hi)
        rgb[find_boundaries(cm == cm[half, half], mode="inner")] = 1
        axs[r, c].imshow(rgb); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    axs[r, 0].set_ylabel(qd, fontsize=8)
fig.suptitle(f"Astrocytes, {GAL}: composite DAPI/ATP1A1/18S/αSMA-Vim (magenta = vimentin channel)", fontsize=9)
fig.tight_layout()
plotting.save_fig(fig, "astro_quadrant_gallery", OUT, SRC)

# %% [markdown]
# ## 2. Leukocyte polarity: perivascular vs parenchymal
# Vessel cells = endothelial + VSMC centroids. Perivascular: nearest vessel cell < 15 µm; parenchymal: > 50 µm.
# Magnitude: `*_polarity_shape` (intensity-weighted vs geometric centroid, ÷ radius) and nucleus offset.
# Direction: cosine between the polarity vector and the vector to the nearest vessel cell (> 0 = towards the vessel).

# %%
vessel = fa.Anno_L1_curated.isin(["Endothelial", "VSMC"])
LEUK = {"MDM": fa.Anno_L2 == "MDM", "Microglia": fa.Anno_L2 == "Microglia", "T cell": fa.Anno_L1_curated == "T cell",
        "DC": fa.Anno_L1_curated == "DC", "B cell": fa.Anno_L1_curated == "B cell"}
fa["vessel_dist_um"], fa["vx"], fa["vy"] = np.nan, np.nan, np.nan
for sec, idx in fa.groupby("section_id", observed=True).indices.items():
    vi = idx[vessel.values[idx]]
    if len(vi) == 0:
        continue
    vxy = fa[["x_centroid", "y_centroid"]].values[vi]
    xy = fa[["x_centroid", "y_centroid"]].values[idx]
    d, j = cKDTree(vxy).query(xy)
    fa.iloc[idx, fa.columns.get_loc("vessel_dist_um")] = d
    fa.iloc[idx, fa.columns.get_loc("vx")] = vxy[j, 0] - xy[:, 0]
    fa.iloc[idx, fa.columns.get_loc("vy")] = vxy[j, 1] - xy[:, 1]
rows = []
for name, m in LEUK.items():
    d = fa[m & (fa.vessel_dist_um > 0)]
    peri, par = d.vessel_dist_um < 15, d.vessel_dist_um > 50
    for ch in CH:
        cosang = (d[f"{ch}_polarity_dx_um"] * d.vx + d[f"{ch}_polarity_dy_um"] * d.vy) / (
            np.hypot(d[f"{ch}_polarity_dx_um"], d[f"{ch}_polarity_dy_um"]) * np.hypot(d.vx, d.vy))
        d = d.assign(cos=cosang)
        pm, p_mag = per_animal_paired(d, f"{ch}_polarity_shape", peri, par)
        cos_an = d[peri].groupby("sample_name", observed=True).cos.agg(["mean", "size"])
        cos_an = cos_an[cos_an["size"] >= 20]["mean"]
        rows.append(dict(cell=name, channel=ch, n_peri=int(peri.sum()), n_par=int(par.sum()),
                         polarity_peri=pm.a.median() if len(pm) else np.nan, polarity_par=pm.b.median() if len(pm) else np.nan,
                         p_magnitude=p_mag, animals=len(pm),
                         cos_towards_vessel_peri=cos_an.median() if len(cos_an) else np.nan,
                         p_direction=wilcoxon(cos_an).pvalue if len(cos_an) >= 5 else np.nan))
pol = pd.DataFrame(rows)
pol.to_csv(OUT / "leukocyte_polarity.csv", index=False)
pol.round(4)

# %% [markdown]
# ## 3. 18S per cell type: lesion vs physiological niche
# Per animal, median normalised cytoplasmic 18S in lesion vs physiological cells of the same `Anno_L2` type.
# Adjusted version: 18S residual after regressing on log transcript density (transcripts / µm²) within type — if the
# lesion drop disappears, 18S is tracking RNA quality / capture, not ribosome biology specifically.

# %%
fa["tx_density"] = np.log(fa.transcript_counts / fa.morph_cell_area_um2)
rows = []
for t, d in fa.groupby("Anno_L2", observed=True):
    if len(d) < 1000:
        continue
    d = d.copy()
    v = section_z(d.r18s_cyto_mean, d.section_id)
    ok = v.notna()
    beta = np.polyfit(d.tx_density[ok], v[ok], 1)
    d["r18s_z"], d["r18s_adj"] = v, v - np.polyval(beta, d.tx_density)
    d["txd_z"] = section_z(d.tx_density, d.section_id)
    out = {"Anno_L2": t, "n": len(d)}
    for col in ("r18s_z", "r18s_adj", "txd_z"):
        pm, p = per_animal_paired(d, col, d.lesion, d.phys)
        out[f"{col} lesion−phys"] = (pm.a - pm.b).median() if len(pm) else np.nan
        out[f"{col} p"] = p
        out["animals"] = len(pm)
    out["rho r18s~txdensity"] = spearmanr(d.r18s_cyto_mean, d.tx_density, nan_policy="omit").statistic
    rows.append(out)
r18 = pd.DataFrame(rows).dropna(subset=["r18s_z p"]).sort_values("r18s_z lesion−phys")
r18.to_csv(OUT / "r18s_lesion_by_type.csv", index=False)
r18.round(3)

# %%
fig, ax = plt.subplots(figsize=(8, 0.32 * len(r18) + 1.2))
y = np.arange(len(r18))
ax.scatter(r18["r18s_z lesion−phys"], y, color=plotting.CATEGORICAL[2], label="18S (within-section z)", zorder=3)
ax.scatter(r18["r18s_adj lesion−phys"], y, color=plotting.CATEGORICAL[6], label="18S adjusted for transcript density", zorder=3)
ax.scatter(r18["txd_z lesion−phys"], y, color="#999999", marker="|", s=80, label="transcript density", zorder=2)
ax.axvline(0, color="#888888", lw=0.8)
ax.set_yticks(y, [f"{t} ({a})" for t, a in zip(r18.Anno_L2, r18.animals)], fontsize=7)
ax.set_xlabel("lesion − physiological (median over animals, z units)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
plotting.save_fig(fig, "r18s_lesion_by_type", OUT, SRC)

# %% [markdown]
# ## 4. Neuropil-loss index
# Raw ATP1A1-channel intensity in each cell's 10 µm extracellular territory ÷ the median of physiological-niche
# cells of the same section and region class (WM / GM). 1 = like healthy tissue of that class; < 1 = less neuropil
# ATP1A1 (loss of neuronal/axonal membrane, oedema, infiltrate).

# %%
fa["bnd_terr_raw"] = fa.bnd_terr_mean * fa.bnd_scale + fa.bnd_bg_local
key = fa.section_id.astype(str) + "|" + fa.region_class
ref = fa[fa.phys & fa.region_class.isin(["WM", "GM"])].groupby(key[fa.phys & fa.region_class.isin(["WM", "GM"])]).bnd_terr_raw.median()
fa["neuropil_index"] = fa.bnd_terr_raw / key.map(ref)
ni = fa[fa.region_class.isin(["WM", "GM"])]
by_niche = ni.groupby(["region_class", "Curated_niche_state"], observed=True).neuropil_index.agg(["median", "size"])
by_niche.round(3)

# %%
pa_n = ni[ni.lesion].groupby("sample_name", observed=True).agg(neuropil=("neuropil_index", "median"),
                                                              score=("score_sacrifice", "first"), n=("section_id", "size"))
pa_n = pa_n[pa_n.n >= 200]
rho, p = spearmanr(pa_n.neuropil, pa_n.score)
print(f"per animal: median neuropil index in lesion niches vs clinical score ρ = {rho:.2f}, p = {p:.3g} (n = {len(pa_n)})")
pa_n.to_csv(OUT / "neuropil_index_per_animal.csv")
# does the transcriptome of the cell explain it? (cell type + niche only)
gl = ni.groupby("Global_niche_group", observed=True).neuropil_index.agg(["median", "size"]).sort_values("median")
gl[gl["size"] >= 500].round(3)

# %%
EX = fa[fa.lesion].section_id.value_counts().index[0]
d = fa[fa.section_id == EX]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
sc_ = axs[0].scatter(d.x_centroid, -d.y_centroid, c=d.neuropil_index.clip(0, 1.5), s=0.5, cmap=plotting.DIV.reversed(),
                     vmin=0, vmax=2, rasterized=True)
axs[0].set_title(f"neuropil index, {EX}"); axs[0].set_aspect("equal"); axs[0].axis("off")
fig.colorbar(sc_, ax=axs[0], shrink=0.6, label="territory ATP1A1 ÷ healthy same-class")
cats = d.Curated_niche_state.astype(str)
for k, c in enumerate(sorted(cats.unique())):
    m = cats == c
    axs[1].scatter(d.x_centroid[m], -d.y_centroid[m], s=0.5, color=plotting.CATEGORICAL[k % 8], label=c, rasterized=True)
axs[1].set_title("Curated_niche_state"); axs[1].set_aspect("equal"); axs[1].axis("off")
axs[1].legend(markerscale=10, fontsize=7, loc="upper right")
fig.tight_layout()
plotting.save_fig(fig, "neuropil_index_map", OUT, SRC)

# %%
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(pa_n.score, pa_n.neuropil, s=20, color=plotting.CATEGORICAL[1])
ax.set(xlabel="clinical score at sacrifice", ylabel="median neuropil index\nin lesion niches",
       title=f"per animal, ρ = {rho:.2f} (p = {p:.2g})")
fig.tight_layout()
plotting.save_fig(fig, "neuropil_index_vs_score", OUT, SRC)
