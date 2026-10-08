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
# # 23 — Where RNA sits in and around cells in EAE lesions
#
# Cell-by-gene tables throw away *where* in the cell each transcript was. With the stain-based outlines and the nuclei
# we can ask what cells are doing with their RNA in lesions. Four questions, all runs 5/6 (25 animals, transcripts are
# only available there):
#
# 1. **Do oligodendrocytes in lesions still send myelin RNA out to their sheaths?** *Mbp* mRNA is transported out of
#    the cell body into the processes and translated where myelin is made; its sister myelin genes (*Mag*, *Mog*,
#    *Cldn11*, …) stay near the soma and serve as the control.
# 2. **What do macrophages and microglia carry inside them?** Myelin and axon transcripts deep inside a myeloid cell's
#    outline, relative to how much is around it, as a trace of engulfed material. Spillover from neighbours is the
#    alternative, so non-phagocytic cells in the same place are the baseline.
# 3. **Do lesion cells hold their RNA back in the nucleus?** Nuclear retention of mRNA is a stress response.
# 4. **What does the space between cells lose in white-matter lesions: myelin RNA or axonal RNA?** The RNA counterpart
#    of the ~17 % ATP1A1 neuropil loss (notebooks 20, 22).
#
# **Design.** Input: `scripts/04_transcript_compartments.py` (every transcript with qv >= 20: in a cell or not, in the
# nucleus or not, depth inside the cell, nearest cell and distance outside). Lesion zones from the lesion regions
# (notebook 16, signed distance to the region edge): **core** >= 30 µm inside, **edge** ±30 µm, **near** 30–150 µm
# outside, **healthy** > 150 µm outside or no region, and not called lesion. Counts are pooled per animal and zone (the
# animal is the unit); core vs healthy is a paired test across animals (Wilcoxon). Cross-sectional: these are
# associations.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from skimage.segmentation import find_boundaries
from statsmodels.stats.multitest import multipletests

from beyondboundaries import data, plotting
from beyondboundaries.io import XeniumBundle, find_bundles

plotting.style()
SRC = "notebooks/23_rna_in_and_around_cells.ipynb"
OUT = ROOT / "results" / "23_rna_in_and_around_cells"
OUT.mkdir(parents=True, exist_ok=True)
SUB = ROOT / "data" / "subcellular"
COL = plotting.CATEGORICAL
WM = ["WM", "WM_Meningeal"]
HALO = 10.0
ZONES = ["healthy", "near", "edge", "core"]
ZCOL = dict(zip(ZONES, ["#4c9a6a", "#9cc28a", "#e0a040", "#c0392b"]))
STAGES = ["CFA", "NONSYMPTOM", "OS1", "PEAK1", "REMISSION1", "PEAK2", "REMISSION2", "PEAK3", "MILD16", "SEVERE16",
          "SEVERE30"]
SETS = {
    "mbp": ["Mbp"],
    "myelin_ctrl": ["Mag", "Mog", "Cldn11", "Mal", "Opalin", "Enpp6"],
    "cnp": ["Cnp"],
    "oligo_soma": ["Sox10", "Olig1", "Olig2", "Nkx6-2", "Myrf", "St18"],
    "axon": ["Nefl", "Nefm", "Nefh", "Stmn2", "Gap43"],
    "neuron_soma": ["Snap25", "Camk2a", "Rbfox3", "Syt1"],
    "cd68": ["Cd68"],
    "phago": ["Trem2", "Lpl", "Cd36", "Gpnmb", "Axl"],
    "astro": ["Gfap", "Aqp4", "Slc1a2", "Slc1a3"],
    "stress": ["Neat1", "Hspa1a", "Hspa1b", "Ddit3", "Fos", "Jun", "Atf4", "Hsph1"],
}
MYELIN = ["mbp", "myelin_ctrl", "cnp"]

# %%
a = ad.read_h5ad(ROOT / "data" / "RRMAP2_all_runs.h5ad", backed="r")
obs = a.obs[a.obs.run_id.isin(["run5", "run6"])][
    ["sample_name", "meta_sample_id", "sample_id", "stage", "model", "Anno_L1_curated", "Curated_niche_state",
     "Global_anatomical_region", "x_centroid", "y_centroid", "cell_area", "nucleus_area"]].copy()
genes = a.var_names.copy()
del a
for c in ["sample_name", "meta_sample_id", "sample_id", "stage", "model", "Anno_L1_curated", "Curated_niche_state",
          "Global_anatomical_region"]:
    obs[c] = obs[c].astype(str)
obs["arm"] = np.where(obs.model.str.startswith("CHRONIC"), "chronic", "RR")
obs = obs.join(pd.read_parquet(ROOT / "data" / "lesion10" / "lesion_calls.parquet")[["lesion_state", "ctype"]])
obs = obs.join(pd.read_parquet(ROOT / "data" / "lesion16" / "regions.parquet")[["reg", "reg_dist"]])
rd = obs.reg_dist
obs["zone"] = np.select(
    [rd >= 30, rd >= -30, rd >= -150, ((rd < -150) | rd.isna()) & (obs.lesion_state == "no lesion")],
    ["core", "edge", "near", "healthy"], default="other")
obs["wm"] = obs.Global_anatomical_region.isin(WM)
clin = pd.read_csv(ROOT / "data" / "clinical" / "animal_course_metrics.csv", index_col=0)
print(obs.zone.value_counts().to_string())
print(pd.crosstab(obs.drop_duplicates("sample_name").stage, obs.drop_duplicates("sample_name").arm).to_string())

# %% [markdown]
# **Per-cell compartment counts.** For every annotated cell: transcripts in the cell and in its nucleus (all genes and
# the gene sets above); transcripts *outside* any cell whose nearest cell is this one, within 10 µm (its "halo"), with
# the free area that halo covers; and, for the gene sets, how many transcripts sit at the rim of the outline (< 1 µm
# from it) or deep inside (>= 2 µm). For the gene-level nuclear screen (question 3), counts are pooled per animal ×
# cell type × zone.

# %%
gi = pd.Series(np.arange(len(genes)), index=genes)
sidx = {k: gi.reindex(v).dropna().astype(int).to_numpy() for k, v in SETS.items()}
gene2set = {g: k for k, v in SETS.items() for g in v}
print({k: len(v) for k, v in sidx.items()})

parts, pool_keys, pool_tot, pool_nuc = [], [], [], []
for sid in sorted(obs.sample_id.unique()):
    z = np.load(SUB / f"{sid}.npz", allow_pickle=True)
    names = z["obs_names"]
    n, G = len(names), len(genes)
    tot = sp.csr_matrix((z["total_data"], z["total_indices"], z["total_indptr"]), shape=(n, G))
    nuc = sp.csr_matrix((z["nuclear_data"], z["nuclear_indices"], z["nuclear_indptr"]), shape=(n, G))
    d = pd.DataFrame(index=pd.Index(names, name="obs_id"))
    d["n_total"], d["n_nuc"] = tot.sum(1).A1, nuc.sum(1).A1
    for k, ix in sidx.items():
        d[f"{k}_cell"], d[f"{k}_nuc"] = tot[:, ix].sum(1).A1, nuc[:, ix].sum(1).A1
    d["halo_area"], d["halo_all"] = z["halo_area_um2"], z["halo_all"]
    p = pd.read_parquet(SUB / f"{sid}_points.parquet", columns=["gene", "in_cell", "cell", "depth_um", "near_cell",
                                                                  "dist_um"])
    p["set"] = p.gene.astype(str).map(gene2set)
    h = p[~p.in_cell & (p.dist_um <= HALO) & p.near_cell.notna()]
    d = d.join(h.groupby(["near_cell", "set"]).size().unstack(fill_value=0).add_prefix("halo_"))
    c = p[p.in_cell & p.cell.notna()]
    lay = np.where(c.depth_um < 1, "rim", np.where(c.depth_um >= 2, "deep", "mid"))
    dd = c.groupby([c.cell, c.set, lay]).size().unstack([1, 2], fill_value=0)
    dd.columns = [f"{l}_{s}" for s, l in dd.columns]
    parts.append(d.join(dd))
    # pooled counts per animal x cell type x zone (healthy / core), for the gene-level nuclear screen
    o = obs.loc[names]
    keep = o.zone.isin(["healthy", "core"]).to_numpy()
    key = (o.sample_name + "|" + o.Anno_L1_curated + "|" + o.zone).to_numpy()
    codes, uniq = pd.factorize(key[keep])
    ind = sp.csr_matrix((np.ones(len(codes)), (codes, np.flatnonzero(keep))), shape=(len(uniq), n))
    pool_keys += list(uniq)
    pool_tot.append((ind @ tot).toarray().astype(np.float32))
    pool_nuc.append((ind @ nuc).toarray().astype(np.float32))
cells = obs.join(pd.concat(parts).fillna(0), how="inner")
pool = pd.DataFrame({"key": pool_keys})
pool_tot, pool_nuc = np.vstack(pool_tot), np.vstack(pool_nuc)
print(f"{len(cells):,} cells; {cells.n_total.sum():,.0f} transcripts in cells; "
      f"{cells.halo_all.sum():,.0f} outside cells within {HALO:.0f} µm of one")


# %% [markdown]
# Helpers: a value per animal × zone from pooled counts (needs >= 20 cells), the core-vs-healthy contrast across
# animals, and a zone profile (each animal's value ÷ its own healthy value).

# %%
def per_zone(d, fn, by=("sample_name", "zone"), min_cells=20):
    g = d.groupby(list(by), observed=True)
    v = g.apply(fn)
    return v[g.size() >= min_cells]


def ratio(num, den):
    return lambda g: g[num].sum() / g[den].sum() if g[den].sum() > 0 else np.nan


def contrast(v, a="healthy", b="core"):
    w = v.unstack()
    if a not in w or b not in w:
        return dict(animals=0)
    lf = np.log2(w[b] / w[a]).replace([np.inf, -np.inf], np.nan).dropna()
    return dict(animals=len(lf), core_over_healthy=2 ** lf.median(), share_lower=(lf < 0).mean(),
                p=wilcoxon(lf).pvalue if len(lf) >= 5 else np.nan)


def profile(ax, v, label, color, ref="healthy"):
    w = v.unstack().reindex(columns=ZONES)
    w = w.div(w[ref], axis=0).replace([np.inf, -np.inf], np.nan)
    m, q1, q3 = w.median(), w.quantile(0.25), w.quantile(0.75)
    x = np.arange(len(ZONES))
    ax.plot(x, m, marker="o", color=color, label=f"{label} (n={w[ZONES[-1]].notna().sum()})")
    ax.fill_between(x, q1, q3, color=color, alpha=0.15)
    ax.set_xticks(x, ZONES)
    ax.axhline(1, color="0.6", lw=0.8, ls="--")


def report(rows):
    t = pd.DataFrame(rows)
    return t.round(3)


# %% [markdown]
# ## 0. Sanity: where each gene's RNA sits in healthy tissue
# Before looking at lesions: share of each focus gene's transcripts inside cell outlines, and of those, in the nucleus
# (healthy zone, all cells pooled). Expectations from cell biology: *Mbp* mostly outside cell bodies (transported into
# myelin), *Mag*/*Mog*/*Sox10* near the soma, neurofilaments outside (axons), *Neat1* nuclear.

# %%
hz = cells[cells.zone == "healthy"]
rows = []
for k in SETS:
    inc, out = hz[f"{k}_cell"].sum(), hz.get(f"halo_{k}", pd.Series(0)).sum()
    rows.append(dict(gene_set=k, genes=", ".join(SETS[k]), in_cells=inc / (inc + out),
                     nuclear_of_in_cell=hz[f"{k}_nuc"].sum() / max(inc, 1)))
san = pd.DataFrame(rows).set_index("gene_set")
san["nuclear_of_in_cell (all genes)"] = hz.n_nuc.sum() / hz.n_total.sum()
san.round(2).to_csv(OUT / "sanity_localisation_healthy.csv")
san.round(2)

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
s = san.sort_values("in_cells")
ax.barh(s.index, s.in_cells, color="#5b8bd0", label="inside cell outlines")
ax.barh(s.index, s.in_cells * s.nuclear_of_in_cell, color="#2c3e70", label="… of which in the nucleus")
ax.set_xlabel("share of transcripts (outside = within 10 µm of a cell, not in any outline)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("healthy tissue: where each gene set's RNA sits")
fig.tight_layout()
plotting.save_fig(fig, "sanity_localisation_healthy", OUT, SRC)

# %% [markdown]
# ## 1. Do oligodendrocytes in lesions still send *Mbp* out?
# White-matter oligodendrocytes (`Anno_L1_curated`). Three readouts per animal × zone:
# - **Mbp kept in the cell body**: *Mbp* in the outline ÷ control myelin genes in the outline. Rises if *Mbp* piles up
#   in the soma (transport failure), falls if the cell sends it all out or stops making it relative to the others.
# - ***Mbp* in the nucleus**: nuclear share of *Mbp* ÷ nuclear share of the control genes.
# - **Own *Mbp* halo**: *Mbp* density (per µm² of free space) within 10 µm of oligodendrocytes ÷ the same around other
#   white-matter cells of the zone (the ambient myelin RNA). Above 1 = the oligodendrocyte has its own exported *Mbp*
#   around it; this corrects for myelin simply being lost from the lesion.
#
# Oligodendrocyte state matters (disease-associated oligodendrocytes, DAO, express myelin genes differently), so the
# readouts are also split by state (`ctype`: MOL, NFOL, DAO).

# %%
wm = cells[cells.wm]
ol = wm[wm.Anno_L1_curated == "Oligodendrocyte"]
other = wm[~wm.Anno_L1_curated.isin(["Oligodendrocyte", "OPC"])]
r_soma = per_zone(ol, ratio("mbp_cell", "myelin_ctrl_cell"))
r_nuc = per_zone(ol, lambda g: (g.mbp_nuc.sum() / g.mbp_cell.sum()) / (g.myelin_ctrl_nuc.sum() / g.myelin_ctrl_cell.sum()))
r_halo = per_zone(ol, ratio("halo_mbp", "halo_area")) / per_zone(other, ratio("halo_mbp", "halo_area"))
r_amb = per_zone(other, ratio("halo_mbp", "halo_area"))
r_ctrl = per_zone(ol, ratio("myelin_ctrl_cell", "n_total"))
q1 = report([dict(readout="Mbp ÷ control myelin genes in the soma", **contrast(r_soma)),
             dict(readout="Mbp nuclear share ÷ control nuclear share", **contrast(r_nuc)),
             dict(readout="own Mbp halo ÷ ambient Mbp (other WM cells)", **contrast(r_halo.dropna())),
             dict(readout="ambient Mbp density around other WM cells", **contrast(r_amb)),
             dict(readout="control myelin genes per transcript in oligos", **contrast(r_ctrl))])
q1.to_csv(OUT / "q1_mbp_core_vs_healthy.csv", index=False)
q1

# %%
rows = []
for st in ["MOL", "NFOL", "DAO"]:
    s = ol[ol.ctype == st]
    rows.append(dict(state=st, cells=len(s), readout="Mbp ÷ control in soma",
                     **contrast(per_zone(s, ratio("mbp_cell", "myelin_ctrl_cell")))))
    rows.append(dict(state=st, cells=len(s), readout="own Mbp halo ÷ ambient",
                     **contrast((per_zone(s, ratio("halo_mbp", "halo_area")) / r_amb).dropna())))
q1s = report(rows)
q1s["zone_mix"] = q1s.state.map(lambda st: ", ".join(f"{z} {v:.0%}" for z, v in
                                                     ol[ol.ctype == st].zone.value_counts(normalize=True)[ZONES].items()))
q1s.to_csv(OUT / "q1_by_oligo_state.csv", index=False)
q1s

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
profile(axs[0], r_soma, "Mbp ÷ control, soma", "#c0392b")
profile(axs[0], r_ctrl, "control myelin genes per transcript", "0.4")
axs[0].set_title("is Mbp kept in the oligodendrocyte soma?")
profile(axs[1], r_nuc, "Mbp nuclear ÷ control nuclear", "#8e44ad")
axs[1].set_title("is Mbp held in the nucleus?")
profile(axs[2], r_halo, "own Mbp halo ÷ ambient", "#2471a3")
profile(axs[2], r_amb, "ambient Mbp (other WM cells)", "0.4")
axs[2].set_title("Mbp exported around the oligodendrocyte")
for ax in axs:
    ax.set_ylabel("÷ the animal's healthy WM (median, IQR)")
    ax.legend(fontsize=7)
fig.suptitle("white-matter oligodendrocytes across lesion zones (animals = unit)", fontsize=10)
fig.tight_layout()
plotting.save_fig(fig, "q1_mbp_zone_profiles", OUT, SRC)

# %% [markdown]
# **Seeing it.** Healthy vs lesion-core white matter of the same piece (the animal with the most core-zone
# oligodendrocytes): ATP1A1/CD45/E-cad channel (grey), cell outlines (thin), oligodendrocytes outlined in yellow, *Mbp*
# transcripts inside an outline (red) and outside (cyan), control myelin genes (yellow dots).

# %%
cfg = data.load_config()
cfg["runs"] = {k: v for k, v in cfg["runs"].items() if k in ("run5", "run6")}
bundles = find_bundles(cfg)


def crop_points(sid, x0, y0, w):
    t = pq.read_table(SUB / f"{sid}_points.parquet", columns=["x", "y", "gene", "in_cell"],
                      filters=[("x", ">=", x0), ("x", "<", x0 + w), ("y", ">=", y0), ("y", "<", y0 + w)])
    return t.to_pandas()


def show_crop(ax, row, w_um, layers, cell_hl=None, hl_color="#f1c40f", title=""):
    b = XeniumBundle(bundles[row.sample_id])
    px = b.pixel_size
    x0, y0 = row.x_centroid - w_um / 2, row.y_centroid - w_um / 2
    img, lab, _ = b.read_window(int(y0 / px), int(x0 / px), int(w_um / px), int(w_um / px))
    bnd = img[1]
    ax.imshow(np.clip(bnd / np.percentile(bnd, 99.5), 0, 1), cmap="gray", extent=(0, w_um, w_um, 0))
    edges = find_boundaries(lab, mode="inner")
    ov = np.zeros((*lab.shape, 4))
    ov[edges] = (0.55, 0.55, 0.55, 0.5)
    if cell_hl is not None:
        ids = b.cells().set_index("cell_id").label
        hl = ids.reindex(cell_hl).dropna().astype(int).to_numpy()
        ov[edges & np.isin(lab, hl)] = (*plt.matplotlib.colors.to_rgb(hl_color), 1)
    ax.imshow(ov, extent=(0, w_um, w_um, 0))
    p = crop_points(row.sample_id, x0, y0, w_um)
    for genes_, inside, color, sz in layers:
        q = p[p.gene.astype(str).isin(genes_)]
        if inside is not None:
            q = q[q.in_cell == inside]
        ax.scatter(q.x - x0, q.y - y0, s=sz, c=color, lw=0, alpha=0.8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


cnt = ol[ol.zone == "core"].groupby("meta_sample_id").size()
pc = cnt.idxmax()
g = ol[ol.meta_sample_id == pc]
rng = np.random.default_rng(0)
picks = [("healthy WM", g[g.zone == "healthy"]), ("lesion core WM", g[g.zone == "core"])]
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
W = 80
for ax, (lab_, s) in zip(axs, picks):
    r = s.iloc[rng.integers(len(s))]
    near = g[(g.x_centroid - r.x_centroid).abs().lt(W / 2) & (g.y_centroid - r.y_centroid).abs().lt(W / 2)]
    show_crop(ax, r, W, [(["Mbp"], False, "#3fd0f0", 3), (["Mbp"], True, "#e74c3c", 5),
                         (SETS["myelin_ctrl"], None, "#f1c40f", 6)],
              cell_hl=near.index.str.split(":").str[1], title=f"{pc} ({g.stage.iloc[0]}) · {lab_} · {W} µm")
fig.suptitle("Mbp inside oligodendrocyte outlines (red) vs outside (cyan); control myelin genes (yellow dots); "
             "oligodendrocytes outlined yellow", fontsize=9)
fig.tight_layout()
plotting.save_fig(fig, "q1_mbp_crops", OUT, SRC)

# %% [markdown]
# ## 2. What do macrophages and microglia carry inside them?
# Myeloid cells (`ctype`: MDM, microglia, CAM) against non-phagocytic cells in the same zone (astrocytes, T cells,
# endothelial cells). Two readouts:
# - **Uptake ratio**: myelin transcripts per µm² inside the outline ÷ myelin transcripts per µm² of free space in the
#   cell's own halo. Spillover from surrounding RNA gives similar ratios for all cell types of a similar size; a cell
#   that has taken up myelin material has a higher one.
# - **Depth**: share of the myelin transcripts in a cell that sit deep (>= 2 µm from the outline) rather than at the rim
#   (< 1 µm), ÷ the same share for the cell's own marker RNA (phagocyte genes for myeloid cells, astrocyte genes for
#   astrocytes). Spillover sits at the rim (ratio well below 1); engulfed material spreads like the cell's own RNA
#   (ratio near 1).
#
# Plus a dose test within myeloid cells: uptake by *Cd68* level (lysosomal/phagocytic marker), tertiles within each
# animal × zone. The same is done for axonal RNA (neurofilaments, *Stmn2*, *Gap43*).

# %%
cells["myelin_cell"] = cells[[f"{k}_cell" for k in MYELIN]].sum(1)
cells["halo_myelin"] = cells[[f"halo_{k}" for k in MYELIN if f"halo_{k}" in cells]].sum(1)
for lay in ["rim", "deep"]:
    cells[f"{lay}_myelin"] = cells[[f"{lay}_{k}" for k in MYELIN if f"{lay}_{k}" in cells]].sum(1)
GROUPS = {"myeloid": ["MDM", "Microglia", "CAM"], "astrocyte": ["Reactive astro", "Homeostatic astro"],
          "T cell": ["T cell"], "endothelial": ["Endothelial"]}
cells["grp"] = None
for k, v in GROUPS.items():
    cells.loc[cells.ctype.isin(v), "grp"] = k


def uptake(mat):
    return lambda g: ((g[f"{mat}_cell"].sum() / g.cell_area.sum()) / (g[f"halo_{mat}"].sum() / g.halo_area.sum())
                      if g[f"halo_{mat}"].sum() > 0 else np.nan)


def depth_index(mat, own):
    def f(g):
        dm = g[f"deep_{mat}"].sum() / max(g[f"deep_{mat}"].sum() + g[f"rim_{mat}"].sum(), 1)
        do = g[f"deep_{own}"].sum() / max(g[f"deep_{own}"].sum() + g[f"rim_{own}"].sum(), 1)
        return dm / do if do > 0 else np.nan
    return f


up = {(grp, mat): per_zone(cells[cells.grp == grp], uptake(mat), by=("sample_name", "zone"))
      for grp in GROUPS for mat in ["myelin", "axon"]}
rows = []
for (grp, mat), v in up.items():
    w = v.unstack()
    for zn in ["healthy", "core"]:
        if zn in w:
            rows.append(dict(material=mat, cells=grp, zone=zn, animals=w[zn].notna().sum(),
                             uptake_median=w[zn].median()))
q2 = pd.DataFrame(rows).pivot_table(index=["material", "zone"], columns="cells", values="uptake_median").round(3)
q2.to_csv(OUT / "q2_uptake_by_celltype.csv")
q2

# %% [markdown]
# Paired across animals within each zone: myeloid uptake ÷ astrocyte (and ÷ T cell) uptake, and the core-vs-healthy
# change of the myeloid ratio.

# %%
rows = []
for mat in ["myelin", "axon"]:
    for ref in ["astrocyte", "T cell", "endothelial"]:
        rel = (up[("myeloid", mat)] / up[(ref, mat)]).replace([np.inf, -np.inf], np.nan).dropna()
        w = rel.unstack()
        for zn in ["healthy", "core"]:
            if zn in w:
                lf = np.log2(w[zn].dropna())
                rows.append(dict(material=mat, vs=ref, zone=zn, animals=len(lf), myeloid_over_ref=2 ** lf.median(),
                                 share_above=(lf > 0).mean(), p=wilcoxon(lf).pvalue if len(lf) >= 5 else np.nan))
        rows.append(dict(material=mat, vs=ref, zone="core ÷ healthy", **{
            k: v for k, v in contrast(rel).items() if k != "share_lower"}))
q2b = report(rows)
q2b.to_csv(OUT / "q2_myeloid_vs_reference.csv", index=False)
q2b

# %%
my = cells[cells.grp == "myeloid"].copy()
my["cd68_level"] = my.cd68_cell / my.n_total.clip(lower=1)
my["cd68_tertile"] = my.groupby(["sample_name", "zone"]).cd68_level.transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]) if len(s) >= 30 else np.nan)
rows = []
for zn in ["healthy", "edge", "core"]:
    s = my[my.zone == zn]
    for mat in ["myelin", "axon"]:
        v = per_zone(s, uptake(mat), by=("sample_name", "cd68_tertile"), min_cells=10).unstack()
        if {"low", "high"} <= set(v.columns):
            lf = np.log2(v.high / v.low).replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(dict(zone=zn, material=mat, animals=len(lf), high_over_low_Cd68=2 ** lf.median(),
                             share_above=(lf > 0).mean(), p=wilcoxon(lf).pvalue if len(lf) >= 5 else np.nan))
q2c = report(rows)
q2c.to_csv(OUT / "q2_uptake_by_cd68.csv", index=False)
q2c

# %%
di = {"myeloid": per_zone(cells[cells.grp == "myeloid"], depth_index("myelin", "phago")),
      "astrocyte": per_zone(cells[cells.grp == "astrocyte"], depth_index("myelin", "astro"))}
rows = [dict(cells=k, zone=zn, animals=v.unstack()[zn].notna().sum(), depth_index_median=v.unstack()[zn].median())
        for k, v in di.items() for zn in ["healthy", "edge", "core"] if zn in v.unstack()]
q2d = report(rows)
q2d.to_csv(OUT / "q2_depth_index.csv", index=False)
q2d

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
for ax, mat in zip(axs[:2], ["myelin", "axon"]):
    for k, colr in zip(GROUPS, COL):
        w = up[(k, mat)].unstack().reindex(columns=ZONES)
        x = np.arange(len(ZONES))
        ax.plot(x, w.median(), marker="o", color=colr, label=k)
        ax.fill_between(x, w.quantile(0.25), w.quantile(0.75), color=colr, alpha=0.12)
    ax.set_xticks(np.arange(len(ZONES)), ZONES)
    ax.set_yscale("log")
    ax.set_ylabel(f"{mat} RNA per µm² inside ÷ around")
    ax.set_title(f"uptake ratio, {mat} RNA (animals: median, IQR)")
    ax.legend(fontsize=7)
for k, colr in zip(di, COL):
    w = di[k].unstack().reindex(columns=ZONES)
    axs[2].plot(np.arange(len(ZONES)), w.median(), marker="o", color=colr, label=k)
axs[2].set_xticks(np.arange(len(ZONES)), ZONES)
axs[2].axhline(1, color="0.6", ls="--", lw=0.8)
axs[2].set_ylabel("deep share myelin ÷ deep share own RNA")
axs[2].set_title("does myelin RNA sit deep, like the cell's own RNA?")
axs[2].legend(fontsize=7)
fig.tight_layout()
plotting.save_fig(fig, "q2_uptake_profiles", OUT, SRC)

# %% [markdown]
# **Seeing it.** Myeloid cells in lesion cores with the highest myelin uptake ratio (>= 20 myelin transcripts; one per
# animal, top 4 animals): ATP1A1/CD45/E-cad channel, the myeloid cell outlined in yellow, myelin transcripts inside
# (red) and outside (cyan) its outline, phagocyte genes (green).

# %%
cand = my[(my.zone == "core") & (my.myelin_cell >= 20)].copy()
cand["u"] = (cand.myelin_cell / cand.cell_area) / ((cand.halo_myelin + 1) / cand.halo_area.clip(lower=1))
top = cand.sort_values("u", ascending=False).drop_duplicates("sample_name").head(4)
fig, axs = plt.subplots(1, len(top), figsize=(4.2 * len(top), 4.4))
for ax, (_, r) in zip(np.atleast_1d(axs), top.iterrows()):
    myel = [g for k in MYELIN for g in SETS[k]]
    show_crop(ax, r, 30, [(myel, False, "#3fd0f0", 6), (myel, True, "#e74c3c", 9), (SETS["phago"] + ["Cd68"], None,
                                                                                    "#2ecc71", 9)],
              cell_hl=[r.name.split(":")[1]], title=f"{r.sample_name} ({r.stage}) · {r.ctype}\n"
                                                    f"{int(r.myelin_cell)} myelin transcripts inside")
fig.suptitle("lesion-core myeloid cells with the most myelin RNA inside relative to around (30 µm crops)", fontsize=9)
fig.tight_layout()
plotting.save_fig(fig, "q2_myeloid_crops", OUT, SRC)

# %% [markdown]
# ## 3. Do lesion cells hold their RNA back in the nucleus?
# The nuclear share of a cell's transcripts depends on how much of the outline the nucleus covers. So per animal ×
# zone × cell type: **nuclear share of transcripts ÷ nuclear share of area** (pooled). Above 1 = RNA concentrated in
# the nucleus beyond its footprint. Core vs healthy across animals, for each major cell type; then which genes shift
# most (relative nuclear share: a gene's nuclear share ÷ the nuclear share of all transcripts in the same cells, so
# shape changes cancel).

# %%
TYPES = ["Oligodendrocyte", "Myeloid", "Astrocyte", "Neuron", "Endothelial", "Fibroblast", "OPC", "T cell"]
rows, nucz = [], {}
for t in TYPES:
    s = cells[cells.Anno_L1_curated == t]
    v = per_zone(s, lambda g: (g.n_nuc.sum() / g.n_total.sum()) / (g.nucleus_area.sum() / g.cell_area.sum()))
    ar = per_zone(s, ratio("nucleus_area", "cell_area"))
    nucz[t] = v
    rows.append(dict(cell_type=t, readout="nuclear RNA share ÷ nuclear area share", **contrast(v)))
    rows.append(dict(cell_type=t, readout="nuclear area share (geometry)", **contrast(ar)))
q3 = report(rows)
q3["fdr"] = np.nan
m = q3.p.notna()
q3.loc[m, "fdr"] = multipletests(q3.loc[m, "p"], method="fdr_bh")[1].round(4)
q3.to_csv(OUT / "q3_nuclear_retention.csv", index=False)
q3

# %%
pk = pool.key.str.split("|", expand=True)
pk.columns = ["animal", "type", "zone"]
rows = []
for t in TYPES:
    res = {}
    for zn in ["healthy", "core"]:
        m = ((pk.type == t) & (pk.zone == zn)).to_numpy()
        T, N = pool_tot[m], pool_nuc[m]
        rel = (N / np.where(T > 0, T, np.nan)) / (N.sum(1, keepdims=True) / T.sum(1, keepdims=True))
        rel[T < 30] = np.nan
        res[zn] = pd.DataFrame(rel, index=pk.animal[m].to_numpy(), columns=genes)
    an = res["healthy"].index.intersection(res["core"].index)
    if len(an) < 6:
        continue
    lf = np.log2(res["core"].loc[an] / res["healthy"].loc[an])
    ok = lf.notna().sum() >= 6
    for g in genes[ok.to_numpy()]:
        x = lf[g].dropna()
        x = x[np.isfinite(x)]
        if len(x) >= 6:
            rows.append(dict(cell_type=t, gene=g, animals=len(x), core_over_healthy=2 ** x.median(),
                             share_higher=(x > 0).mean(), p=wilcoxon(x).pvalue))
scr = pd.DataFrame(rows)
scr["fdr"] = scr.groupby("cell_type").p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
scr.to_csv(OUT / "q3_gene_screen.csv", index=False)
print(scr.groupby("cell_type").apply(lambda d: pd.Series(dict(genes=len(d), fdr05=(d.fdr < 0.05).sum(),
                                                             up=((d.fdr < 0.05) & (d.core_over_healthy > 1)).sum()))))
scr[scr.fdr < 0.05].sort_values("p").groupby("cell_type").head(8).round(4)

# %%
scr[scr.gene.isin(SETS["stress"])].sort_values(["gene", "cell_type"]).round(3)

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for t, colr in zip(TYPES, COL):
    profile(axs[0], nucz[t], t, colr)
axs[0].set_ylabel("nuclear RNA ÷ nuclear area share,\n÷ the animal's healthy value")
axs[0].set_title("nuclear RNA concentration across lesion zones")
axs[0].legend(fontsize=6, ncol=2)
for i, t in enumerate(TYPES):
    d = scr[scr.cell_type == t]
    if len(d):
        axs[1].scatter(np.log2(d.core_over_healthy), -np.log10(d.p), s=4, color=COL[i], alpha=0.5, label=t)
axs[1].set_xlabel("log2 relative nuclear share, core ÷ healthy (median over animals)")
axs[1].set_ylabel("-log10 p")
axs[1].set_title("which genes are held in the nucleus in lesion cores?")
axs[1].legend(fontsize=6, markerscale=3)
fig.tight_layout()
plotting.save_fig(fig, "q3_nuclear_retention", OUT, SRC)

# %% [markdown]
# **Does it follow disease severity?** Per animal, nuclear retention in the core ÷ healthy (log2), against the clinical
# score at sacrifice and the first-attack peak (Spearman; exploratory, n = animals with lesion cores).

# %%
rows = []
for t in TYPES:
    w = nucz[t].unstack()
    if "core" not in w:
        continue
    lf = np.log2(w.core / w.healthy).dropna().rename("lf").to_frame().join(clin[["score", "first_peak", "auc"]])
    for c in ["score", "first_peak", "auc"]:
        x = lf[["lf", c]].dropna()
        if len(x) >= 8:
            r = spearmanr(x.lf, x[c])
            rows.append(dict(cell_type=t, clinical=c, animals=len(x), rho=r.statistic, p=r.pvalue))
q3c = report(rows)
q3c.to_csv(OUT / "q3_vs_clinical.csv", index=False)
q3c

# %% [markdown]
# ## 4. What does the space between cells lose in white-matter lesions?
# RNA outside cell outlines in white matter, per µm² of free space within 10 µm of a cell: myelin RNA (*Mbp* and the
# other myelin genes), axonal RNA (neurofilaments, *Stmn2*, *Gap43*), astrocyte RNA (*Gfap*, *Aqp4*, …) and all genes.
# **Local comparison as in notebook 20:** within each piece, core-zone WM cells against healthy WM cells within 150 µm
# of them, so tract anatomy can't drive the difference. Next to it, the ATP1A1 neuropil index of the same cells
# (notebook 20 definition), so protein and RNA loss can be compared piece by piece.

# %%
fa = pd.read_parquet(ROOT / "data" / "features_norm_all.parquet",
                     columns=["segmentation_method", "bnd_terr_mean", "bnd_scale", "bnd_bg_local"])
fa = fa[fa.segmentation_method == "Segmented by interior stain (18S)"]
raw = fa.bnd_terr_mean * fa.bnd_scale + fa.bnd_bg_local
w4 = cells[cells.wm].join(raw.rename("atp_raw"))
w4["atp_index"] = w4.atp_raw / w4.groupby("meta_sample_id").atp_raw.transform("median")
w4["halo_myelin"] = w4[[f"halo_{k}" for k in MYELIN]].sum(1)
MATS = {"myelin RNA": "halo_myelin", "axonal RNA": "halo_axon", "astrocyte RNA": "halo_astro", "all RNA": "halo_all"}
rows = []
for pc, g in w4.groupby("meta_sample_id"):
    L, H = g[g.zone == "core"], g[g.zone == "healthy"]
    if len(L) < 30 or len(H) < 30:
        continue
    dist, _ = cKDTree(L[["x_centroid", "y_centroid"]].to_numpy()).query(H[["x_centroid", "y_centroid"]].to_numpy())
    Hn = H[dist <= 150]
    if len(Hn) < 30:
        continue
    r = dict(piece=pc, animal=g.sample_name.iloc[0], stage=g.stage.iloc[0], arm=g.arm.iloc[0], core_cells=len(L),
             healthy_cells=len(Hn), ATP1A1=L.atp_index.median() / Hn.atp_index.median())
    for lab_, c in MATS.items():
        r[lab_] = (L[c].sum() / L.halo_area.sum()) / (Hn[c].sum() / Hn.halo_area.sum())
    r["myelin share of outside RNA"] = (L.halo_myelin.sum() / L.halo_all.sum()) / (Hn.halo_myelin.sum() / Hn.halo_all.sum())
    r["axonal share of outside RNA"] = (L.halo_axon.sum() / L.halo_all.sum()) / (Hn.halo_axon.sum() / Hn.halo_all.sum())
    rows.append(r)
q4p = pd.DataFrame(rows)
q4p.to_csv(OUT / "q4_pieces_core_vs_local_healthy.csv", index=False)
vals = ["ATP1A1"] + list(MATS) + ["myelin share of outside RNA", "axonal share of outside RNA"]
an = q4p.groupby("animal")[vals].apply(lambda d: np.exp(np.log(d).mean()))
q4 = pd.DataFrame([dict(readout=v, pieces=q4p[v].notna().sum(), animals=an[v].notna().sum(),
                        core_over_local_healthy=np.exp(np.log(an[v]).median()), share_lower=(an[v] < 1).mean(),
                        p=wilcoxon(np.log(an[v].dropna())).pvalue) for v in vals]).round(4)
q4.to_csv(OUT / "q4_summary.csv", index=False)
q4

# %%
cor = pd.DataFrame([dict(x="ATP1A1", y=v, **dict(zip(["rho", "p"], spearmanr(np.log(q4p.ATP1A1), np.log(q4p[v]))))) for v in
                    list(MATS)]).round(3)
cor.to_csv(OUT / "q4_atp1a1_vs_rna_pieces.csv", index=False)
cor

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
vv = ["ATP1A1", "myelin RNA", "axonal RNA", "astrocyte RNA", "all RNA"]
lg = np.log2(an[vv])
axs[0].boxplot([lg[v].dropna() for v in vv], showfliers=False)
for i, v in enumerate(vv):
    axs[0].scatter(np.full(lg[v].notna().sum(), i + 1) + rng.normal(0, 0.05, lg[v].notna().sum()), lg[v].dropna(),
                   s=12, color="0.3", zorder=3)
axs[0].set_xticks(range(1, len(vv) + 1), vv, rotation=20)
axs[0].axhline(0, color="0.6", ls="--", lw=0.8)
axs[0].set_ylabel("log2 lesion core ÷ nearby healthy WM")
axs[0].set_title("per animal: what is lost from the space between cells")
for ax, v in zip(axs[1:], ["myelin RNA", "axonal RNA"]):
    ax.scatter(np.log2(q4p.ATP1A1), np.log2(q4p[v]), s=18, c=[COL[0] if a_ == "chronic" else COL[1] for a_ in q4p.arm])
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.axvline(0, color="0.6", ls="--", lw=0.8)
    r = cor.set_index("y").loc[v]
    ax.set_xlabel("log2 ATP1A1 neuropil, core ÷ local healthy")
    ax.set_ylabel(f"log2 {v} outside cells, core ÷ local healthy")
    ax.set_title(f"pieces: ρ = {r.rho:+.2f} (p = {r.p:.2g}); blue chronic, orange RR", fontsize=9)
fig.tight_layout()
plotting.save_fig(fig, "q4_neuropil_rna", OUT, SRC)

# %% [markdown]
# **Seeing it.** The same healthy and lesion-core WM fields as in section 1, now with myelin RNA (cyan) and axonal RNA
# (magenta) outside cell outlines.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
rng = np.random.default_rng(0)
for ax, (lab_, s) in zip(axs, picks):
    r = s.iloc[rng.integers(len(s))]
    show_crop(ax, r, W, [([g for k in MYELIN for g in SETS[k]], False, "#3fd0f0", 3),
                         (SETS["axon"], False, "#ff4fd8", 7)], title=f"{pc} · {lab_} · {W} µm")
fig.suptitle("RNA outside cell outlines: myelin genes (cyan) and axonal genes (magenta)", fontsize=9)
fig.tight_layout()
plotting.save_fig(fig, "q4_neuropil_crops", OUT, SRC)

# %% [markdown]
# ## 5. Across the disease course
# Per animal (core ÷ healthy, log2) for the main readout of each question, by stage and against clinical score
# (exploratory: few animals per stage).

# %%
summ = pd.DataFrame({
    "Mbp kept in soma (Q1)": np.log2(r_soma.unstack().core / r_soma.unstack().healthy),
    "own Mbp halo (Q1)": np.log2(r_halo.unstack().core / r_halo.unstack().healthy),
    "myeloid myelin uptake ÷ astro (Q2)": np.log2((up[("myeloid", "myelin")] / up[("astrocyte", "myelin")]).unstack().core),
    "oligo nuclear retention (Q3)": np.log2(nucz["Oligodendrocyte"].unstack().core / nucz["Oligodendrocyte"].unstack().healthy),
    "myeloid nuclear retention (Q3)": np.log2(nucz["Myeloid"].unstack().core / nucz["Myeloid"].unstack().healthy),
    "WM myelin RNA outside cells (Q4)": np.log2(an["myelin RNA"]),
    "WM axonal RNA outside cells (Q4)": np.log2(an["axonal RNA"]),
}).replace([np.inf, -np.inf], np.nan)
st = obs.drop_duplicates("sample_name").set_index("sample_name")[["stage", "arm"]]
summ = summ.join(st).join(clin[["score", "first_peak"]])
summ.round(3).to_csv(OUT / "per_animal_summary.csv")
rows = []
for c in summ.columns[:7]:
    for k in ["score", "first_peak"]:
        x = summ[[c, k]].dropna()
        if len(x) >= 8:
            r = spearmanr(x[c], x[k])
            rows.append(dict(readout=c, clinical=k, animals=len(x), rho=r.statistic, p=r.pvalue))
q5 = report(rows)
q5.to_csv(OUT / "course_vs_clinical.csv", index=False)
q5

# %%
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, c in zip(axs, ["Mbp kept in soma (Q1)", "myeloid myelin uptake ÷ astro (Q2)", "oligo nuclear retention (Q3)",
                       "WM myelin RNA outside cells (Q4)"]):
    d = summ[[c, "stage", "arm"]].dropna()
    order = [s_ for s_ in STAGES if s_ in set(d.stage)]
    x = d.stage.map({s_: i for i, s_ in enumerate(order)})
    ax.scatter(x + rng.normal(0, 0.06, len(d)), d[c], c=[COL[0] if a_ == "chronic" else COL[1] for a_ in d.arm], s=22)
    ax.set_xticks(range(len(order)), order, rotation=45, fontsize=7)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.set_title(c, fontsize=9)
axs[0].set_ylabel("log2 core ÷ healthy (per animal)")
fig.suptitle("per animal across the disease course (blue chronic, orange RR)", fontsize=10)
fig.tight_layout()
plotting.save_fig(fig, "course", OUT, SRC)

# %% [markdown]
# ## Findings
# (filled in after the run)
