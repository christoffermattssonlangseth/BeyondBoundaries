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
# # 04 — What do the segmentation-stain images add to the transcriptome?
#
# Within each cell type (18S-segmented cells; ≥ 500 cells, ≥ 5 animals; max 30,000 cells per type),
# cross-validated **by animal** (5-fold GroupKFold on `sample_name`):
#
# 1. **Image ← transcriptome.** For each image feature: R² from technical covariates alone (section, spinal level,
#    log transcripts, log area, log edge distance), R² from covariates + 50 transcriptome PCs, and the difference
#    (what the transcriptome adds). `1 − R²_full` = image signal not explained here.
# 2. **Transcriptome ← image.** Transcriptome PCs and individual genes predicted from covariates ± image features.
# 3. **Is the unexplained part signal or noise?** Spatial coherence of residuals among same-type neighbours.
# 4. **Does the unexplained part carry disease information?** Per-feature residual lesion-vs-physiological shift
#    (per animal), lesion classification with transcriptome vs transcriptome + image, residual clusters.
#
# Caveats carried from earlier phases: ATP1A1 channel has no detectable CD45; 18S drew these masks, so 18S
# *distribution* features (radial, rim) are partly by construction.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from beyondboundaries import data, plotting
from beyondboundaries import orthogonality as orth

plotting.style()
SRC = "notebooks/04_orthogonality.ipynb"
OUT = ROOT / "results" / "04_orthogonality"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "data" / "residuals"
RES.mkdir(parents=True, exist_ok=True)
CH = data.CHANNELS
MIN_CELLS, MIN_ANIMALS, MAX_CELLS = 500, 5, 30000
rng = np.random.default_rng(0)


def bh(p):
    """Benjamini–Hochberg q-values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out

# %%
fa = pd.read_parquet(ROOT / "data" / "features_norm.parquet")
fa["lesion"] = fa.Curated_niche_state.astype(str).str.startswith("Lesion")
fa["phys"] = fa.Curated_niche_state.astype(str) == "Physiological"
fa["lesion_dist_um"] = orth.lesion_distance(fa, fa.lesion.values)
for ch in CH:   # raw-ratio membrane/context features (scale- and background-free)
    fa[f"{ch}_rim_over_ring"] = (fa[f"{ch}_rim_mean"] + fa[f"{ch}_bg_local"] / fa[f"{ch}_scale"]) / \
                                (fa[f"{ch}_ring_mean"] + fa[f"{ch}_bg_local"] / fa[f"{ch}_scale"])
f = fa[(fa.segmentation_method == "Segmented by interior stain (18S)")
       & ~fa.Anno_L1_curated.isin(["Doublet", "T_B_doublet"])].copy()

FEATS = []
for ch in CH:
    FEATS += [f"{ch}_{c}_{s}" for c in ("nuc", "cyto", "rim", "ring", "terr") for s in ("mean", "p90")]
    FEATS += [f"{ch}_radial_b{k}" for k in range(5)]
    FEATS += [f"{ch}_polarity_shape", f"{ch}_polarity_nuc", f"{ch}_rim_over_ring"]
    FEATS += [f"{ch}_glcm_{k}" for k in ("contrast", "homogeneity", "asm", "entropy", "correlation")]
FEATS += [c for c in f.columns if c.startswith("morph_") and c != "morph_cell_orientation"] + ["nuc_offset"]
FAM = data.feature_families(FEATS).reindex(FEATS).fillna("intensity: rim/ring ratio")
FAM[[c for c in FEATS if c.endswith("_rim_over_ring")]] = "membrane ratio"
CHAN = pd.Series({c: (c.split("_")[0] if c.split("_")[0] in CH else "morph") for c in FEATS})

n_by = f.groupby("Anno_L1_curated", observed=True).agg(n=("section_id", "size"), animals=("sample_name", "nunique"))
TYPES = n_by[(n_by.n >= MIN_CELLS) & (n_by.animals >= MIN_ANIMALS)].sort_values("n", ascending=False).index.tolist()
print(len(FEATS), "image features;", len(TYPES), "cell types:", TYPES)

# %%
cfg = data.load_config()
adata = ad.read_h5ad(ROOT / cfg["annotation"]["h5ad"])
gene_names = adata.var_names.values

# %% [markdown]
# ## Fit per cell type (results cached in `data/residuals/`)

# %%
results = {}
for t in TYPES:
    cache = RES / f"{t.replace('/', '_').replace(' ', '_')}.pkl"
    if cache.exists():
        results[t] = pd.read_pickle(cache)
        continue
    d = f[f.Anno_L1_curated == t]
    if len(d) > MAX_CELLS:
        d = d.loc[rng.choice(d.index.values, MAX_CELLS, replace=False)]
    feats = [c for c in FEATS if d[c].isna().mean() < 0.05]
    X = adata[d.index].X
    Xln = orth.lognorm(X)
    det = np.asarray((X > 0).mean(0)).ravel()
    var = np.asarray(Xln.power(2).mean(0)).ravel() - np.asarray(Xln.mean(0)).ravel() ** 2
    genes = np.where(det >= 0.05)[0]
    genes = genes[np.argsort(-var[genes])[:500]]
    r = orth.run_celltype(d, X, feats, genes=genes)
    r["reverse_genes"].index = gene_names[r["reverse_genes"].index]
    r["n"] = len(d)
    pd.to_pickle(r, cache)
    results[t] = r
    print(f"{t}: n={len(d)}, median R²_cov={r['r2']['cov'].median():.3f}, "
          f"median ΔR²_tx={r['r2']['tx_unique'].median():.3f}", flush=True)

# %% [markdown]
# ## 1. How much of each image feature does the transcriptome explain?

# %%
r2 = pd.concat({t: r["r2"] for t, r in results.items()}, names=["cell_type", "feature"]).reset_index()
r2["family"] = r2.feature.map(FAM)
r2["channel"] = r2.feature.map(CHAN)
r2.to_csv(OUT / "r2_image_from_transcriptome.csv", index=False)
summ = r2.groupby("cell_type")[["cov", "full", "tx_unique"]].median().loc[TYPES]
summ["unexplained"] = 1 - summ.full
summ.round(3)

# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ax, val, title, vmax in ((axs[0], "tx_unique", "ΔR² transcriptome | covariates (median per family)", 0.3),
                             (axs[1], "full", "R² covariates + transcriptome (median per family)", 0.6)):
    piv = r2.pivot_table(index="cell_type", columns=["channel", "family"], values=val, aggfunc="median").loc[TYPES]
    im = ax.imshow(piv.values, cmap=plotting.SEQ, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{a} · {b}" for a, b in piv.columns], rotation=75, ha="right", fontsize=6)
    ax.set_yticks(range(len(TYPES)), TYPES, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.6)
fig.tight_layout()
plotting.save_fig(fig, "r2_heatmaps", OUT, SRC)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for k, (fam_, g) in enumerate(r2.groupby("family")):
    ax.scatter(g["cov"], g["full"], s=6, alpha=0.6, color=plotting.CATEGORICAL[k % 8], label=fam_)
ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--")
ax.set(xlabel="R² covariates only", ylabel="R² covariates + transcriptome", xlim=(-0.05, 1), ylim=(-0.05, 1))
ax.legend(fontsize=6, ncol=2, loc="lower right")
fig.tight_layout()
plotting.save_fig(fig, "r2_cov_vs_full", OUT, SRC)

# %% [markdown]
# Features the transcriptome explains *most* (largest ΔR²), per cell type — these are the image readouts that
# largely re-measure transcriptional state:

# %%
top_tx = r2.sort_values("tx_unique", ascending=False).groupby("cell_type").head(5)
top_tx[["cell_type", "feature", "cov", "full", "tx_unique"]].round(3).set_index("cell_type").loc[TYPES]

# %% [markdown]
# ## 2. Reverse: how much of the transcriptome do the images predict?

# %%
rev = pd.concat({t: r["reverse_pcs"] for t, r in results.items()}, names=["cell_type", "pc"]).reset_index()
rev.to_csv(OUT / "r2_transcriptome_pcs_from_image.csv", index=False)
pv = rev.pivot(index="cell_type", columns="pc", values="img_unique").loc[TYPES][[f"PC{i}" for i in range(1, 21)]]
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(pv.values, cmap=plotting.SEQ, vmin=0, vmax=0.5, aspect="auto")
ax.set_xticks(range(20), pv.columns, fontsize=7); ax.set_yticks(range(len(TYPES)), TYPES, fontsize=8)
fig.colorbar(im, ax=ax, label="ΔR² image | covariates", shrink=0.7)
ax.set_title("Transcriptome PCs (within type) predicted by image features beyond covariates")
fig.tight_layout()
plotting.save_fig(fig, "reverse_pcs", OUT, SRC)

genes_tab = pd.concat({t: r["reverse_genes"] for t, r in results.items()}, names=["cell_type", "gene"]).reset_index()
genes_tab.to_csv(OUT / "r2_genes_from_image.csv", index=False)
top_g = genes_tab.sort_values("img_unique", ascending=False).groupby("cell_type").head(8)
top_g.round(3).set_index("cell_type").loc[TYPES]

# %% [markdown]
# ## 3. Is the unexplained image signal structured or noise?
# Spatial coherence: correlation between a cell's residual and the mean residual of its 10 nearest same-type
# neighbours in the same section. Noise → ≈ 0. Tissue state (or local technical artefact) → > 0.

# %%
coh = {}
for t, r in results.items():
    res = r["residuals"]
    meta = f.loc[res.index, ["section_id", "x_centroid", "y_centroid"]]
    nb_mean = np.full(res.shape, np.nan)
    for _, idx in meta.groupby("section_id", observed=True).indices.items():
        if len(idx) < 30:
            continue
        xy = meta.iloc[idx][["x_centroid", "y_centroid"]].values
        _, nn = cKDTree(xy).query(xy, k=11)
        nb_mean[idx] = res.values[idx][nn[:, 1:]].mean(1)
    ok = np.isfinite(nb_mean[:, 0])
    coh[t] = pd.Series([np.corrcoef(res.values[ok, j], nb_mean[ok, j])[0, 1] for j in range(res.shape[1])],
                       index=res.columns)
coh = pd.DataFrame(coh)
coh.to_csv(OUT / "residual_spatial_coherence.csv")
coh_f = coh.groupby(coh.index.map(CHAN) + " · " + coh.index.map(FAM)).median()[TYPES]
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(coh_f.values, cmap=plotting.SEQ, vmin=0, vmax=0.6, aspect="auto")
ax.set_yticks(range(len(coh_f)), coh_f.index, fontsize=6); ax.set_xticks(range(len(TYPES)), TYPES, rotation=60, ha="right")
fig.colorbar(im, ax=ax, label="residual vs neighbour-residual r", shrink=0.6)
fig.tight_layout()
plotting.save_fig(fig, "residual_spatial_coherence", OUT, SRC)

# %% [markdown]
# ## 4a. Does the unexplained image signal differ between lesion and physiological niches?
# Per animal: mean residual in lesion-niche cells − mean residual in physiological-niche cells (both ≥ 20 cells);
# Wilcoxon signed-rank across animals; BH over all type × feature tests. Effect in residual SD units.

# %%
rows = []
for t, r in results.items():
    res = r["residuals"]
    m = f.loc[res.index, ["sample_name", "lesion", "phys"]]
    for an, g in m.groupby("sample_name", observed=True):
        L, P = g.index[g.lesion], g.index[g.phys]
        if len(L) >= 20 and len(P) >= 20:
            diff = res.loc[L].mean() - res.loc[P].mean()
            rows.append(pd.DataFrame({"cell_type": t, "animal": an, "feature": diff.index, "diff": diff.values}))
dl = pd.concat(rows)
tests = []
for (t, feat), g in dl.groupby(["cell_type", "feature"]):
    if len(g) >= 5:
        tests.append(dict(cell_type=t, feature=feat, n_animals=len(g), median_diff=g["diff"].median(),
                          frac_pos=(g["diff"] > 0).mean(), p=wilcoxon(g["diff"]).pvalue))
lt = pd.DataFrame(tests)
lt["q"] = bh(lt.p)
lt["family"] = lt.feature.map(FAM)
lt["coherence"] = [coh.loc[ft, ct] for ct, ft in zip(lt.cell_type, lt.feature)]
lt.to_csv(OUT / "residual_lesion_shift.csv", index=False)
print(f"{(lt.q < 0.05).sum()} / {len(lt)} type × feature residuals shift with lesion (q < 0.05)")
lt[lt.q < 0.05].sort_values("median_diff", key=abs, ascending=False).head(30).round(3)

# %%
piv = lt.assign(v=np.where(lt.q < 0.05, lt.median_diff, np.nan)).pivot(index="feature", columns="cell_type", values="v")
piv = piv.reindex(columns=TYPES).dropna(how="all")
piv = piv.loc[piv.abs().max(1).sort_values(ascending=False).index[:40]]
fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(piv.values, cmap=plotting.DIV, vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_yticks(range(len(piv)), piv.index, fontsize=7); ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=60, ha="right")
fig.colorbar(im, ax=ax, label="lesion − physiological residual (SD), q<0.05", shrink=0.5)
fig.tight_layout()
plotting.save_fig(fig, "residual_lesion_shift_top40", OUT, SRC)

# %% [markdown]
# ## 4b. Does adding image features improve lesion-niche classification beyond the transcriptome?
# Within each type, lesion vs physiological cells; L2 logistic regression; GroupKFold by animal; inputs
# (i) covariates + 50 transcriptome PCs, (ii) + image features. Note lesion niches were defined from transcriptomic
# neighbourhoods, so the transcriptome has a head start by construction.

# %%
cls = []
for t, r in results.items():
    idx = r["residuals"].index
    d = f.loc[idx]
    keep = (d.lesion | d.phys).values
    if keep.sum() < 300 or d.lesion[keep].mean() < 0.05 or d.lesion[keep].mean() > 0.95:
        continue
    d = d[keep]
    y = d.lesion.values
    C = orth.covariates(d, True)
    Xln = orth.lognorm(adata[d.index].X)
    I = orth.rank_int(d[r["r2"].index]).values
    pred = {k: np.zeros(len(d)) for k in ("tx", "tx+img", "img")}
    for tr, te in GroupKFold(5).split(C, groups=d.sample_name.astype(str).values):
        pca = PCA(50, svd_solver="covariance_eigh", random_state=0).fit(Xln[tr])
        P = pca.transform(Xln)
        P = P / P[tr].std(0)
        for k, Z in (("tx", np.c_[C, P]), ("tx+img", np.c_[C, P, I]), ("img", np.c_[C, I])):
            m = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tr], y[tr])
            pred[k][te] = m.predict_proba(Z[te])[:, 1]
    cls.append(dict(cell_type=t, n=len(d), lesion_frac=y.mean(), **{f"AUROC {k}": roc_auc_score(y, v) for k, v in pred.items()}))
cls = pd.DataFrame(cls).set_index("cell_type")
cls["ΔAUROC (img | tx)"] = cls["AUROC tx+img"] - cls["AUROC tx"]
cls.to_csv(OUT / "lesion_classification.csv")
cls.round(3)

# %% [markdown]
# ## 4c. Residual clusters
# Leiden on the PCA of residuals (per type). For each cluster: share in lesion vs physiological niche (per animal),
# median distance to lesion, defining image features (mean residual z) and top marker genes (to see whether the
# cluster is transcriptionally distinct after all, i.e. a non-linear transcriptome effect).

# %%
clus_rows, clus_feats, clus_genes = [], {}, {}
for t, r in results.items():
    res = r["residuals"]
    a_ = ad.AnnData(res.values.astype(np.float32), obs=f.loc[res.index, ["sample_name", "lesion", "phys",
                                                                         "lesion_dist_um", "section_id"]])
    sc.pp.pca(a_, n_comps=min(15, res.shape[1] - 1))
    sc.pp.neighbors(a_, n_neighbors=15)
    sc.tl.leiden(a_, resolution=0.3, flavor="igraph", n_iterations=2, key_added="rc")
    lab = a_.obs.rc.astype(str)
    clus_feats[t] = res.groupby(lab.values).mean()
    for k in sorted(lab.unique(), key=int):
        mk = (lab == k).values
        if mk.sum() < 100:
            continue
        per = a_.obs.assign(k=mk).groupby("sample_name", observed=True).apply(
            lambda g: pd.Series({"L": g.k[g.lesion].mean() if g.lesion.sum() >= 20 else np.nan,
                                 "P": g.k[g.phys].mean() if g.phys.sum() >= 20 else np.nan}), include_groups=False).dropna()
        p = wilcoxon(per.L - per.P).pvalue if len(per) >= 5 else np.nan
        top = clus_feats[t].loc[k].sort_values(key=abs, ascending=False).head(4)
        clus_rows.append(dict(cell_type=t, cluster=k, n=int(mk.sum()), frac=mk.mean(),
                              lesion_minus_phys=(per.L - per.P).median() if len(per) else np.nan, p=p,
                              lesion_dist_median=np.nanmedian(a_.obs.lesion_dist_um[mk]),
                              type_lesion_dist_median=np.nanmedian(a_.obs.lesion_dist_um),
                              top_features=", ".join(f"{i} {v:+.2f}" for i, v in top.items())))
    ag = adata[res.index].copy()
    sc.pp.normalize_total(ag, target_sum=100); sc.pp.log1p(ag)
    ag.obs["rc"] = lab.values
    big = lab.value_counts()
    ag = ag[ag.obs.rc.isin(big[big >= 100].index)].copy()
    if ag.obs.rc.nunique() > 1:
        sc.tl.rank_genes_groups(ag, "rc", method="wilcoxon", n_genes=8)
        clus_genes[t] = sc.get.rank_genes_groups_df(ag, None)
ct = pd.DataFrame(clus_rows)
ct["q"] = bh(ct.p.fillna(1))
ct.to_csv(OUT / "residual_clusters.csv", index=False)
pd.concat(clus_genes, names=["cell_type"]).to_csv(OUT / "residual_cluster_markers.csv")
ct.sort_values("q").round(3).head(30)

# %% [markdown]
# Marker genes of residual clusters with a lesion association (q < 0.05) — large log-fold changes here mean the
# "unexplained" image signal still tracks a transcriptional state the linear model missed.

# %%
for _, row in ct[ct.q < 0.05].sort_values("q").head(12).iterrows():
    g = clus_genes.get(row.cell_type)
    if g is None:
        continue
    gg = g[g.group == row.cluster].head(6)
    print(f"{row.cell_type} c{row.cluster} (n={row.n}, lesion−phys {row.lesion_minus_phys:+.3f}): "
          + ", ".join(f"{a} ({b:.1f})" for a, b in zip(gg.names, gg.logfoldchanges)))
