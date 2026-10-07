"""Phase 4: how much of each image feature does the transcriptome explain, and vice versa?

Within one cell type, cross-validated by animal (GroupKFold on `sample_name`):
  cov   : technical covariates -> image feature
  full  : covariates + transcriptome PCs -> image feature
  tx    : transcriptome PCs only
ΔR²(tx | cov) = max(R²_full, 0) − max(R²_cov, 0) is what the transcriptome adds (clipped: with animal-grouped CV
the covariate model can score < 0 because cell-state composition differs between animals); 1 − R²_full is image
signal nothing here explains.
Reverse: covariates (+ image features) -> transcriptome PCs and individual genes. All R² are pooled out-of-fold.
Transcriptome PCA used as a *predictor* is fitted on the training folds only; the PCs used as reverse *targets* are
one PCA per cell type (unsupervised, so consistent targets across folds). Image features are rank-inverse-normal transformed within
the cell type (monotone, label-free) so heavy tails don't dominate R².
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ALPHAS = np.logspace(-2, 4, 13)


def lognorm(X, target: float = 100.0):
    X = sparse.csr_matrix(X, dtype=np.float32)
    tot = np.asarray(X.sum(1)).ravel()
    X = sparse.diags((target / np.maximum(tot, 1)).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def rank_int(Y: pd.DataFrame) -> pd.DataFrame:
    """Median-impute, then rank-based inverse normal transform per column."""
    Y = Y.fillna(Y.median())
    n = len(Y)
    return Y.apply(lambda c: pd.Series(norm.ppf((rankdata(c) - 0.5) / n), index=c.index))


def covariates(df: pd.DataFrame, with_area: bool = True) -> np.ndarray:
    """Section + spinal level one-hot, log transcripts, log area, log edge distance (standardised numerics)."""
    parts = [pd.get_dummies(df.section_id.astype(str), prefix="sec", drop_first=True),
             pd.get_dummies(df.region.astype(str), prefix="reg", drop_first=True)]
    num = pd.DataFrame({"log_tx": np.log1p(df.transcript_counts), "log_edge": np.log1p(df.edge_um.clip(lower=0))},
                       index=df.index)
    if with_area:
        num["log_area"] = np.log(df.morph_cell_area_um2)
    num = (num - num.mean()) / num.std()
    return pd.concat(parts + [num], axis=1).astype(np.float32).values


def r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def _gain(full, cov):
    """ΔR² with both terms floored at 0 (a model worse than the mean explains nothing, not negative variance)."""
    return np.clip(full, 0, None) - np.clip(cov, 0, None)


def _ridge(Xtr, Ytr, Xte):
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr, Ytr)
    return m.predict(Xte)


def run_celltype(df: pd.DataFrame, counts, feats: list[str], n_pcs: int = 50, n_rev_pcs: int = 20,
                 genes: np.ndarray | None = None, n_folds: int = 5, group: str = "sample_name") -> dict:
    """df: rows = cells of one type (features + metadata); counts: matching raw count matrix (cells x genes).
    feats: image feature columns. genes: column indices of genes to predict in the reverse direction."""
    Y = rank_int(df[feats]).values.astype(np.float32)
    morph = np.array([c.startswith("morph_") for c in feats])
    C_area, C_noarea = covariates(df, True), covariates(df, False)
    Xln = lognorm(counts)
    G = None
    if genes is not None and len(genes):
        G = Xln[:, genes].toarray()
        G = (G - G.mean(0)) / np.maximum(G.std(0), 1e-6)
    n = len(df)
    oof = {k: np.zeros_like(Y) for k in ("cov", "full", "tx")}
    T = PCA(n_rev_pcs, svd_solver="covariance_eigh", random_state=0).fit_transform(Xln)
    T = T / T.std(0)
    rev = {k: np.zeros_like(T) for k in ("cov", "full")}
    gen = {k: np.zeros((n, G.shape[1]), np.float32) for k in ("cov", "full")} if G is not None else None
    for tr, te in GroupKFold(n_folds).split(Y, groups=df[group].astype(str).values):
        pca = PCA(n_pcs, svd_solver="covariance_eigh", random_state=0).fit(Xln[tr])
        Ptr, Pte = pca.transform(Xln[tr]), pca.transform(Xln[te])
        s = Ptr.std(0)
        Ptr, Pte = Ptr / s, Pte / s
        for cols, C in ((~morph, C_area), (morph, C_noarea)):
            if not cols.any():
                continue
            oof["cov"][np.ix_(te, cols)] = _ridge(C[tr], Y[tr][:, cols], C[te])
            oof["full"][np.ix_(te, cols)] = _ridge(np.c_[C[tr], Ptr], Y[tr][:, cols], np.c_[C[te], Pte])
            oof["tx"][np.ix_(te, cols)] = _ridge(Ptr, Y[tr][:, cols], Pte)
        # reverse: image -> transcriptome PCs
        rev["cov"][te] = _ridge(C_area[tr], T[tr], C_area[te])
        rev["full"][te] = _ridge(np.c_[C_area[tr], Y[tr]], T[tr], np.c_[C_area[te], Y[te]])
        if G is not None:
            gen["cov"][te] = _ridge(C_area[tr], G[tr], C_area[te])
            gen["full"][te] = _ridge(np.c_[C_area[tr], Y[tr]], G[tr], np.c_[C_area[te], Y[te]])
    out = {
        "r2": pd.DataFrame({k: r2(Y, v) for k, v in oof.items()}, index=feats),
        "residuals": pd.DataFrame(Y - oof["full"], index=df.index, columns=feats),
    }
    out["r2"]["tx_unique"] = _gain(out["r2"]["full"], out["r2"]["cov"])
    out["reverse_pcs"] = pd.DataFrame({"cov": r2(T, rev["cov"]), "full": r2(T, rev["full"])},
                                      index=[f"PC{i + 1}" for i in range(n_rev_pcs)])
    out["reverse_pcs"]["img_unique"] = _gain(out["reverse_pcs"].full, out["reverse_pcs"]["cov"])
    if G is not None:
        out["reverse_genes"] = pd.DataFrame({"cov": r2(G, gen["cov"]), "full": r2(G, gen["full"])}, index=genes)
        out["reverse_genes"]["img_unique"] = _gain(out["reverse_genes"].full, out["reverse_genes"]["cov"])
    return out


def lesion_distance(df: pd.DataFrame, lesion: np.ndarray, x: str = "x_centroid", y: str = "y_centroid",
                    by: str = "section_id") -> np.ndarray:
    """Distance (µm) from each cell to the nearest lesion-niche cell in the same section (0 inside lesions)."""
    from scipy.spatial import cKDTree

    d = np.full(len(df), np.nan)
    for _, idx in df.groupby(by, observed=True).indices.items():
        les = idx[lesion[idx]]
        if len(les) == 0:
            continue
        tree = cKDTree(df[[x, y]].values[les])
        d[idx] = tree.query(df[[x, y]].values[idx])[0]
    return d


def nearest_of(df: pd.DataFrame, target: np.ndarray, x: str = "x_centroid", y: str = "y_centroid",
               by: str = "section_id") -> pd.DataFrame:
    """Per cell: distance (µm) and vector (dx, dy) to the nearest `target` cell in the same section (self excluded)."""
    from scipy.spatial import cKDTree

    out = np.full((len(df), 3), np.nan)
    xy_all = df[[x, y]].values
    for _, idx in df.groupby(by, observed=True).indices.items():
        ti = idx[target[idx]]
        if len(ti) < 2:
            continue
        d, j = cKDTree(xy_all[ti]).query(xy_all[idx], k=2)
        self_hit = ti[j[:, 0]] == idx
        jj = np.where(self_hit, j[:, 1], j[:, 0])
        dd = np.where(self_hit, d[:, 1], d[:, 0])
        out[idx, 0] = dd
        out[idx, 1:] = xy_all[ti[jj]] - xy_all[idx]
    return pd.DataFrame(out, index=df.index, columns=["dist_um", "dx_um", "dy_um"])
