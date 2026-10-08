"""Synthetic check: a feature driven by gene expression is explained by the transcriptome; a noise feature is not."""
import numpy as np
import pandas as pd
from scipy import sparse

from beyondboundaries.orthogonality import run_celltype


def test_explained_vs_orthogonal():
    rng = np.random.default_rng(0)
    n, g = 1500, 60
    state = rng.normal(size=n)                                   # latent cell state
    lam = np.exp(1 + 0.8 * np.outer(state, rng.normal(size=g)).clip(-3, 3))
    counts = sparse.csr_matrix(rng.poisson(lam))
    df = pd.DataFrame({
        "img_tx": state + 0.3 * rng.normal(size=n),             # transcriptome-explained
        "img_free": rng.normal(size=n),                          # orthogonal
        "morph_cell_area_um2": rng.uniform(40, 80, n),
        "section_id": rng.choice(["s1", "s2"], n), "region": "L",
        "transcript_counts": np.asarray(counts.sum(1)).ravel(), "edge_um": rng.uniform(0, 500, n),
        "sample_name": rng.choice([f"a{i}" for i in range(8)], n),
    })
    out = run_celltype(df, counts, ["img_tx", "img_free"], n_pcs=10, n_rev_pcs=3, genes=np.arange(5))
    r = out["r2"]
    assert r.loc["img_tx", "tx_unique"] > 0.6
    assert abs(r.loc["img_free", "full"]) < 0.05
    assert out["residuals"].shape == (n, 2)
    assert out["reverse_pcs"].loc["PC1", "img_unique"] > 0.3   # image feature predicts the state PC


def test_nonlinear_arm_finds_nonlinear_dependence():
    """A feature driven nonlinearly (|state|) by the transcriptome: the tree arm explains it, ridge cannot."""
    from beyondboundaries.orthogonality import run_celltype_nonlinear

    rng = np.random.default_rng(1)
    n, g = 1500, 60
    state = rng.normal(size=n)
    lam = np.exp(1 + 0.8 * np.outer(state, rng.normal(size=g)).clip(-3, 3))
    counts = sparse.csr_matrix(rng.poisson(lam))
    df = pd.DataFrame({
        "img_nonlin": np.abs(state) + 0.2 * rng.normal(size=n),
        "img_free": rng.normal(size=n),
        "morph_cell_area_um2": rng.uniform(40, 80, n),
        "section_id": rng.choice(["s1", "s2"], n), "region": "L",
        "transcript_counts": np.asarray(counts.sum(1)).ravel(), "edge_um": rng.uniform(0, 500, n),
        "sample_name": rng.choice([f"a{i}" for i in range(8)], n),
    })
    nl = run_celltype_nonlinear(df, counts, ["img_nonlin", "img_free"], n_pcs=10, n_jobs=1)["r2"]
    lin = run_celltype(df, counts, ["img_nonlin", "img_free"], n_pcs=10, n_rev_pcs=3)["r2"]
    # log-count PCs and the transcript-count covariate already capture part of |state| linearly (ridge ΔR² ~0.3), so
    # the margin is modest; on this seed the tree arm reaches ~0.43 vs ~0.32
    assert nl.loc["img_nonlin", "tx_unique"] > lin.loc["img_nonlin", "tx_unique"] + 0.05
    assert nl.loc["img_free", "tx_unique"] < 0.05
