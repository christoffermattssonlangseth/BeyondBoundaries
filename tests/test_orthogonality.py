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
