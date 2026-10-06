"""Unit tests on small synthetic images with known values."""
import numpy as np
import pandas as pd
import pytest

from beyondboundaries.extract import plan_windows, run_window
from beyondboundaries.features import FeatureParams, glcm_features, grouped_stats, window_features
from beyondboundaries.io import decode_cell_id

CH = ["a", "b"]
P = FeatureParams(pixel_size=0.5, rim_um=1.0, ring_um=1.0, territory_um=3.0, radial_bins=5,
                  glcm_levels=8, glcm_distance_um=0.5)


def disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


@pytest.fixture
def two_cells():
    """Cell 1: disc r=12 at (25,25), nucleus r=5; cell 2: disc r=10 at (25,70), no nucleus.
    ch a: 100 in nucleus 1, 20 in cytoplasm 1, 50 in cell 2, 5 outside cells.
    ch b: left half of cell 1 = 0, right half = 40 (polarised to +x); 0 elsewhere."""
    shape = (50, 100)
    cell = np.zeros(shape, np.uint32)
    cell[disc(shape, 25, 25, 12)] = 1
    cell[disc(shape, 25, 70, 10)] = 2
    nuc = np.zeros(shape, np.uint32)
    nuc[disc(shape, 25, 25, 5)] = 1
    a = np.full(shape, 5.0)
    a[cell == 1] = 20
    a[nuc == 1] = 100
    a[cell == 2] = 50
    b = np.zeros(shape)
    xx = np.mgrid[:shape[0], :shape[1]][1]
    b[(cell == 1) & (xx > 25)] = 40
    return np.stack([a, b]).astype(np.float32), cell, nuc


def test_compartment_intensities(two_cells):
    img, cell, nuc = two_cells
    df = window_features(img, cell, nuc, [1, 2], CH, P)
    n_nuc, n_cell = (nuc == 1).sum(), (cell == 1).sum()
    assert df.loc[1, "a_nuc_mean"] == 100
    assert df.loc[1, "a_cyto_mean"] == 20
    assert df.loc[1, "a_rim_mean"] == 20
    assert df.loc[1, "a_cell_sum"] == pytest.approx(100 * n_nuc + 20 * (n_cell - n_nuc))
    assert df.loc[1, "a_cell_p99"] == 100 and df.loc[1, "a_cell_p50"] == 20
    assert df.loc[1, "a_ring_mean"] == 5 and df.loc[1, "a_terr_mean"] == 5
    assert df.loc[2, "a_cell_mean"] == 50
    assert np.isnan(df.loc[2, "a_nuc_mean"])           # no nucleus
    assert df.loc[1, "morph_cell_area_um2"] == pytest.approx(n_cell * 0.25)
    assert df.loc[1, "morph_nuc_cell_ratio"] == pytest.approx(n_nuc / n_cell)
    assert not df.truncated.any()


def test_rim_width(two_cells):
    img, cell, nuc = two_cells
    df = window_features(img, cell, nuc, [1], CH, P)
    # rim 1 um = 2 px deep: annulus of a r=12 disc between r~10 and 12
    expected = (cell == 1).sum() - disc(cell.shape, 25, 25, 10).sum()
    assert abs(df.loc[1, "rim_npx"] - expected) / expected < 0.15


def test_radial_profile(two_cells):
    img, cell, nuc = two_cells
    df = window_features(img, cell, nuc, [1, 2], CH, P)
    prof1 = df.loc[1, [f"a_radial_b{k}" for k in range(5)]].values
    assert prof1[0] > prof1[-1]                        # nucleus-bright centre
    prof2 = df.loc[2, [f"a_radial_b{k}" for k in range(5)]].values
    assert np.allclose(prof2, 1)                       # flat cell -> 1 everywhere


def test_polarity(two_cells):
    img, cell, nuc = two_cells
    df = window_features(img, cell, nuc, [1, 2], CH, P)
    assert df.loc[1, "b_polarity_shape"] > 0.3
    assert df.loc[1, "b_polarity_dx_um"] > 0 and abs(df.loc[1, "b_polarity_dy_um"]) < 0.1
    assert df.loc[1, "a_polarity_shape"] < 0.05       # symmetric pattern
    assert df.loc[1, "nuc_offset"] < 0.05


def test_glcm_constant_vs_checkerboard():
    g = np.ones((20, 20), np.int64)
    flat = glcm_features(np.zeros((20, 20), np.int64), g, 1, 4, 1)
    assert flat["contrast"][0] == 0 and flat["asm"][0] == 1
    checker = (np.indices((20, 20)).sum(0) % 2) * 3
    ch = glcm_features(checker, g, 1, 4, 1)
    assert ch["contrast"][0] > 4


def test_grouped_stats_matches_numpy():
    rng = np.random.default_rng(0)
    g = rng.integers(1, 6, 1000)
    v = rng.normal(size=(1000, 2))
    s = grouped_stats(g, v, 6)
    for k in range(1, 6):
        for c in range(2):
            x = v[g == k, c]
            assert s["mean"][k - 1, c] == pytest.approx(x.mean())
            assert s["q90"][k - 1, c] == pytest.approx(np.percentile(x, 90))
    assert s["count"][5] == 0 and np.isnan(s["q50"][5, 0])


def test_tiling_invariance():
    """Same features whether computed in one window or tiled with a halo."""
    rng = np.random.default_rng(1)
    shape = (200, 260)
    cell = np.zeros(shape, np.uint32)
    nuc = np.zeros(shape, np.uint32)
    rows = []
    lab = 0
    for cy in range(15, 200, 30):
        for cx in range(15, 260, 30):
            lab += 1
            r = rng.integers(7, 12)
            cell[disc(shape, cy + rng.integers(-3, 4), cx + rng.integers(-3, 4), r) & (cell == 0)] = lab
            nuc[disc(shape, cy, cx, 4) & (cell == lab)] = lab
    img = rng.gamma(2, 10, size=(2, *shape)).astype(np.float32)
    labels = np.unique(cell[cell > 0])
    ys, xs = np.nonzero(cell)
    cyx = pd.DataFrame({"label": cell[ys, xs], "y": ys, "x": xs}).groupby("label").mean()
    cells = pd.DataFrame({"label": labels, "y_centroid": cyx.y.values, "x_centroid": cyx.x.values})

    def read(y0, x0, h, w):
        out = [np.zeros((2, h, w), np.float32), np.zeros((h, w), np.uint32), np.zeros((h, w), np.uint32)]
        a, b = max(y0, 0), min(y0 + h, shape[0])
        c, d = max(x0, 0), min(x0 + w, shape[1])
        out[0][:, a - y0:b - y0, c - x0:d - x0] = img[:, a:b, c:d]
        out[1][a - y0:b - y0, c - x0:d - x0] = cell[a:b, c:d]
        out[2][a - y0:b - y0, c - x0:d - x0] = nuc[a:b, c:d]
        return out

    full = window_features(img, cell, nuc, labels, CH, P)
    tiled = pd.concat(run_window(read, shape, ty, tx, lab_, 64, 32, CH, P)
                      for (ty, tx), lab_ in plan_windows(cells, 1.0, 64).items()).sort_index()
    assert not tiled.truncated.any()
    pd.testing.assert_frame_equal(full.drop(columns="truncated"), tiled.drop(columns="truncated"),
                                  check_exact=False, rtol=1e-6)


def test_decode_cell_id():
    assert decode_cell_id(0x0001BD08, 1) == "aaablnai-1"
