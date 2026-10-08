"""Load per-section feature tables and join them to the cell annotation."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ["dapi", "bnd", "r18s", "smavim"]
OBS_COLS = ["sample_id", "sample_name", "meta_sample_id", "condition", "model", "stage", "score_sacrifice", "region", "sex",
            "Anno_L1_curated", "Anno_L2", "Anno_L3", "Curated_niche_state", "Global_niche_group",
            "Global_anatomical_region", "Physiological_niche"]


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(open(path or ROOT / "config.yaml"))


def load_features(cfg: dict, sections=None, annotated_only: bool = True, h5ad: str | Path | None = None) -> pd.DataFrame:
    """All section parquets -> one frame indexed '<section_id>:<cell_id>' (= annotation obs index),
    float64 feature columns downcast to float32, annotation columns joined (from `h5ad`, default the config's)."""
    fdir = ROOT / cfg["features_dir"]
    parts = []
    for p in sorted(fdir.glob("*.parquet")):
        if p.stem.endswith("_bench") or (sections and p.stem not in sections):
            continue
        d = pd.read_parquet(p)
        d.index = d.section_id + ":" + d.index
        parts.append(d)
    f = pd.concat(parts)
    num = f.select_dtypes("float64").columns
    f[num] = f[num].astype(np.float32)
    obs = load_obs(cfg, h5ad)
    f = f.join(obs, how="inner" if annotated_only else "left")
    return f


def load_obs(cfg: dict, h5ad: str | Path | None = None) -> pd.DataFrame:
    a = ad.read_h5ad(ROOT / (h5ad or cfg["annotation"]["h5ad"]), backed="r")
    obs = a.obs[[c for c in OBS_COLS if c in a.obs]].copy()
    a.file.close()
    return obs


def load_counts(cfg: dict, index) -> ad.AnnData:
    """Raw counts for the given cells (order preserved)."""
    a = ad.read_h5ad(ROOT / cfg["annotation"]["h5ad"])
    return a[pd.Index(index)].copy()


def feature_families(cols) -> pd.Series:
    """Map feature column -> family label for catalogue tables."""
    fam = {}
    for c in cols:
        if c.startswith("morph_") or c == "nuc_offset":
            fam[c] = "morphology"
        elif "_radial_" in c:
            fam[c] = "radial profile"
        elif "_polarity_" in c:
            fam[c] = "polarity"
        elif "_glcm_" in c:
            fam[c] = "texture (Haralick)"
        elif c.split("_")[0] in CHANNELS and c.split("_")[1] in ("cell", "nuc", "cyto", "rim", "ring", "terr"):
            fam[c] = f"intensity: {c.split('_')[1]}"
    return pd.Series(fam, name="family")
