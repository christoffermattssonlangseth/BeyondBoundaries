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
# # Beyond Boundaries — what do the Xenium multimodal segmentation stains add to the transcriptome?
#
# **Data.** Mouse EAE spinal cord (RRMAP2 runs 5 and 6): 18 annotated Xenium 5K sections, 25 animals (chronic and
# relapse-remitting EAE, CFA controls), 500,379 annotated cells. Four morphology channels besides transcripts:
# DAPI · ATP1A1/CD45/E-Cadherin (boundary) · 18S rRNA (interior RNA) · αSMA/Vimentin (interior protein).
#
# **Approach.** Per cell and channel: intensities in 6 compartments (nucleus, cytoplasm, 1 µm rim, 2 µm ring,
# 10 µm territory), radial profile, polarity, texture, morphology (201 features; notebook 01) → background /
# normalisation (02) → do the stains show known biology (03) → how much of each image feature does the transcriptome
# explain, and vice versa (04) → is the rest technical or biological; does it add per animal (05) → targeted
# readouts (06) → look at the cells (07).
#
# This notebook only *collects* results (figures from `results/`, tables from CSVs); every number links to the
# notebook that computed it.

# %%
import io
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display
from PIL import Image as PILImage

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RES = ROOT / "results"
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)


def fig(path, width=900):
    """Embed a display-sized JPEG copy (full-resolution PDF/PNG stay in results/)."""
    p = RES / path
    if not p.exists():
        display(Markdown(f"*missing figure: `{path}`*"))
        return
    im = PILImage.open(p).convert("RGB")
    if im.width > 2 * width:
        im = im.resize((2 * width, round(im.height * 2 * width / im.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    display(Image(data=buf.getvalue(), format="jpeg", width=width))
    display(Markdown(f"<sub>`results/{path}`</sub>"))

# %% [markdown]
# ## Bottom line
#
# | question | answer | where |
# |---|---|---|
# | Do the stains show known biology? | **Yes** — VSMC αSMA, leukocyte & ependymal vimentin, neuron 18S, low ch3 in neurons/oligos; all 25 animals | 03 |
# | Is CD45 usable? | **No** — boundary channel behaves as ATP1A1 (neuropil) only, even in low-neuropil areas | 03 |
# | How much of the image does the transcriptome explain? | **~15 %** per feature (median); morphology and rim/cyto intensities best | 04 |
# | Is the unexplained ~85 % hidden biology? | **Mostly not** — a local optical/staining field shared across cell types + cell-level variation unrelated to lesions | 05 |
# | Do images add to the transcriptome for lesion identity? | **No** (≤ +0.015 AUROC; niches are transcriptome-defined) | 04 |
# | …for clinical score per animal? | Images alone ρ = 0.46 (p = 0.04) vs transcriptome 0.86; combined not better | 05 |
# | What *do* they add? | Measurements of genes **not on the panel** (*Vim*, *Acta2*, *Atp1a1*): vimentin reactivity timing, **WM neuropil loss (−25 %)**, **18S gain in activated lesion cells**, **T-cell 18S polarity towards vessels**, a candidate **18S-texture severity marker** (holds within section and after a crowding adjustment) | 05–07 |

# %% [markdown]
# ## 1. Features look right (notebook 01)
# Mask areas equal Xenium's exactly; centroids within 0.28 µm; 0 truncated cells. Each stain has its own compartment —
# notably **ATP1A1 is higher *outside* cells** (neuropil).

# %%
fig("01_features/compartment_profile.png", 800)
fig("01_features/cell_gallery.png", 430)

# %% [markdown]
# ## 2. Background and normalisation (notebook 02)
# No broadband myelin autofluorescence (18S WM/GM 0.97). Final normalisation = local background subtraction, scaled
# per section by the p90 of physiological-niche cells. **Stain intensity tracks disease at piece level**: αSMA/Vim
# brighter (ρ = 0.45, p = 8e-4) and 18S dimmer (ρ = −0.42, p = 0.002) in physiological-niche cells of sicker animals.

# %%
fig("02_normalisation_qc/af_wm_vs_gm.png", 800)
fig("02_normalisation_qc/piece_intensity_vs_lesion.png", 800)
fig("02_normalisation_qc/spatial_bnd.png", 800)

# %% [markdown]
# ## 3. The stains recover known biology — except CD45 (notebook 03)

# %%
display(pd.read_csv(RES / "03_sanity_atlas/expectation_tests.csv").query("stratum == 'all'")[["test", "feature", "auroc", "verdict"]].round(3))
fig("03_sanity_atlas/expectation_tests.png", 700)
fig("03_sanity_atlas/E1_cd45_context.png", 550)
fig("03_sanity_atlas/atlas_celltype_heatmap.png", 900)

# %% [markdown]
# Segmentation method shifts shape/distribution features for every cell type (the stain drew that edge), so all
# downstream analyses use 18S-segmented cells (96 %).

# %%
fig("03_sanity_atlas/atlas_by_segmethod.png", 900)

# %% [markdown]
# ## 4. The transcriptome explains ~15 % of the image; images predict transcriptional state (notebook 04)

# %%
r2 = pd.read_csv(RES / "04_orthogonality/r2_image_from_transcriptome.csv")
s = r2.groupby("cell_type")[["cov", "full", "tx_unique"]].median().sort_values("full")
s.columns = ["R² covariates", "R² covariates + transcriptome", "ΔR² transcriptome"]
display(s.round(3))
fig("04_orthogonality/r2_heatmaps.png", 900)
fig("04_orthogonality/reverse_pcs.png", 700)

# %% [markdown]
# Genes best predicted by the images (beyond covariates), top 6 per cell type — reactive astrocyte genes, neurofilaments,
# oligodendrocyte maturation genes, and neighbour spill-over genes:

# %%
g = pd.read_csv(RES / "04_orthogonality/r2_genes_from_image.csv")
top = g.sort_values("img_unique", ascending=False).groupby("cell_type").head(6)
display(top.groupby("cell_type").gene.apply(lambda x: ", ".join(x)).to_frame("top genes predicted by image"))

# %% [markdown]
# Images are redundant with the transcriptome for lesion-niche identity:

# %%
display(pd.read_csv(RES / "04_orthogonality/lesion_classification.csv", index_col=0).round(3))

# %% [markdown]
# ## 5. What is the unexplained 85 %? (notebook 05)
# Much of it is shared with neighbouring cells of *other* types (a local field) and that field does not follow lesions;
# the cell-specific remainder barely does either.

# %%
fig("05_field_and_animal/field_share_heatmap.png", 700)
fig("05_field_and_animal/field_lesion_vs_technical.png", 800)
display(pd.read_csv(RES / "05_field_and_animal/lesion_shift_by_part.csv").groupby("part")
        .apply(lambda d: pd.Series({"tests": len(d), "q<0.05": int((d.q < 0.05).sum())}), include_groups=False))

# %% [markdown]
# ## 6. Per animal: images carry real but weaker disease information; 18S texture stands out (notebooks 05, 07)

# %%
display(pd.read_csv(RES / "05_field_and_animal/animal_score_prediction.csv", index_col=0).round(3))
fig("05_field_and_animal/animal_score_prediction.png", 900)
fig("05_field_and_animal/r18s_texture_within_section.png", 650)
fig("07_visual_checks/r18s_texture_low_vs_high_score.png", 800)
fig("07_visual_checks/r18s_texture_per_animal_hist.png", 550)

# %% [markdown]
# Is it just crowding (bright 18S neighbours bleeding in)? Mostly not — the within-section link survives adjustment
# for neighbouring 18S, local density and cell size:

# %%
fig("07_visual_checks/r18s_texture_crowding_adjustment.png", 600)

# %% [markdown]
# ## 7. Targeted readouts (notebooks 06, 07)
# ### Astrocyte reactivity: vimentin protein vs RNA — transcription comes first

# %%
fig("06_targeted_readouts/astro_protein_vs_rna.png", 900)
fig("06_targeted_readouts/astro_quadrant_components.png", 800)
display(pd.read_csv(RES / "06_targeted_readouts/astro_quadrants.csv", index_col=0).round(3))
fig("07_visual_checks/astro_protein_only_vsmc_gallery.png", 800)

# %% [markdown]
# ### Leukocyte polarity: 18S points towards vessels in T cells (not in neurons/oligos); DAPI/αSMA "polarity" is bleed

# %%
fig("06_targeted_readouts/polarity_vessel_direction.png", 550)
fig("07_visual_checks/tcell_perivascular_polarity.png", 900)

# %% [markdown]
# ### 18S rises in activated cells in lesions, beyond RNA content

# %%
fig("06_targeted_readouts/r18s_lesion_by_type.png", 700)

# %% [markdown]
# ### Neuropil loss: −25 % surrounding ATP1A1 in white-matter lesions (within piece)

# %%
fig("06_targeted_readouts/neuropil_index_map.png", 900)
fig("06_targeted_readouts/neuropil_index_paired.png", 450)

# %% [markdown]
# ## Caveats
# - CD45 component undetectable → no immune-membrane readout from ch1.
# - αSMA and vimentin share a channel; 18S drew ~95 % of masks (18S distribution features partly by construction).
# - Lesion niches are transcriptome-defined, favouring the transcriptome in lesion comparisons.
# - 25 animals; only 2 CFA control pieces in runs 5/6; runs 1–3 have no accessible raw images.
#
# ## Next steps
# 1. Validate the 18S-texture severity marker on runs 1–3 (raw images on the P drive) or a new run.
# 2. Disentangle vimentin vs αSMA in "protein-only" astrocytes (VSMC proximity, notebook 07; ideally separate channels).
# 3. Perivascular T-cell 18S polarity: relate to cuff position / migration state.
# 4. Use the neuropil-loss index and 18S per cell as covariates in the RRMAP2 lesion analyses.
