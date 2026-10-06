# RESULTS log — Beyond Boundaries

## 2026-10-06 — Step 0: inspection (no analysis yet)

Scripts: `scripts/00_inspection/` (run on the analysis Mac, env `sc_py312`). Outputs: `results/00_inspection/`.

### Where things are
- Raw Xenium bundles live on the **analysis Mac** (`ki-cljjykhfx7`, Tailscale), not this laptop:
  `/Volumes/moldiassd/20260506__111857__GoncaloTing_5kMouse_run5` (13 sections), `.../20260506__112634__GoncaloTing_5kMouse_run6` (12 sections).
- Annotated object (local): `~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad`
  (1,384,881 cells, runs 1/2/3/5/6; obs index `obs_id` = `<sample_id>:<cell_id>`; X log-norm, `layers['counts']`, 5,101 genes).
  Runs 1–3 have no raw bundles accessible (P drive) → image analysis restricted to run5/6.

### Bundles (all 25 identical in format)
- XOA `xenium-4.0.1.0`, file format 6.0, instrument sw 4.0.1.4, **fresh frozen**, pixel 0.2125 µm, 5K mouse + 95 custom.
- `morphology_focus/ch000{0..3}_*.ome.tif`: one uint16 plane per file, JPEG2000 1024² tiles, 7 sub-resolution levels (SubIFDs). Multi-file OME (every file's header lists all 4 channels; tifffile `series` cannot read it → read page/tiles directly).
- Channel order verified from OME metadata AND image content (crop `results/00_inspection/C2_G1_Mid_channels_masks_crop.png`):
  ch0 DAPI (nuclear) · ch1 ATP1A1/CD45/E-Cad (diffuse neuropil + rims) · ch2 18S (perinuclear cytoplasm) · ch3 αSMA/Vimentin (vessel wall, ependyma). Same names in all 25 bundles.
- `cells.zarr.zip`: `masks/1` = cell labels, `masks/0` = nucleus labels (full-res, uint32, chunks 431×1600).
  - cell label L ↔ row L-1 of `cells.parquet` / decoded `cell_id` array (100% at centroids; mask areas = `cell_area`).
  - **nucleus labels are a separate index**: nucleus L → `polygon_sets/0/cell_index[L-1]` → cell label +1 (100% concordant). Nuclei never extend outside their cell.
  - `homogeneous_transform` = 4.7059 = 1/0.2125 (µm → px), no offset.
- Segmentation method is ~93–97% **interior (18S)**, 0.5–4% boundary stain, 2–3% nucleus expansion (D_SC: 87/6/8%). Boundary-segmented cells: ~30% have no nucleus.
  ⇒ circularity is mostly an 18S problem, and boundary-method strata will be small.

### Annotation ↔ bundles
- 18/25 bundles are in the annotated object (500,379 cells). Missing: `D_SC` and 6× `ON_*` (optic nerve; probably `optic_nerve_merged*.h5ad` on moldiassd).
- 100% of annotated cell_ids are found in their bundle; some sections were heavily filtered upstream (C2_G2_Bot 36.5k→22.1k, C2_G2_Top 24.4k→18.3k, C2_G3_Bot 27.9k→17.9k).
- Each Xenium section multiplexes several animals: `meta_sample_id` (tissue piece) / `sample_name` (animal) — use `sample_name` as the CV split key.
- Run5/6 design: CHRONIC EAE 202,904 cells, RELAPSE-REMITTING EAE 280,913, **CONTROL only 16,562 (CFA, 2 pieces in run5)**.
- Cell types: `Anno_L1_curated` (19), `Anno_L2` (42), `Anno_L3` (137), `Anno_L4` (174). Lesion/niche: `Curated_niche_state`, `Global_niche(_group)`, `Global_anatomical_region` (WM/GM/…).

### Open questions
- Controls are thin in run5/6 → condition contrast is mostly between EAE stages/niches, not EAE vs control.
- ATP1A1 is pan-neuropil in CNS → ch1 "membrane rim" will be dominated by neighbouring neuropil; needs a local-background (ring outside cell) feature.
- Include optic-nerve / D_SC bundles? (need their annotation).

## 2026-10-06 — Phase 1: per-cell features (`scripts/01_extract_features.py`, `notebooks/01_feature_overview.ipynb`)

- Package `src/beyondboundaries` (io / features / extract), env `environment.yml` (`bb` on analysis Mac), 8 unit tests pass
  (synthetic known values + **exact tiling invariance**; the test caught an image-border bug, fixed).
- 2048² windows + 256 px halo, 6 workers: **66 min for 18 sections**, 537,716 cells × 201 columns, **0 truncated cells**.
  Output `data/features/<section>.parquet` (remote, 671 MB). 500,379 cells (93.1 %) are in the annotation; non-annotated
  cells are QC-poor (median 629 vs 973 transcripts).
- Concordance: mask areas = Xenium `cell_area` exactly; centroids ≤ 0.28 µm. Structural NaNs only (no nucleus 0.55 %,
  of which boundary-segmented cells 31 %; no outer ring 2.2 % — fully enclosed cells).
- Where stains live (compartment ÷ cell mean, median): DAPI nucleus 1.83; **ATP1A1 is higher outside cells (ring 1.29)**
  = neuropil; 18S cell-confined (ring 0.22, partly by construction: 18S drew ~95 % of masks); αSMA/Vim cytoplasmic.
- Raw biology already visible (heatmap): ependymal αSMA/Vim↑, neurons 18S↑ + large, fibroblasts αSMA/Vim↑, Schwann 18S↓.
  **Leukocytes are boundary-channel-dim**, not CD45-bright: the channel is dominated by neuropil ATP1A1 → test CD45 as
  rim/ring enrichment (Phase 3).

## 2026-10-06 — Phase 2: background, AF, normalisation (`notebooks/02_normalisation_qc.ipynb`)

- Level-3 (1.7 µm) maps per section: tissue mask, cell-free tissue (> 5 µm from cells), AF proxy = cell-free ∩ lowest-25 %
  transcript density; local background = σ 25 µm normalised convolution of cell-free pixels. Cached in `data/qc/`.
- **Autofluorescence WM vs GM** (AF-proxy pixels, median over sections): DAPI 1.0 vs 1.0; 18S 11.9 vs 11.5 (WM/GM 0.97);
  ATP1A1 55 vs 15 (per-section median ratio 1.5); αSMA/Vim 1.0 vs 0.2 (tiny absolute). No broadband myelin AF (18S flat);
  the WM boundary-channel excess is more likely axolemmal ATP1A1. Local background subtraction handles both.
- Local background ≈ 1.3× the cell mean for ATP1A1 in GM (cells dimmer than neuropil); ~0.2 for 18S.
- 2.2 % of cells within 20 µm of the tissue edge; `edge_um` kept as covariate.
- **Normalisation**, after two failed variants (recorded so they're not retried):
  1. scale = median background-subtracted reference cell mean → ≤ 0 for ATP1A1 and ~0 for αSMA/Vim (unusable);
  2. scale = median raw reference mean → αSMA/Vim noise-level (most cells ≈ 0);
  3. **final: (x − local bg) / p90 of raw cell means of physiological-niche cells, per section** (= slide region, the
     technical unit). Between-section CV (stain-positive reference type): DAPI 0.077→0.052, 18S 0.099→0.065.
     Per-piece scaling was tried and rejected because piece-level intensity tracks disease (next bullet).
- **Finding: within a section, pieces from sicker animals have brighter αSMA/Vim (ρ = 0.45 vs lesion fraction,
  p = 8e-4; ρ = 0.43 vs score, p = 0.002) and dimmer 18S (ρ = −0.42, p = 0.002) in their physiological-niche cells**
  (51 pieces). Caveat: not yet controlled for cell-type composition / spinal level → Phase 4.
- Output `data/features_norm.parquet` (500,379 annotated cells × 232 columns).

### Open questions (updated)
- Piece-level αSMA/Vim↑ / 18S↓ with disease: composition or level-driven, or cell-intrinsic? (Phase 4 within type.)

## 2026-10-06 — Phase 3: sanity atlas (`notebooks/03_sanity_atlas.ipynb`)

Expectation tests, AUROC (all cells / 18S-seg / boundary-seg / nucleus-exp; animals with AUROC > 0.5):
| | expectation | all | 18S | boundary | nuc-exp | animals |
|---|---|---|---|---|---|---|
| E1 | leukocyte CD45 at rim (bnd rim/ring) | **0.48 ✗** | 0.48 | 0.54 | 0.46 | 7/25 |
| E2 | VSMC αSMA-bright | 0.66 ✓ | 0.66 | 0.60 | 0.64 | 25/25 |
| E3 | leukocyte vimentin > neuron/oligo | 0.73 ✓ | 0.74 | 0.77 | 0.55 | 25/25 |
| E4 | neuron/oligo low ch3 | 0.74 ✓ | 0.75 | 0.78 | 0.58 | 25/25 |
| E5 | ependymal vimentin | 0.86 ✓ | 0.86 | 0.88 | 0.54 | 25/25 |
| E6 | neuron 18S (cytoplasm) | 0.78 ✓ | 0.78 | 0.85 | 0.60 | 25/25 |

- **CD45 is not detectable**: leukocytes are not boundary-bright overall (0.55) nor in low-ATP1A1 context (lowest local-bg
  quartile per section: AUROC 0.41–0.49 for every leukocyte type). In this mouse spinal cord ch1 behaves as ATP1A1 only.
  → ask 10x whether the kit's CD45 antibody is mouse-reactive.
- **Segmentation-method main effects are large for shape/distribution features** (boundary-seg: bnd outer radial bin
  +10 SD, solidity +6 SD — the stain drew the edge; nucleus-exp: solidity +6, 18S entropy −6). Between-type pattern
  after removing offsets: 18S 0.998, boundary 0.63, nucleus-exp 0.30 (r with all-cells pattern).
  → Phase 4 main analysis on 18S-segmented cells (96 %), all-cells as sensitivity; 18S distribution features flagged
  as partly by construction.

## 2026-10-06 — Phase 4: orthogonality (`notebooks/04_orthogonality.ipynb`, src `orthogonality.py`)

Setup: 18S-segmented cells, 14 types (≥ 500 cells, ≥ 5 animals; ≤ 30k cells/type), 118 image features (rank-INT within
type), ridge, 5-fold GroupKFold by animal; covariates = section, spinal level, log transcripts, log area, log edge dist;
transcriptome = 50 PCs of log-normalised counts fitted in-fold. ~2.3 h on the shared Mac (load 40–60).
Bug caught mid-run: reverse-direction covariate models scored R² < 0 under animal-grouped CV (cell-state composition
differs between animals) → inflated "image gain". Fixed: consistent PCA targets, pooled OOF R², gains clipped at 0.

- **The transcriptome explains little of the image**: median R²(covariates + transcriptome) per feature 0.11–0.18 across
  types (transcriptome-unique ΔR² 0.06–0.10) → 82–89 % of image-feature variance is not linearly explained.
  Best explained: morphology (0.39; nucleus:cell ratio ΔR² 0.40–0.47 in every type), rim/cyto/nuclear intensities
  (0.23–0.36); least: polarity (0.06), radial profile (0.09), territory intensity (0.09).
- **The unexplained intensity signal is spatially structured**: a cell's residual vs its 10 nearest same-type
  neighbours r ≈ 0.44–0.49 (rim/cyto/nuc intensity), 0.29–0.31 (ring/territory); polarity 0.08, radial 0.13
  (≈ noise). Not yet separated into technical field vs tissue biology → next check.
- **Reverse — images predict transcriptional state**: e.g. neuron PC1 R² 0.75 (image gain 0.36; neurofilament genes
  *Nefm/Nefh/Nefl*), astrocyte PC1 0.59 (gain 0.41; *C3*, *A2m*, *Gfap*, *Mt2*, *Slc6a11*, *Fgfr3*, *Aldoc*, *Gjb6*),
  oligodendrocyte PC1 0.43 (gain 0.35; *Klk6*, *Ermn*, *Ptgds* and neighbour-spillover genes *Snap25*, *Eno2*, *Sncb*).
  αSMA/Vim intensity/texture in astrocytes is well transcriptome-explained (ΔR² 0.23–0.27) → the vimentin stain largely
  re-measures transcriptional reactivity.
- **Lesion information beyond the transcriptome is small**: residual lesion-vs-physiological shift (per animal) in
  32/1114 type×feature tests (q < 0.05), 0.1–0.26 SD; clearest = lower ATP1A1 ring/territory around myeloid, endothelial,
  VSMC in lesions (neuropil loss around the cell). Lesion-niche classification: image alone AUROC 0.47–0.82, but
  image adds ≤ 0.015 to the transcriptome (0.79–0.96) — partly by construction (niches are transcriptome-defined).
- Residual clustering (Leiden, res 0.3): no discrete hidden states; 1 cluster in most types, 2 continuous-gradient
  splits elsewhere (mostly ring/territory intensity); none lesion-associated (all q ≥ 0.46).

### Open questions (updated)
- Is the spatially coherent residual a technical field (stain/focus) or tissue biology? Decompose into a smooth
  all-cell-type field + cell-specific part; test both against lesion distance / piece disease.
- Lesion labels are transcriptome-derived; an image-independent outcome (clinical score / stage per animal) is the
  fairer test of added value.
