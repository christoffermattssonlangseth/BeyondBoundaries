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

## 2026-10-07 — Unexplained signal: field vs cell; animal level (`notebooks/05_field_vs_cell_and_animal_level.ipynb`)

- **Residual = cross-type field + cell-specific part.** Field = mean residual of the 30 nearest *other-type* cells.
  r(residual, field) for rim/cyto/nuc intensity 0.37–0.39 (ATP1A1 0.39, αSMA/Vim 0.17, DAPI 0.16, 18S 0.13) → a
  large share of the "unexplained" intensity signal is shared by all cells in a neighbourhood.
- The field does **not** track lesion distance (within-section ρ ≈ 0 for every channel/family) but follows local
  background where optics predict it (ATP1A1 nuclear intensity vs local bg ρ = −0.55: neuropil glow into nuclei).
  The cell-specific remainder doesn't track lesion distance either (ρ ≈ 0); per-animal lesion shifts 19/924 (q < 0.05),
  ~0.1 SD. → The ~85 % unexplained is mostly a local optical/staining field + cell-level variation unrelated to lesions.
- **Animal level** (25 animals, lumbar only, LOO ridge, clinical score): transcriptome ρ = 0.86 (R² 0.75, perm p 0.005);
  image ρ = 0.46 (R² 0.28, perm p 0.04); transcriptome + image ρ = 0.74 (adding ~670 noisy features to n = 25 hurts);
  section-membership baseline uninformative (no batch confound).
- **Candidate image biomarker: 18S texture.** 18S Haralick correlation (smoother, less punctate 18S) rises with clinical
  score in 6 cell types (ρ 0.37–0.76) and **holds within section** (animals vs same-section animals: ρ 0.53–0.86,
  12 animal-sections); DAPI texture (same optics) does not (ρ −0.18…0.39). Tissue-wide → biology (ribosome
  redistribution / RNA degradation in inflamed tissue) or animal-level tissue handling; needs another run to confirm.

## 2026-10-07 — Phase 5 targeted readouts (`notebooks/06_targeted_readouts.ipynb`)

*Vim*, *Acta2*, *Atp1a1* are **not on the 5K panel** → for 3 of 4 stain targets the image is the only per-cell readout.

1. **Astrocyte reactivity, protein vs RNA** (47,062 astrocytes). Protein score (spec: vimentin-channel cell mean +
   cyto p90 − ATP1A1 rim, within-section z) vs RNA (reactive C3/Gfap/Serpina3n/Cd44/Osmr/Timp1/Socs3/Serping1/H2-D1/
   Psmb8/Gbp2 − homeostatic Slc1a3/Kcnj10/Slc6a11/Fgfr3/Gjb6/Aldoc/Aldh1l1): ρ = 0.48 (per section 0.17–0.58); driven
   by the vimentin channel (components ρ 0.44 / 0.62; ATP1A1 rim 0.14). Strongest single genes: *C3* 0.50, *Serping1*
   0.39, *Gfap* 0.38; homeostatic *Slc6a11* −0.44, *Aldoc* −0.37.
   - **RNA-only** (reactive transcription, vimentin-quiet): 95 % in lesions, share peaks in active disease (PEAK1 20 %,
     PEAK2 17 % vs CFA 3.5 %), lower in remission → **reactive transcription precedes vimentin protein**
     (protein-only − RNA-only per animal: active −0.08, remission −0.03, pre/none +0.04; remission vs active p = 0.019).
   - **Protein-only** (vimentin-channel-high, homeostatic RNA; n = 401): mostly GM, outside lesions, deep tissue;
     most frequent in CFA/pre-symptomatic animals. Caveat: channel pools αSMA → arteriole-associated astrocytes possible.
2. **Leukocyte polarity.** Magnitude perivascular (< 15 µm to vessel) vs parenchymal (> 50 µm): no meaningful
   difference. Direction: **18S polarity points towards the nearest vessel in T cells (cos 0.14), fibroblasts (0.17),
   MDM (0.06), microglia (0.05) but not neurons (0.00) or oligodendrocytes (−0.01)** → cell-type-specific, not blur.
   DAPI and αSMA/Vim "towards vessel" also appear in neurons/oligos/astrocytes → optical bleed (artefact).
3. **18S per cell type, lesion vs physiological** (per animal, within-section z): up in lesion Schwann (+0.73),
   fibroblasts (+0.57), endothelium (+0.54), DAO (+0.44), CD4 T (+0.44), astrocytes (+0.26–0.34), microglia (+0.21);
   down in OPC/COP (−0.18) and MOL (−0.14). 18S tracks transcript density (ρ 0.6–0.75 within type); after adjusting,
   increases persist (e.g. endothelium +0.39, Schwann +0.83, homeostatic astro +0.39) while the oligodendrocyte drop
   ~vanishes (MOL +0.04) → lesion 18S gain beyond RNA content in activated cells; oligo loss = RNA-content loss.
4. **Neuropil-loss index** (territory ATP1A1 ÷ median of same piece × WM/GM; first version referenced per section and
   was confounded by piece-level intensity — replaced). **WM lesion cells have 25 % less surrounding ATP1A1 than
   physiological WM of the same piece (38 pieces, p = 8e-6); GM no difference (45 pieces, p = 0.62);** not related to
   clinical score. Highest in ventral/dorsal rim OL niches (1.3–1.4).

### Open questions (updated)
- 18S texture biomarker: validate on runs 1–3 (raw images on P drive) or a new run; visual check of high vs low animals.
- Protein-only astrocytes: vimentin or αSMA from arterioles? (distance to VSMC; channel can't separate.)
- T-cell 18S polarity towards vessels: uropod / migration orientation? Check perivascular-cuff T cells by eye.

## 2026-10-07 — Visual checks (`notebooks/07_visual_checks.ipynb`) and summary (`notebooks/00_summary.ipynb`)

- Perivascular T cells: cos(18S polarity, to vessel) mean 0.13 (n = 4,163), 0.20 in the most polarised quarter; 9/12
  random strongly polarised cells point vessel-wards in the gallery.
- "Protein-only" astrocytes vs VSMC: median nearest-VSMC distance 58 µm, same as quiet astrocytes (p = 0.4); αSMA/Vim
  not shifted towards the VSMC (cos 0.01); 7 % within 10 µm of a VSMC (vs 1.5 %) → mostly vimentin, a minority may be
  arteriole αSMA.
- 18S texture, same section (run5_C2_G3_Top): C_P1_6 (score 3.25) vs C_CFA_6 — smoother 18S inside cells (median r 0.73
  vs 0.67 myeloid, 0.73 vs 0.69 astrocytes) and more bright 18S around cells. **Crowding test:** neighbouring 18S, local
  density and area explain 17–33 % of per-cell texture, but the within-section score link largely remains (ρ 0.41–0.81
  adjusted vs 0.53–0.86 raw).
- All findings written into the notebooks as **Finding** cells; `00_summary` collects them with figures;
  `report/BeyondBoundaries_summary.pdf` = rendered summary.
