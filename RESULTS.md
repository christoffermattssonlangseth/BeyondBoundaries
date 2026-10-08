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

## 2026-10-07 — Image-only prediction (`notebooks/08_image_only_prediction.ipynb`, in progress)

Gradient-boosted trees on image features only (no transcript counts), 5-fold CV grouped by animal, 18S-segmented cells.
- **Cell type (14 L1 types):** balanced accuracy 0.46 with cell + image-neighbourhood features (0.44 cell only; chance
  0.07). Ependymal 0.86, neuron 0.81, Schwann 0.76; immune poor (myeloid 0.16, NK/DC 0.17, DC 0.25, T 0.26 — no CD45).
  Errors stay within lineage (immune↔immune, endothelium↔VSMC↔fibroblast, OPC↔oligo).
- **Per stain:** 18S 0.27, αSMA/Vim 0.25, DAPI 0.24, ATP1A1 0.21, morphology 0.17; all except 18S 0.40 → identity is
  spread across stains, not an artefact of the 18S masks.
- **Subtypes:** neurons 0.64 (chance 0.33; cholinergic 0.72), astrocytes 0.60 (0.25; reactive 0.68), myeloid 0.53
  (0.25; microglia 0.72, MDM 0.53), oligodendrocyte lineage 0.47 (0.25).
- **Anatomical region (10 classes):** images 0.63 (chance 0.10; central canal 0.88, dorsal horn 0.80, DRG 0.79, GM 0.75,
  WM 0.45) vs transcriptome cell-type composition of 15 NN 0.68; **both 0.73 → images add anatomical information**.

## 2026-10-07 — Images vs tissue pieces vs animals (check requested)

Each Xenium image (`sample_id`) holds 2–3 tissue pieces (`meta_sample_id`) from (usually) different animals
(`sample_name`). Checked: 51 pieces in 18 images (15 × 3, 3 × 2); every piece = one animal; no piece spans images;
closest pieces 326 µm apart (typically 650–930 µm). Spatial neighbourhoods computed within an image never cross pieces
(k = 15/30: 0 cells; k = 50: 0.01 %); lesion distance always from the same piece → earlier results unaffected.
Hardening: `lesion_distance` / `nearest_of` now group by piece by default; check + figure added to notebook 02.
Statistical unit throughout = animal (`sample_name`); normalisation per image (shared staining/imaging).
- **Lesion state (7 classes):** images 0.54 (chance 0.14), transcriptome composition 0.65, both 0.67.
- **Lesion maps (lesion vs physiological, held-out animals):** AUROC images 0.90 (cell only 0.84), transcriptome
  composition 0.92, both 0.93; maps reproduce lesion extent incl. a fully lesioned piece. Top signals: vimentin-channel
  texture/intensity in surrounding tissue (AUROC 0.76–0.78).
- **Animal metadata (per held-out animal):** clinical score ρ images 0.62 / transcriptome 0.79 / both 0.76; day of
  sacrifice (chronic) 0.52 / 0.84 / 0.75; chronic phase accuracy 0.56 / 1.00 / 0.94; RR timing and peak-vs-remission not
  predictable (9 animals); sex 0.25 (no signal, control). **Run5 vs run6: images 1.00, transcriptome 0.44 → strong
  technical run signature in the images** → runs 1–3 must be compared within run / batch-corrected.
- Process notes: a cache-key collision (RR tasks reused chronic results) was caught and fixed before any RR result was
  reported; the machine was swapping (concurrent c2l job, since stopped) → float32 features + freeing AnnData.

## 2026-10-07 — Conclusions + cross-run transfer (`notebooks/09_conclusions.ipynb`)

Same image features / model as notebook 08, trained on all animals of one run, tested on the other.
- **Lesion vs physiological:** run5 → run6 AUROC 0.90 (median per animal 0.91, 6 animals), run6 → run5 0.87 (19
  animals); within-run CV 0.90. Rank-within-image features: 0.87 / 0.86 (no gain).
- **Cell type (14):** balanced accuracy 0.45 / 0.42 (rank 0.45 / 0.43) vs 0.46 within run; chance 0.07.
- → The run signature (images identify run at 100 %) does not carry the lesion or cell-type models; per-image
  normalisation (notebook 02) suffices for transfer between these two runs. Raw feature values still differ by run.
- One-figure summary `results/09_conclusions/conclusions_figure.{pdf,png}`; conclusions + "how to use the stains" table.

## 2026-10-07 — Lesions redefined; lesion states across both disease courses (`notebooks/10_lesion_states.ipynb`)

All five runs for RNA (1.38 M cells, 67 animals; full object `data/RRMAP2_all_runs.h5ad` on the analysis Mac); images
runs 5/6 only (runs 1–3 copying). Daily clinical scores/weights: `data/clinical/` (per-animal metrics in
`animal_course_metrics.csv`).
- **Curated niches overcall lesions:** never-immunised controls 4–20 % "lesion" (almost all Lesion_mix).
- **Control-referenced lesions** (30-NN neighbourhood: 23 cell types/states, 7 programs, cellularity; Mahalanobis vs
  control neighbourhoods of the same region class; threshold = 99th pct of leave-one-control-animal-out): controls
  0.3–2.4 % (MOG-CFA and PLP-CFA alike → pooled reference OK); 72 % of Lesion_mix cells are not lesion.
- **Lesions resolve** (median share, RR): PEAK1 0.93 → REMISSION1 0.53 → MONOPHASIC 0.35 → REMISSION2 0.30 →
  REMISSION2_LONG 0.14. Chronic: PEAK1 0.87, SEVERE16 0.52 vs MILD16 0.28.
- **Six lesion states** (k-means on 10 axes; named by the axes that distinguish them): S1 monocyte-derived myeloid
  (active; ~35 % of tissue at PEAK1/PEAK2, ~0.5 % in REMISSION2_LONG), S4 monocyte-derived + oligo disease state
  (peak), S2 fibrosis + myelinating-oligo loss (late: SEVERE30 20 %, MILD30 13 %, REMISSION2 12 %), S0 astro reactivity
  + DAO (persists after peak), S3 astro reactivity + MHC-II, S5 lymphocytic infiltration. Course: active infiltrate →
  fibrotic/demyelinated + glial-reactive.
- REMISSION1 vs MONOPHASIC state mix is similar (S1 7.6 % vs 5.3 %; S2 6.0 % vs 8.0 %) — an earlier impression that
  monophasic animals lack active tissue does not hold up in the medians.
- **Clinical scores:** chronic severity is set at the first attack (SEVERE all peak 3.5, never below 2.25; MILD peak
  2.0–2.5); relapse is not predictable from the first attack (monophasic vs relapsing peaks/nadirs overlap).
- **Image pilot (runs 5/6, 7 peak vs 5 recovery animals, uncorrected):** neuropil index (vs the piece's own
  non-lesion tissue) lower at peak than in recovery within the same state (S1 0.79 vs 0.95; S5 0.82 vs 0.94; S3 0.93
  vs 1.05, p = 0.01); vimentin trends the other way (tissue vimentin in S1 1.8 vs 3.2 z; astro vimentin in S0 0.13 vs
  0.90). Suggests neuropil recovers while a vimentin scar builds — needs runs 1–3 (13 more recovery animals).

## 2026-10-08 — Runs 1–3 images; disease courses (`notebooks/11_disease_courses.ipynb`)

- Runs 1–3 (36 sections) copied (images + masks only), integrity-checked (`scripts/check_bundle_integrity.py`: all
  zip CRCs, parquet/h5 reads, every image tile; one corrupted cells.zarr.zip from the copy was re-fetched and
  md5-verified), extracted (201 features, XOA 3.2 works after skipping AppleDouble `._*` files) and normalised with
  the notebook 02 method (`scripts/02_normalise_all_runs.py` → `data/features_norm_all.parquet`; runs 5/6 reproduce
  `features_norm.parquet` exactly). Runs 1–3 have no transcripts.parquet → no AF proxy (QC only).
- Pieces touch in runs 1–3 (6–30 µm in 8 images): cells < 20 µm from another piece excluded from image readouts
  (0.01–0.03 % of cells).
- **Vimentin marks contained/resolving lesions** (astro vimentin in lesions, z within image): MILD16 vs SEVERE16 2.3
  vs 0.7, MILD30 vs SEVERE30 4.4 vs 2.0, MONOPHASIC vs REMISSION1 3.3 vs 1.1, chronic vs RR PEAK1 2.6 vs 0.2
  (p = 0.04), peak → recovery within state S0 0.16 → 1.14; chronic-late ρ(score) = −0.66. Small groups; consistency
  across contrasts is the evidence.
- **Chronic severity = persistent active inflammation**: severe vs mild more lesion, S1/S4, T cells, cellularity
  (ρ with score 0.77–0.89 over 13 chronic-late animals; run1 = MILD30 confound), MOL depleted; neuropil no
  difference. Tissue-loss hypothesis not supported.
- **Relapse and B cells**: RR B cells rise after the first attack and move into aggregates/meninges (PEAK1 14 →
  REMISSION1 22 per 1000; aggregated 0.26 → 0.55); MONOPHASIC < REMISSION1 (9 vs 22, p = 0.03), with a deeper first
  recovery (nadir 0.25 vs 0.75, p = 0.03). Chronic arm ≤ 8 per 1000.
- **Chronic vs RR PEAK1**: RR more S4 (0.26 vs 0.17), less lipid-associated myeloid, more B cells, more weight loss.
- **Damage memory (all runs)**: neuropil back towards normal, vimentin up from peak to recovery within states —
  direction replicates the pilot; best q = 0.13.
- Open: lesion neuropil index > 1 in some chronic groups vs notebook 06's WM loss — compare references.

### Correction (2026-10-08): MILD30 clinical scores
The first notebook 11 run missed the MILD30 scores (annotation `C_L_k` vs score sheet `C_M30_k`), so the chronic-late
score correlations used 8 animals (run 5 only). Fixed in `scripts/clinical_metrics.py` (k = 1, 4, 5 unique on day +
sex; 2/3 paired by number). With all 13: strongest correlate of score = **low lesion astrocyte vimentin ρ = −0.80**;
lesion share 0.58, T cells 0.47, cellularity 0.38 (run 5 only: 0.77–0.89). Notebook 12: vimentin effect survives a
within-piece contrast (lesion − non-lesion astrocytes of the same animal: MILD16 2.1 vs SEVERE16 1.0, MILD30 3.0 vs
SEVERE30 1.9, MONOPHASIC 4.6 vs REMISSION1 1.3, chronic vs RR PEAK1 1.1 vs 0.6) while the DAPI contrast is ~0 →
not piece brightness.

## 2026-10-08 — Replication on runs 1–3 (`notebooks/13_replication_runs123.ipynb`)
Same code on runs 5/6 (reproduces earlier numbers) and on runs 1–3 (new):
- **18S texture severity marker — partly replicated.** Within-section ρ in runs 1–3 holds for astrocytes 0.76,
  myeloid 0.61, oligodendrocytes 0.54, but not endothelium (−0.01), fibroblasts (0.09), neurons (−0.32); median over six
  types 0.76 → 0.32. DAPI control ~0. Downgrade from "candidate biomarker" to "glial/myeloid-specific, needs work".
- **WM neuropil loss — replicated:** lesion ÷ physiological ×0.72 in runs 1–3 (58 pieces, p < 1e-4) vs ×0.79 in runs
  5/6 (weaker with control-referenced calls: ×0.93 / ×0.96).
- **Notebook 11's "neuropil index > 1"** is not a reference problem (references agree, ratio 1.00 GM / 1.005 WM): it
  pooled WM and GM lesion cells; GM lesion cells sit at ≥ 1, WM at 0.91–0.98. Use WM-only.
- **Astrocyte RNA vs vimentin — replicated:** ρ 0.51–0.58 per run; RNA-only astrocytes more common in active disease
  than recovery (runs 1–3: 0.076 vs 0.046, p = 0.01; runs 5/6: 0.111 vs 0.071, p = 0.02).
- **T-cell 18S polarity towards vessels — replicated:** cos 0.19 (33 animals, p < 1e-4) vs 0.14; also fibroblasts,
  MDM, microglia; neurons ~0, oligodendrocytes 0.016 (small optical floor).
- **Leave-one-run-out image models — transfer to every run:** lesion AUROC 0.88–0.89 (curated) / 0.85–0.91
  (control-referenced); cell type 0.42–0.47 (chance 0.07).

## 2026-10-08 — Relapse: new lesions or reactivated old ones? (`notebooks/14_relapse_lesion_origin.ipynb`)
First attack (10 animals) vs relapse peaks (9):
- Active (S1/S4) tissue at relapse touches old fibrotic S2 tissue far more: within 50 µm 0.23 vs 0.05 (p = 4e-4);
  median distance 101 vs 265 µm; enrichment over permutation 0.33 vs 0.14 (p = 9e-4; < 1 in both → separate patches
  that meet at borders).
- But most relapse activity is in **new** lesions: more lesion objects (17 vs 8, p = 0.01); only 19 % of active relapse
  tissue sits in mixed active/old objects (0 % at first attack, p = 0.002). Active tissue ~95 % WM in both.
- B-cell aggregates at relapse sit near old tissue (0.72 vs 0.31) and in meninges (0.33 vs 0.05).
- Vimentin shows no active/old interface signature (paired p ≥ 0.16).
- Caveat: S2 also forms a thin pial rim at first attack (1.7 % of lesion tissue) — "old" is not purely old.

## 2026-10-08 — Vimentin robustness, lesion border, αSMA (`notebooks/15_vimentin_robustness.ipynb`)
- Vimentin effect (milder − severe) keeps its direction in all four contrasts under every test: within animal, far
  from vessels, no αSMA-type cell within 20 µm, deep WM, reactive astrocytes only, within lesion states. Controls ~0
  (DAPI, neurons/oligos, sharpness, smooth-muscle/pericyte/fibroblast RNA in the astrocytes) → not technical, not αSMA.
- VSMC calibrator also higher in the milder group for MILD16 vs SEVERE16 (+1.16) and chronic vs RR PEAK1 (+0.91):
  piece-level channel brightness may contribute there; within-animal contrast removes it (MILD16/SEVERE16 +1.10;
  chronic vs RR +0.40 = weakest). MILD30/SEVERE30 and MONOPHASIC/REMISSION1 calibrators are flat.
- **Edge ring:** MILD and MONOPHASIC animals have a sharp vimentin peak in the first ~10 µm inside the lesion edge
  (tissue vimentin z ≈ 1.6 / 3.3), absent in SEVERE, REMISSION1 and PEAK; core vimentin also highest in MILD/MONOPHASIC.

## 2026-10-08 — Scar and containment; day ~30 vs chronic peak (`notebooks/16_scar_containment.ipynb`)
Collaborator's hypothesis: a successful astrocyte scar restricts immune infiltration. 419 lesion objects in 53 animals.
- **Border forms by d30 in milder animals only:** border tissue vimentin chronic PEAK1 0.04 → MILD16 0.97 (p = 0.02),
  SEVERE16 0.02; MONOPHASIC strongest at d31–33 (2.91 vs PEAK2 0.30); RR PEAK1 → MONOPHASIC p = 0.03.
- **Within animals, not containment-like:** better-bordered lesions have *more* immune cells just outside (escape
  ρ +0.3–0.45, p < 0.01). Surface lesions confound (glia limitans + meningeal entry; depth ρ −0.44 / −0.34), but deep
  lesions only and depth-adjusted: still ρ +0.25–0.31 (p 0.003–0.03). Reading: the vimentin border forms where immune
  cells are active at the lesion edge (a response); whether it later restricts spread needs time-resolved data.
- First leakage measure (outside ÷ inside) was confounded by lesions emptying while resolving — replaced by escape vs
  the animal's distant tissue.

### Correction (2026-10-08): lesion objects → lesion regions (notebook 16)
The user spotted that the "best vs worst bordered lesion" example looked alike. Checking showed the cell-linkage lesion
objects (cells < 30 µm apart) were often diffuse scatters without a real outline, and tissue-mask depth was wrong next
to roots/meninges. Redone with lesion **regions** (smoothed lesion-cell density on a 10 µm grid, threshold 0.5; 356
regions, median 96 % lesion cells inside; depth from the cord outline):
- **Withdrawn:** "better-bordered lesions have more immune cells around them" (now median ρ 0.00, p = 0.65; deep ρ −0.39
  n.s.) and "border builds by d30 in mild animals" (border vimentin PEAK1 0.67 → MILD16 0.50; SEVERE16 0.02).
  Containment is neither supported nor refuted.
- **Withdrawn:** notebook 15's "vimentin ring at the lesion edge". With region outlines there is no edge peak; vimentin
  rises on entering a lesion and stays high through its interior. The ring came from the cell-based edge distance
  ("just inside" = isolated lesion cells in healthy tissue).
- **Holds, now with regions:** astrocyte vimentin inside lesions minus outside: PEAK1 0.60; MILD16 2.27 vs SEVERE16 1.01;
  MILD30 3.30 vs SEVERE30 2.07; MONOPHASIC 4.28 vs REMISSION1 1.26 → a lesion-wide astrocyte response, low at peak,
  strongest in animals that recover.

## 2026-10-08 — Local barrier tests (`notebooks/17_local_barrier.ipynb`)
- **Border segments** (100 µm stretches of lesion-region edges, 3,188 in 392 lesions, 53 animals; within-lesion, adjusted
  for immune load inside, activity, depth): astrocyte vimentin vs immune share just outside, median per-animal ρ +0.035
  (95 % CI −0.016 to +0.073, p = 0.12); 30–60 µm outside +0.018. Territory vimentin +0.069 (p = 0.05; contaminated by
  leukocyte vimentin). Vimentin-rich stretches have more immune cells inside the edge, same outside.
- **Old lesions at relapse** (685 S2 patches, 51 lesions, 9 animals): astrocyte vimentin vs nearby active tissue ρ −0.02
  (p = 0.36); territory +0.16 (n.s.).
- **Answer:** no measurable barrier; vimentin is a marker of the resolution phase (lesion-wide, highest in animals that
  recover), not a physical restriction of infiltrating immune cells at this resolution.

## 2026-10-08 — How strong is the vimentin–outcome link? (`notebooks/18_vimentin_outcome.ipynb`)
35 post-peak animals; lesion astrocyte vimentin (inside − outside lesion regions, within-image z).
- vs clinical score: ρ −0.34 (p = 0.046); adjusted for days since first peak −0.44 (p = 0.008); leave-one-run-out
  −0.42 to −0.55 (all p < 0.03); within lesion states S0 −0.49, S2 −0.56, S1 −0.45, S4 −0.41 (p < 0.04).
- Weakened by lesion share (−0.26, n.s.) and first-attack height (−0.29, p = 0.09): entangled with disease burden.
- Within-image pairs: 12/19 concordant (63 %, p = 0.36); same-day 4/5.
- Runs 1–3 "non-replication" = no score variance (chronic post-peak in runs 1–3 = MILD30, all score 1.0); RR runs 1–3
  ρ −0.46 (p = 0.10).
- Beyond RNA: astrocyte RNA reactivity ρ +0.29 with score; vimentin | RNA reactivity ρ −0.63 (p < 0.001).
- Strength: moderate; hypothesis with decent support, not established.
- **VSMC check:** VSMC αSMA/Vim per animal also falls with score (ρ −0.36, p = 0.04) and tracks astrocyte vimentin
  (ρ +0.56) → part of the effect is channel-wide (staining/brightness or vascular biology). Astrocyte-specific remainder:
  ρ −0.32 adjusted for VSMC (p = 0.06); astrocyte − VSMC −0.36 (p = 0.03). Revised strength: weak to moderate.

## 2026-10-08 — T-cell 18S polarity: WITHDRAWN (`notebooks/19_tcell_polarity.ipynb`)
All runs: perivascular T cells' 18S centroid points at the nearest vessel cell (median cos 0.17, 98 % of 54 animals;
also B, DC, MDM, fibroblasts; not neurons) and much less at a same-distance non-vessel neighbour (0.05). But:
- DAPI "points" at vessels as strongly (0.14); relative to the cell's own nucleus the T-cell effect is 0.047 (controls −0.04).
- Brightness-matched control (non-vessel neighbour at the same distance, ≥ as bright in 18S): T cells point *more* at the
  bright neighbour (0.19) than at the vessel (0.08); vessel − bright < 0 in 76 % of animals (p = 4e-5); relative to the
  nucleus nothing vessel-specific remains (0.017 vs 0.046).
- → bleed from 18S-bright perivascular neighbours + outline geometry. The finding in notebooks 06, 07, 09, 13 and in the
  reports/deck is withdrawn. Lesson: polarity of small cells near bright neighbours needs brightness-matched and
  outline-free controls.
