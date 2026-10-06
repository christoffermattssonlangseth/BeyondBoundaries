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
