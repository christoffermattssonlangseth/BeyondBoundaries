# Human MS / Baló spinal cord: what the segmentation stains add

Same questions as the mouse EAE analysis, on human tissue from the Netherlands Brain Bank: two MS spinal cord
sections (S07-256, S13-019) and two Baló's concentric sclerosis sections (Balo-1, Balo-2). Xenium 5K human panel
(hAtlas v1.1 + 100 custom genes), the same four segmentation stains (DAPI, ATP1A1/CD45/E-cadherin, 18S,
αSMA/vimentin), XOA 4.0.1, one run (2026-01-15).

**Data policy.** No raw or per-cell data goes to git: no bundles, transcripts, h5ad, or per-cell or per-transcript
tables (`human/data/` is ignored). Code, notebooks and figures may be committed.

Data on the analysis Mac: `/Volumes/moldiassd/human_MS_Balo/` (bundles + `processed_data/baloMS_clustered.h5ad`,
Leiden clusters from the BaloMS repo, no cell-type labels). Config: `human/config.yaml`.

## Known issue: out-of-focus holes
About 6 % of each Baló section lies in round no-transcript holes. Inside them all four stains are out of focus (no
nuclei in DAPI, no texture, a bright rim at the edge) and there are no transcripts or cells, while the post-Xenium
H&E shows intact tissue: air bubbles or local lifting of the tissue during the run. These areas are masked out;
they are not cell-free tissue.

## Plan
1. `scripts/01_label_clusters.py` — marker-based cell-type labels for the Leiden clusters.
2. Feature extraction with the mouse pipeline (`scripts/01_extract_features.py --config human/config.yaml`).
3. `notebooks/h01_*` — does CD45 work in human (boundary channel in *PTPRC*+ cells), stain sanity checks, Baló rings,
   RNA in and around cells.
