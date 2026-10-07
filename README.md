# Beyond Boundaries

What do the Xenium Multimodal Cell Segmentation stains (DAPI, ATP1A1/CD45/E-Cadherin, 18S rRNA, αSMA/Vimentin) add to
the transcriptome? Mouse EAE spinal cord (RRMAP2 runs 5/6; 18 sections, 25 animals, 500k annotated cells).

**Start here:** `report/BeyondBoundaries_conclusions.pdf` — the answer, conclusions and key figures (9 pages).
Lab-meeting deck: `report/slides/BeyondBoundaries_labmeeting.html` (PDF alongside). All findings in detail:
`notebooks/00_summary.ipynb` (also `report/BeyondBoundaries_summary.pdf`).
Every analysis notebook has **Finding** cells next to the plots that support them.

| notebook | content |
|---|---|
| `00_summary` | the whole story: bottom-line table, key figures and tables |
| `01_feature_overview` | per-cell features: coverage, concordance with Xenium, where each stain lives, cell gallery |
| `02_normalisation_qc` | tissue/background/autofluorescence maps, normalisation, piece-level intensity vs disease, spatial maps |
| `03_sanity_atlas` | expected biology tests, CD45 check, segmentation-method effects, cell type × lesion atlas |
| `04_orthogonality` | image ↔ transcriptome cross-prediction per cell type, residual coherence, lesion tests, clustering |
| `05_field_vs_cell_and_animal_level` | is the unexplained signal technical or biological; clinical-score prediction per animal; 18S texture |
| `06_targeted_readouts` | astrocyte vimentin vs RNA reactivity, leukocyte polarity vs vessels, 18S lesion vs physiological, neuropil loss |
| `07_visual_checks` | images behind the findings: perivascular T cells, protein-only astrocytes vs VSMC, 18S texture, crowding test |
| `08_image_only_prediction` | what images alone predict: cell type, subtypes, anatomy, lesion state + lesion maps, animal metadata (vs transcriptome) |
| `09_conclusions` | cross-run transfer test (train run5 → test run6 and back), one-figure summary, conclusions + how to use the stains |

Code: `src/beyondboundaries` (io, features, extract, background, orthogonality, data, plotting) with tests in `tests/`;
batch extraction `scripts/01_extract_features.py`; config `config.yaml`; env `environment.yml`.
Notebooks (`notebooks/*.ipynb`, executed, with outputs) are paired with jupytext percent scripts in `notebooks/py/`
(edit either; `jupytext --sync notebooks/NN_name.ipynb` keeps them in step). Executed on the analysis Mac (kernel `bb`).
Running log of what was run and key numbers: `RESULTS.md`. Figures (PDF + PNG): `results/<notebook>/`.
