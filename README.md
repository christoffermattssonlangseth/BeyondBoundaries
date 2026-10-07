# Beyond Boundaries

What do the Xenium Multimodal Cell Segmentation stains (DAPI, ATP1A1/CD45/E-Cadherin, 18S rRNA, αSMA/Vimentin) add to
the transcriptome? Mouse EAE spinal cord (RRMAP2 runs 5/6; 18 sections, 25 animals, 500k annotated cells).

**Start here:** `notebooks/00_summary.ipynb` (also `report/BeyondBoundaries_summary.pdf`) — all findings with figures.
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

Code: `src/beyondboundaries` (io, features, extract, background, orthogonality, data, plotting) with tests in `tests/`;
batch extraction `scripts/01_extract_features.py`; config `config.yaml`; env `environment.yml`.
Notebooks are authored as jupytext percent scripts (`notebooks/*.py`) and executed on the analysis Mac (kernel `bb`).
Running log of what was run and key numbers: `RESULTS.md`. Figures (PDF + PNG): `results/<notebook>/`.
