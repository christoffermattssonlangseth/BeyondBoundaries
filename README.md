# Beyond Boundaries

What do the Xenium Multimodal Cell Segmentation stains (DAPI, ATP1A1/CD45/E-Cadherin, 18S rRNA, αSMA/Vimentin) add to
the transcriptome? Mouse EAE spinal cord (RRMAP2 runs 1–6: 54 sections, 67 animals, 1.38 M annotated cells; notebooks 01–09 use runs 5/6).

## The answer

> **Beyond segmentation, the kit gives us per-cell protein for genes the panel lacks (vimentin, αSMA, ATP1A1) and
> the tissue context around each cell, and the images alone can map lesions and cell types in a new run.
> It reveals no hidden cell states and no CD45.**

| ✅ What it adds | ❌ What it does not add |
|---|---|
| **Protein the panel can't measure:** *Vim*, *Acta2*, *Atp1a1* aren't on the 5K panel. Example: reactive astrocyte RNA *without* vimentin protein is enriched in active disease ("RNA-only" astrocytes 20 % at peak vs 3.5 % in CFA), consistent with RNA preceding protein (inferred from animals sacrificed at successive stages) | **CD45:** undetectable in mouse spinal cord; the boundary channel is ATP1A1 only |
| **Tissue context:** dense white-matter lesions have ~17 % less ATP1A1 neuropil around each cell than nearby healthy white matter (×0.83, 98 % of ~103 pieces, all runs; piece-wide ×0.79 in runs 5/6, 38 pieces, ×0.72 in runs 1–3, 58 pieces). Images add anatomy to the transcriptome (0.68 → 0.73; pooled within-run CV, optimistic, no cross-run estimate) | **Hidden cell states:** ~85 % of the image isn't explained by RNA, but that is mostly a local optical/staining field |
| **Images alone, on an unseen run** (trained on the other runs, leave-one-run-out over 5 runs): cell type 42–47 % (14 types, chance 7 %); lesion maps AUROC 0.88–0.89. *Optimistic pooled within-run CV (runs 5/6, grouped by animal; animals are nested in runs, so run is not held out): 46 % and 0.90* | **Better lesion calls or severity than RNA:** images are informative but redundant (clinical score ρ 0.62 vs 0.79; pooled per-animal CV across runs 5/6, optimistic) |

Animals are nested within imaging runs (each animal in exactly one run; `scripts/02_nesting_audit.py`), so headline
prediction numbers are cross-run; pooled animal-grouped numbers are kept as optimistic secondary values.

**Verdict (one page):** `report/BeyondBoundaries_verdict.pdf` — do the segmentation images add anything, what holds, what was withdrawn.
**Start here:** `report/BeyondBoundaries_conclusions.pdf` — what the kit adds (9 pages) · `report/BeyondBoundaries_lesion_story.pdf`
— disease biology across all five runs (9 pages) · `report/BeyondBoundaries_methods_results.{md,pdf}` — full Methods and
Results (editable Markdown).
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
| `07_visual_checks` | images behind the findings: perivascular T cells (orientation later withdrawn, see 19), protein-only astrocytes vs VSMC, 18S texture, crowding test |
| `08_image_only_prediction` | what images alone predict: cell type, subtypes, anatomy, lesion state + lesion maps, animal metadata (vs transcriptome) |
| `09_conclusions` | cross-run transfer test (train run5 → test run6 and back), one-figure summary, conclusions + how to use the stains |
| `10_lesion_states` | lesions redefined against control tissue (all 5 runs), six lesion states, trajectories along both disease courses |
| `11_disease_courses` | chronic mild vs severe, relapse vs monophasic, B cells, chronic vs RR, damage memory (all runs incl. images) |
| `12_lesion_story` | **illustrated story of 10–16**: tissue maps, microscopy, galleries next to every finding |
| `13_replication_runs123` | do the earlier image findings replicate in runs 1–3 (and leave-one-run-out models) |
| `14_relapse_lesion_origin` | relapse: new lesions or reactivated old ones; B-cell aggregates |
| `15_vimentin_robustness` | vimentin effect vs batch, brightness, focus, αSMA; vimentin ring at the lesion edge |
| `16_scar_containment` | lesion regions (real outlines); where vimentin sits; does the scar contain lesions (lesion level) |
| `17_local_barrier` | within-lesion barrier tests: edge stretches vs immune escape; do scarred old lesions stay quiet at relapse |
| `18_vimentin_outcome` | how strong is the vimentin–outcome link: adjusted models, leave-one-run-out, within lesion states, beyond RNA, within-image pairs |
| `19_tcell_polarity` | T-cell 18S "orientation towards vessels" tested with outline-free and brightness-matched controls: **an imaging artefact (withdrawn)** |
| `20_neuropil_loss` | white-matter neuropil loss stress test: crowding, composition, tissue surface, channel controls; microscopy — **holds** |

Code: `src/beyondboundaries` (io, features, extract, background, orthogonality, data, plotting) with tests in `tests/`;
batch extraction `scripts/01_extract_features.py`; config `config.yaml`; env `environment.yml`.
Notebooks (`notebooks/*.ipynb`, executed, with outputs) are paired with jupytext percent scripts in `notebooks/py/`
(edit either; `jupytext --sync notebooks/NN_name.ipynb` keeps them in step). Executed on the analysis Mac (kernel `bb`).
Running log of what was run and key numbers: `RESULTS.md`. Figures (PDF + PNG): `results/<notebook>/`.
