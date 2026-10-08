# Beyond Boundaries — verdict

**Do the Xenium Multimodal Cell Segmentation images add anything beyond segmentation?**
Mouse EAE spinal cord, RRMAP2 runs 1–6: 54 images, 67 animals, 1.38 M cells. Status 8 October 2026; details in
`BeyondBoundaries_methods_results.pdf` (methods + all numbers) and `BeyondBoundaries_lesion_story.pdf` (biology).

> **A modest yes.** The images are first of all a segmentation aid. Beyond that they give a small number of robust
> readouts the 5K panel cannot. Their pooled channels and optical bleed limit them as a protein assay, so specific
> protein questions need dedicated stains.

## What holds (replicated across runs, with controls)

| Readout | Evidence | Use |
|---|---|---|
| **Segmentation** (the kit's purpose) | 18S interior stain draws ~95 % of cell masks | real; not benchmarked against alternatives here |
| **Neuropil loss at the lesion's grey/white-matter border** (ATP1A1 around each cell; *Atp1a1* not on the panel) | ~15–20 % less ATP1A1 in lesions within ~75 µm of grey matter than in healthy WM at the same distance, both run sets (×0.78, 25/31 animals; ×0.83, 13/18); **none deeper in white matter** | tissue damage the panel can't show directly; matched by lower neuron-derived RNA between cells at the same place (notebook 23). **Corrected 8 October:** the earlier "~17 % white-matter loss" was mostly the ATP1A1 gradient away from grey matter (lesions sit deeper in WM; notebook 20, section 9) |
| **Astrocyte reactive RNA without vimentin protein in active disease**, consistent with RNA preceding protein (stages from different animals) (*Vim* not on the panel) | protein vs RNA ρ 0.46–0.58 per run; "RNA-only" astrocytes enriched in active disease in both run sets | the clearest protein-vs-RNA insight |
| **Images alone map lesions and cell types on unseen runs** | lesion AUROC 0.85–0.91, cell type 0.42–0.47 (chance 0.07), leave-one-run-out over 5 runs | QC and annotation cross-check; redundant with RNA for biology |

## What is weak or partial

- **Lesion vimentin and outcome:** low at peak, fills lesion interiors, tends to be higher in animals that do well
  (ρ ≈ −0.45 after time adjustment). But part of it is shared with VSMCs in the same channel (astrocyte-specific
  ρ ≈ −0.33), it is entangled with disease burden, and the same-image test is weak (12/19 pairs). **Weak to moderate.**
- **18S texture as a severity marker:** replicates in astrocytes, myeloid cells and oligodendrocytes only.
- **18S gain in activated lesion cells beyond RNA content:** plausible, not stress-tested.

## What does not work, or was withdrawn after controls

- **CD45** is undetectable in mouse spinal cord; the boundary channel reports ATP1A1 only.
- **No hidden cell states:** ~85 % of image variance isn't explained by RNA, but it is mostly a local optical/staining field.
- **Images don't improve lesion identity or severity prediction** over the transcriptome.
- **Narrowed:** "white-matter neuropil loss" holds only at the grey/white-matter border (see above).
- **Withdrawn (notebook 23):** nuclear RNA retention in lesion oligodendrocytes (a white/grey-matter mix effect).
- **Withdrawn:** T cells orienting 18S towards vessels (bleed from bright neighbours + outline geometry); a vimentin
  ring at lesion edges and vimentin-border containment (artefacts of a lesion-object definition). No evidence that
  vimentin is a barrier to immune cells (within-lesion test ρ +0.035, 95 % CI −0.016 to +0.073).

## Why: three structural limits

1. **Pooled channels** (αSMA + vimentin; ATP1A1 + CD45 + E-cadherin): every protein claim carries "or the other one".
2. **Optical bleed and outline geometry:** small cells next to bright neighbours borrow their signal; polarity and
   "around the cell" measures need brightness-matched and nucleus-relative controls.
3. **The stains were chosen to draw cell boundaries**, not to answer biological questions.

## Recommendations

- **Keep the kit** for segmentation; use border neuropil loss and astrocyte vimentin state as add-on readouts, with the caveats above.
- **For protein biology:** post-Xenium immunofluorescence on the same sections with single-target antibodies (vimentin,
  αSMA, GFAP; a mouse-reactive CD45). The most informative sections: MILD16 vs SEVERE16 (they share images) and
  MONOPHASIC vs REMISSION1.
- **Future panels:** add *Vim*, *Acta2*, *Gfap*, *Atp1a1* as custom genes.
- **Worth writing up as methods:** control-referenced lesion definitions; lesion regions instead of linked cells;
  brightness-matched and nucleus-relative controls for polarity.

## The bigger contribution may be the disease biology (transcriptome-led, all five runs)

Lesions defined against controls (controls ~1 %, not 4–20 %) resolve along a clear course: active monocyte-derived
infiltrate → fibrotic, demyelinated and glial-reactive tissue. Chronic severity is set at the first attack and means
lesions that stayed active. Relapse can't be predicted clinically; B cells accumulate after the first attack in
meningeal aggregates (fewer in never-relapsing animals), and relapses are mostly new lesions.

**For discussion:** (1) the paper's framing, disease biology or kit evaluation; (2) whether the slides are available
for post-Xenium immunofluorescence; (3) which open threads to close first (B cells and relapse with more REMISSION1-stage
animals; vimentin with separate antibodies).
