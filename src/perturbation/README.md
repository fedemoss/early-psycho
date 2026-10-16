# perturbation

**Question:** which node perturbations move a patient's model dynamics toward the healthy regime, and do these match the regions identified by Mana et al. (2023)? Their whole-brain modelling study of schizophrenia used the Desikan-Killiany atlas and bifurcation-parameter perturbations.

**Region priors** (Mana et al. 2023, mapped to Schaefer-100 by Harvard-Oxford overlap; `core.atlas.mana_region_sets`):
- **restorative** (perturbing these in the schizophrenia model restores healthy dynamics): medial/lateral OFC, pars orbitalis, posterior and isthmus cingulate, parahippocampal cortex. This gives 12 Schaefer nodes.
- **pathogenic** (perturbing these in the healthy model induces schizophrenia-like dynamics): postcentral, insula, inferior parietal, fusiform, middle temporal. This gives 18 nodes.

**Maps** are computed in `gec_model`: `p2h_noise`, `p2h_aneg` (toward healthy), `p2p_apos` (toward patients) and `p2o_noise`. A higher value is more toward the target; the best node is the `argmax`.

**Tests:**
- Mean within-subject percentile of the set's nodes, compared with 5000 random node sets of the same size.
- The best node per subject.
- Group tests and symptom correlations for map summaries (max; mean over the restorative set).

**Run:** `python -m src.perturbation.run`

**Outputs** (`data/derived/v2/perturbation/`): `mana_sets.csv`, `mana_tests.csv`, `best_nodes.csv`, `subject_measures.csv`, `group_tests.csv`, `symptom_correlations.csv`. Figures: `notebooks/03_perturbation.ipynb`.

**Results (v2):**
- **No enrichment of the restorative set** in patients' P2H maps (mean percentile 0.52, p = 0.34).
- **No enrichment of the pathogenic set** in controls' P2P maps (p = 0.40).
- The most frequent best P2H node in patients is `LH_Default_pCunPCC_1` (precuneus/PCC, 48 of 92 patients). It is adjacent to Mana's posterior cingulate but below the overlap threshold; this is a post-hoc observation.
- **Note:** v1 notebooks used `argmin(P2H)` as the "best" node, which actually selects the most damaging perturbation.
