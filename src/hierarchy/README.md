# hierarchy

**Question:** do early-psychosis patients differ from controls in hierarchical, non-equilibrium brain dynamics, and does this track symptoms? Acero-Pousa et al. (2025) report more hierarchy in chronic schizophrenia, correlating with negative symptoms and thought disorder.

**Measures:**
- **Model-based:**
  - `Dhat`: mean FDT perturbability P_j, the FDT deviation (Acero-Pousa eq. 16–17).
  - `sd_P`: spread of P_j across nodes.
  - `GEC_asym`: ‖C − Cᵀ‖ / ‖C‖.
  - `GEC_strength`: total coupling.
- **Model-free:**
  - `irr_lagL`: mean pairwise time-irreversibility (FS(L) − FS(L)ᵀ)², for L = 1–3 TR.
  - `irr_hier_lagL`: SD of nodal irreversibility.
- **Statistics:**
  - Group differences: permutation F test (5000 permutations), Hedges g, FDR. Also run per Yeo network.
  - Symptoms: partial Spearman with PANSS and YMRS, controlling for AP/NAP.
  - Control analysis: irreversibility adjusted for mean FC and lag-1 autocorrelation.

**Run:** `python -m src.hierarchy.run` (needs `gec_model` and `core.features` outputs)

**Outputs** (`data/derived/v2/hierarchy/`): `subject_measures.csv`, `group_tests.csv`, `group_tests_networks.csv`, `symptom_correlations.csv`, `irreversibility_adjusted.csv`. Figures: `notebooks/02_hierarchy.ipynb`.

**Results (v2):**
- **Irreversibility is lower in patients:**
  - NAP vs HC g ≈ −0.89, AP vs HC g ≈ −0.60, q &lt; 0.001.
  - Present at all lags and in all 7 networks.
  - Still present after adjusting for mean FC and autocorrelation (g ≈ −0.6).
- **GEC asymmetry is lower too** (g −0.5 / −0.77) and tracks empirical irreversibility (ρ = 0.91).
- **FDT deviation (`Dhat`) shows no group difference.**
- **No symptom correlation survives FDR.** The strongest is `Dhat` vs YMRS, ρ = 0.20, p = 0.055.
- **Caveats:**
  - The direction is opposite to Acero-Pousa, who studied chronic patients.
  - Motion and medication are not available, so they could not be controlled for.
