# higher_order

**Question:** do early-psychosis patients differ in higher-order statistical interactions, and can the linear Hopf model reproduce them?
- Background: synergy is lower and redundancy higher in schizophrenia (Tarchi et al. 2026; Li et al. 2025), and ketamine increases redundancy (Herzog et al. 2024).

**Method:**
- **Measure:** Gaussian O-information (Rosas et al. 2019) for all 161,700 region triplets, computed in closed form from the correlation matrix. Ω = TC − DTC: Ω > 0 is redundancy-dominated, Ω < 0 is synergy-dominated.
- **Estimators:** a Gaussian-copula estimator (Ince et al. 2017), with Pearson as a check.
- **Per-subject summaries:** global mean Ω, fraction of synergistic triplets, mean total correlation (TC), mean Ω per network composition (84 multisets of 3 Yeo networks), and nodal Ω.
- **Model vs empirical:** triplet Ω of the subject's GEC-simulated FC compared with their empirical FC.
- **Conceptual note:**
  - Under a Gaussian, every O-information quantity is a function of the covariance, so it adds no information beyond pairwise statistics. What it adds is a nonlinear multivariate summary.
  - Fair comparisons are therefore against matched pairwise summaries (FC network blocks; see `symptom_prediction`).
  - Pairwise-coupled linear models need sign-mixed couplings to produce synergy (Caprioglio, Mediano & Berthouze 2026, PRL).

**Run:** `python -m src.higher_order.run`

**Outputs** (`data/derived/v2/higher_order/`): `subject_measures.csv`, `group_tests.csv`, `group_tests_compositions.csv`, `symptom_correlations.csv`, `model_vs_empirical.csv`. Figures: `notebooks/04_higher_order.ipynb`.

**Results (v2):**
- Patients trend toward less redundancy and more synergy than controls: mean Ω g ≈ −0.4 and fraction synergistic g ≈ +0.4, p ≈ 0.07–0.10. This partly overlaps with their lower mean FC.
- **No correlation with PANSS or YMRS.**
- **The GEC model produces 0% synergistic triplets**, against about 22–28% empirically. Triplet Ω correlates with the empirical values at r ≈ 0.66 within subject (Pearson FC).

**Next steps:**
- Dynamical (lagged) O-information from the lag-1 covariance, linking to the irreversibility result.
- A GEC fit that allows negative couplings, so that model-based synergy, and "perturbation-to-healthy" measured in O-information space, become possible.
- Santoro-style triangle higher-order connectomics as a non-Gaussian complement.
