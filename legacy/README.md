# legacy (v1)

The v1 notebooks and scripts (May–Sep 2026) are kept for reference. **They are superseded by `src/`** and should not be used for new results:

- They read paths like `./data/...` that no longer exist. Inputs now live in `data/input/` and v1 outputs in `data/derived/v1/`.
- They contain the issues fixed in v2 (see the main README):
  - ω without 2π;
  - GEC normalised to 1;
  - transposed FS target;
  - non-homologous "homotopic" pairs;
  - TR = 2.0 in `perturbation_panss_hypothesis.ipynb`;
  - the inverted P2H sign;
  - full-sample feature selection in the PANSS models.
- `notebooks/raw_models/` treats the AP/PA **phase-encoding** files (`hcpep_schaefer1000_REST*_AP/PA.mat`) as affective/non-affective classes. The same subjects therefore appear twice.

| file | what it was |
|---|---|
| `gec_and_pert_analysis.py` | Cluster script: GEC fit + perturbability/P2H/P2O maps |
| `Ceff_EP_HCP.m` | Original MATLAB GEC template (reference for `maxC = 0.1`) |
| `notebooks/testing.ipynb` | Conversion of the `.mat` inputs to `.npz` |
| `notebooks/gec_matrices.ipynb` | Pilot GEC fits (10 subjects/group, TR = 2.0) |
| `notebooks/output_study.ipynb` | Asymmetry, perturbability, P2H/P2O exploration |
| `notebooks/clinical_scores_*model.ipynb` | XGBoost PANSS models on summary features |
| `notebooks/panss_model_region_overfit.ipynb` | Edge-selection + ridge PANSS model (leaky R² = 0.31) |
| `notebooks/panss_model_sc_regions.ipynb` | SC-selected edges |
| `notebooks/perturbability_*.ipynb` | Perturbability vs PANSS / model error |
| `notebooks/perturbation_panss_hypothesis.ipynb` | Perturbation → predicted PANSS |
