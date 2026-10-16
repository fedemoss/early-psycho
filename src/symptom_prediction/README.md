# symptom_prediction

**Question:** can symptom severity (PANSS total, YMRS) be predicted from connectivity, model-derived or higher-order features under leak-free evaluation?

## `leakage_audit.py`

Re-examines the v1 headline model (`legacy/notebooks/panss_model_region_overfit.ipynb`): lag-1 FC edges with |corr(edge, PANSS)| > 0.2, selected on the full sample, then ridge with LOOCV (R² = 0.31).

| | R² (threshold 0.2) |
|---|---|
| v1, leaky selection | 0.31 |
| selection inside folds | −0.21 (−0.21 to −0.33 across thresholds) |
| leaky pipeline on shuffled PANSS | mean 0.15; 11% of shuffles reach ≥ 0.31 |

**Conclusion:** the v1 result was produced by leakage.

## `run.py`

Evaluates 13 feature sets for each target with `core.evaluation.evaluate`:
- **CV scheme:** repeated 10-fold CV. Standardisation, top-k selection and the ridge penalty are fit inside each training fold.
- **Group handling:** AP/NAP means are removed inside the fold.
- **Statistic:** within-group correlation between predictions and scores.
- **Null:** the full pipeline rerun on scores permuted within group (200 permutations).
- **Feature sets:**
  - FC edges, lag-1 FS edges and GEC edges.
  - FC network blocks (28).
  - O-information network compositions (168), alone and with FC blocks; nodal O-information (100).
  - FDT P_j plus nodal irreversibility.
  - Perturbation maps.
  - Prior-based sets: Mana restorative nodes (perturbation maps and incident FC edges), and Keane et al. 2025 somatomotor–visual FC edges.

```bash
python -m src.symptom_prediction.leakage_audit [--n-perm 200]
python -m src.symptom_prediction.run [--n-perm 200 --n-repeats 5]
```

**Outputs** (`data/derived/v2/symptom_prediction/`): `leakage_audit.csv`, `leakage_null.csv`, `model_comparison.csv`. Figures: `notebooks/05_symptom_prediction.ipynb`.

**Results (v2):**
- No feature set predicts PANSS total or YMRS beyond the AP/NAP difference (all p > 0.15). R² is never above the group-only model.
- Negative within-group r values are a known CV artefact (shrinkage toward the training mean). The permutation null shows the same bias, with a mean of about −0.25.

**Likely next steps:**
- Symptom factors instead of totals: Foster et al. 2025 predicted PANSS positive/negative but not total in early psychosis. This needs item-level PANSS from NDA collection #2914.
- An ID-aligned rebuild of the dataset from `data/raw/`.
