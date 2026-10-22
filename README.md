# early-psycho

Whole-brain modelling of resting-state fMRI in early psychosis (HCP for Early Psychosis). We fit a
per-subject linearised Hopf (Stuart-Landau) model to get each subject's Generative Effective
Connectivity (GEC), then use it to study:

- **hierarchy and non-equilibrium dynamics:** FDT deviation and time-irreversibility;
- **in-silico perturbations:** which node perturbations move a patient's dynamics toward the healthy
  regime, tested against Mana et al. 2023;
- **higher-order interactions:** Gaussian O-information, comparing synergy and redundancy;
- **symptom prediction:** PANSS and YMRS, using leak-free evaluation.

## Data

HCP-EP subset, Schaefer-100 parcellation (7 Yeo networks), TR = 0.8 s, 1581 frames per subject (4 concatenated runs, band-pass 0.008–0.08 Hz).

| group | n | clinical scores |
|---|---|---|
| HC (healthy controls) | 48 | none |
| AP (affective psychosis) | 25 | PANSS total, YMRS |
| NAP (non-affective psychosis) | 76 | PANSS total, YMRS |

PANSS total is available for 92 patients and YMRS for 98. The structural connectome is a group-average SC on the same parcellation.

Data are **not** in the repository (`data/` is gitignored). Expected layout:

```
data/
├── input/                       preprocessed inputs used by src/
│   ├── HCP_early_psychosis_timeseries_cleaned.npz   ts_hc, ts_ap, ts_nap  (n, 100, 1581)
│   ├── HCP_early_psychosis_clinical_scores.npz      panss_AP/NAP, ymrs_AP/NAP
│   ├── Structural_Connectomes.npz                   SC, SC_D  (100, 100)
│   └── *.mat                                        original MATLAB files these were built from
├── raw/                         HCP-EP Schaefer-1000 runs with subject IDs + clinical tables
└── derived/
    ├── v1/                      results of the legacy pipeline (see legacy/)
    └── v2/<study>/              outputs written by src/
```

Set `EP_DATA_DIR=/path/to/data` to use a different location.

## Repository layout

```
src/
├── core/                 shared code: paths, data loading, atlas, Hopf model, measures, statistics, evaluation
├── gec_model/            fit the GEC and the perturbation maps for every subject
├── hierarchy/            FDT hierarchy and time-irreversibility: group tests, symptom correlations
├── perturbation/         P2H/P2O maps and the Mana et al. 2023 region priors
├── higher_order/         O-information (synergy/redundancy): group tests, model vs empirical
└── symptom_prediction/   leakage audit of the v1 model + leak-free models for PANSS / YMRS
notebooks/                one visualisation notebook per study (reads data/derived/v2)
legacy/                   v1 notebooks and scripts (superseded; kept for reference)
docs/                     progress slides
```

Each `src/<study>/README.md` describes that study's question, method, outputs and current results.

## Setup

```bash
conda create -n early-psycho python=3.14
conda activate early-psycho
pip install -r requirements.txt
python -m ipykernel install --user --name early-psycho
```

## Running the pipeline

Run everything from the repository root, as modules:

```bash
export OMP_NUM_THREADS=1                      # one BLAS thread per joblib worker

python -m src.gec_model.refit                 # ~20 min on 8 cores -> derived/v2/gec_model/model_results.pkl
python -m src.core.features                   # ~1 min  -> derived/v2/features/empirical_features.pkl
python -m src.hierarchy.run
python -m src.perturbation.run
python -m src.higher_order.run
python -m src.symptom_prediction.leakage_audit
python -m src.symptom_prediction.run          # ~20-30 min (permutation tests)
```

Then open the notebooks in `notebooks/` for figures.

## Main results (v2, 2026-10)

- **Model fit:** GEC fits improve from FC r ≈ 0.61 (v1) to 0.77 after the fixes below.
- **Time-irreversibility is lower in patients**, in every Yeo network and at lags 1–3 TR:
  - NAP vs HC: Hedges g ≈ −0.9; AP vs HC: g ≈ −0.6; FDR q &lt; 0.001.
  - Still present after adjusting for mean FC and autocorrelation (NAP g ≈ −0.65, AP g ≈ −0.49).
  - Model GEC asymmetry tracks it (ρ = 0.91).
  - The direction is opposite to Acero-Pousa et al. 2025, who found more hierarchy in chronic schizophrenia. Motion and medication were not available to control for.
- **No group difference in the FDT deviation (Acero-Pousa "hierarchy").**
- **Higher-order trend:** patients look less redundancy-dominated and more synergistic (g ≈ 0.4, p ≈ 0.07–0.10). A non-negative GEC model produces no synergistic triplets, whereas about 25% of empirical triplets are synergistic.
- **Mana et al. 2023 priors:** no enrichment of the restorative or pathogenic sets in the perturbation maps.
- **Symptom prediction:** no feature set predicts PANSS total or YMRS under leak-free evaluation.
  - The v1 result of R² = 0.31 came from feature-selection leakage: R² is below 0 with in-fold selection, and the leaky pipeline gives R² ≥ 0.31 in 11% of permutations.

## Fixes relative to v1 (`legacy/`)

1. Hopf Jacobian uses ω = 2π·f (v1 used f in Hz).
2. GEC rescaled to max 0.1 (as in `legacy/Ceff_EP_HCP.m`), not to 1.
3. The empirical lag-1 FS target had the transposed orientation of the model's `expm(τA)K`.
4. Homotopic connections now link true L/R homologues; in Schaefer ordering, region `i` and `i+50` are not homologues.
5. TR = 0.8 s everywhere (one v1 notebook used 2.0).
6. P2H sign: the best perturbation is `argmax(P2H)`. v1 used `argmin`, which selects the most damaging node.
7. Symptom models: selection and regularisation inside CV folds, with a full-pipeline permutation null.

## Key references

- Deco, Kringelbach et al.: GEC and FDT/perturbability framework (e.g. Phys Rev E 2023, doi:10.1103/physreve.108.064410).
- Acero-Pousa et al. 2025, *Transl Psychiatry*: reconfiguration of functional brain hierarchy in schizophrenia.
- Mana et al. 2023, *Cereb Cortex* 33:7642: in-silico perturbational approach in schizophrenia.
- Rosas et al. 2019, *Phys Rev E* 100:032305: O-information.
- Jacobs et al. 2024, *Schizophr Bull*: Human Connectome Project for Early Psychosis.
