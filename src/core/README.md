# core

Code shared by all studies. Nothing here writes results, except `features.py`.

| module | contents |
|---|---|
| `paths.py` | Data layout (`data/input`, `data/raw`, `data/derived/v2`), `study_dir(name)`, `EP_DATA_DIR` override. |
| `data.py` | `load_data()` returns timeseries, SC and PANSS/YMRS. Also `load_model_results()`, `load_features()`, group constants. |
| `atlas.py` | Schaefer-100 labels, networks, centroids and L/R homologue pairs. Mana et al. 2023 region sets, mapped via Harvard-Oxford overlap. |
| `hopf.py` | Linearised Hopf model: Jacobian, Lyapunov covariance, lagged covariance. Also `fit_gec`, `fdt_perturbability` (Acero-Pousa eq. 16) and `perturbation_maps`. |
| `measures.py` | Time-irreversibility, Gaussian-copula transform, closed-form triplet O-information, network-level summaries. |
| `stats.py` | Permutation F test, Hedges g, Benjamini-Hochberg FDR, partial Spearman with within-group permutations. |
| `evaluation.py` | Leak-free CV: in-fold selection and ridge, group means removed in-fold, within-group permutation null. |
| `features.py` | Computes and caches per-subject empirical features (FC, lag-1 FS, O-information, irreversibility, FC blocks). |

## Conventions

- Lagged FC: `FS_ij = cov(x_i(t), x_j(t+lag)) / sqrt(var_i var_j)`, for both the empirical data and the model.
- Model constants: `TR = 0.8`, `sigma = 0.01`, `a = -0.02`, `maxC = 0.1`, `omega = 2*pi*f_peak`.
- Perturbation maps are measured relative to the unperturbed model. A **higher** `p2h_*` means the perturbation moves the subject toward the healthy FC.
- O-information units are nats. Ω > 0 means redundancy-dominated; Ω < 0 means synergy-dominated.

Build the shared feature cache (after `src.gec_model.refit`):

```bash
python -m src.core.features
```
