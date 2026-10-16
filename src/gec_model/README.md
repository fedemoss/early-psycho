# gec_model

**Question:** fit a whole-brain model to every subject that reproduces their FC and lag-1 FC, giving per-subject effective connectivity and a basis for in-silico perturbation.

**Method:**
- Linearised Hopf (Stuart-Landau) network with a = −0.02, σ = 0.01 and node frequencies from the empirical peak in 0.008–0.08 Hz.
- The GEC starts from the group SC (scaled to max 0.1).
- It is updated by the Deco/Kringelbach pseudo-gradient rule toward the empirical FC and lag-1 FS. Only SC-connected pairs and L/R homologues are updated.
- For each subject, `refit.py` also computes:
  - `fdt_P`: FDT perturbability map (Acero-Pousa et al. 2025).
  - `p2h_noise`, `p2o_noise`: extra noise at node j; change in similarity to the healthy mean FC, and shift from baseline.
  - `p2h_aneg`: a_j lowered by 0.02 (Mana "noise" protocol), change in similarity to healthy.
  - `p2p_apos`: a_j raised by 0.01 (Mana "synchronisation" protocol), change in similarity to the patient mean FC.

Healthy and patient reference FCs are leave-one-out for subjects that belong to them.

**Run** (about 20 min on 8 cores):

```bash
OMP_NUM_THREADS=1 python -m src.gec_model.refit [--n-jobs 8]
```

**Output:** `data/derived/v2/gec_model/model_results.pkl`, a dict with:
- `results[group][k]`, holding `GEC, omega, FC_sim, FS_sim, fit_FC, fit_FS, n_iter, fdt_P, p2h_noise, p2o_noise, p2h_aneg, p2p_apos`;
- `atlas`;
- `settings`.

**Results:** fit r(FC) = 0.77 ± 0.03 and r(FS) = 0.75 ± 0.04, similar across groups. The v1 pipeline gave 0.61 / 0.59. Visualised in `notebooks/01_gec_model.ipynb`.
