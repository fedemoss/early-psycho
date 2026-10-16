"""
Hierarchy / non-equilibrium study.

Model-based: FDT deviation (Acero-Pousa et al. 2025): mean and SD of the perturbability map P_j, and
GEC asymmetry. Model-free: pairwise time-irreversibility (FS(lag) vs FS(lag)^T) and its spread
across nodes, lags 1-3 TR.

Outputs in data/derived/v2/hierarchy/:
  subject_measures.csv, group_tests.csv, group_tests_networks.csv,
  symptom_correlations.csv, irreversibility_adjusted.csv

Run from the repository root:  python -m src.hierarchy.run
"""
import numpy as np
import pandas as pd

from src.core import paths
from src.core.data import GROUPS, load_data, load_features, load_model_results
from src.core.features import LAGS
from src.core.measures import upper
from src.core.stats import fdr, group_tests, hedges_g, perm_F, symptom_correlations

MEASURES = (["Dhat", "sd_P", "GEC_asym", "GEC_strength", "FC_mean"]
            + [f"irr_lag{l}" for l in LAGS] + [f"irr_hier_lag{l}" for l in LAGS])


def subject_measures(res, feats, ts, scores):
    rows = []
    for g in GROUPS:
        for k in res[g]:
            r, f, C = res[g][k], feats[g][k], res[g][k]["GEC"]
            x = ts[g][k]
            row = {"group": g, "k": k, "Dhat": r["fdt_P"].mean(), "sd_P": r["fdt_P"].std(),
                   "GEC_asym": np.linalg.norm(C - C.T) / np.linalg.norm(C), "GEC_strength": C.sum() / len(C),
                   "FC_mean": upper(f["FC"]).mean(),
                   "lag1_autocorr": np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in x]),
                   "PANSS": scores[g]["PANSS"][k] if g != "HC" else np.nan,
                   "YMRS": scores[g]["YMRS"][k] if g != "HC" else np.nan}
            for l in LAGS:
                row[f"irr_lag{l}"] = f["irr_global"][l]
                row[f"irr_hier_lag{l}"] = f["irr_hier"][l]
            rows.append(row)
    return pd.DataFrame(rows)


def network_tests(res, feats, network):
    grp = np.array([g for g in GROUPS for _ in res[g]])
    rows = []
    for kind in ("fdt_P", "irr_lag2"):
        for n in np.unique(network):
            sel = network == n
            v = np.array([(res[g][k]["fdt_P"] if kind == "fdt_P" else feats[g][k]["irr_nodal"][2])[sel].mean()
                          for g in GROUPS for k in res[g]])
            f0, p = perm_F(v, grp, n_perm=2000)
            rows.append({"measure": kind, "network": n, **{f"mean_{g}": v[grp == g].mean() for g in GROUPS},
                         "F": f0, "p_perm": p, "g_AP_vs_HC": hedges_g(v[grp == "AP"], v[grp == "HC"]),
                         "g_NAP_vs_HC": hedges_g(v[grp == "NAP"], v[grp == "HC"])})
    out = pd.DataFrame(rows)
    out["q_FDR"] = fdr(out["p_perm"])
    return out


def adjusted_irreversibility(S, n_perm=5000, seed=0):
    """Group effect on log irreversibility after regressing out mean FC and lag-1 autocorrelation."""
    rows = []
    rng = np.random.default_rng(seed)
    g = S["group"].values
    X = np.c_[np.ones(len(S)), S["FC_mean"], S["lag1_autocorr"]]
    for l in LAGS:
        y = np.log(S[f"irr_lag{l}"].values)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        f0, p = perm_F(resid, g, n_perm, seed)
        rows.append({"lag": l, "F_adj": f0, "p_perm": p,
                     "g_AP_vs_HC_adj": hedges_g(resid[g == "AP"], resid[g == "HC"]),
                     "g_NAP_vs_HC_adj": hedges_g(resid[g == "NAP"], resid[g == "HC"]),
                     "g_NAP_vs_AP_adj": hedges_g(resid[g == "NAP"], resid[g == "AP"])})
    return pd.DataFrame(rows)


def main():
    out = paths.study_dir("hierarchy")
    ts, _, scores = load_data()
    mr = load_model_results()
    res, network = mr["results"], mr["atlas"]["network"]
    feats = load_features()
    S = subject_measures(res, feats, ts, scores)
    S.to_csv(out / "subject_measures.csv", index=False)
    G = group_tests(S, MEASURES)
    G.to_csv(out / "group_tests.csv", index=False)
    network_tests(res, feats, network).to_csv(out / "group_tests_networks.csv", index=False)
    symptom_correlations(S, MEASURES).to_csv(out / "symptom_correlations.csv", index=False)
    A = adjusted_irreversibility(S)
    A.to_csv(out / "irreversibility_adjusted.csv", index=False)
    print(G.round(4).to_string())
    print(A.round(4).to_string())


if __name__ == "__main__":
    main()
