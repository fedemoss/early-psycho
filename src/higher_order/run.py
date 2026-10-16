"""
Higher-order interactions study (Gaussian O-information, Rosas et al. 2019).

1. Empirical: per subject, O-information of all 161,700 region triplets (Gaussian copula estimator),
   summarised globally, per network composition (84) and per node. Group tests and symptom correlations.
2. Model vs empirical: triplet O-information of each subject's GEC-simulated FC vs their empirical FC
   (both Pearson-based). A non-negative pairwise-coupled linear model is expected to be unable to
   produce synergy-dominated triplets (Caprioglio, Mediano & Berthouze 2026, PRL).

Outputs in data/derived/v2/higher_order/:
  subject_measures.csv, group_tests.csv, group_tests_compositions.csv, symptom_correlations.csv,
  model_vs_empirical.csv

Run from the repository root:  python -m src.higher_order.run
"""
import numpy as np
import pandas as pd

from src.core import paths
from src.core.data import GROUPS, load_data, load_features, load_model_results
from src.core.measures import triplet_oinfo
from src.core.stats import fdr, group_tests, hedges_g, perm_F, symptom_correlations

MEASURES = ["Omega_mean", "syn_frac", "TC_mean", "Omega_mean_pearson", "syn_frac_pearson"]


def main():
    out = paths.study_dir("higher_order")
    _, _, scores = load_data()
    feats = load_features()
    res = load_model_results()["results"]

    S = pd.DataFrame([{"group": g, "k": k,
                       "Omega_mean": f["oinfo"]["global_mean"], "syn_frac": f["oinfo"]["global_syn"],
                       "TC_mean": f["oinfo"]["global_tc"],
                       "Omega_mean_pearson": f["oinfo_pearson"]["global_mean"],
                       "syn_frac_pearson": f["oinfo_pearson"]["global_syn"],
                       "PANSS": scores[g]["PANSS"][k] if g != "HC" else np.nan,
                       "YMRS": scores[g]["YMRS"][k] if g != "HC" else np.nan}
                      for g in GROUPS for k, f in feats[g].items()])
    S.to_csv(out / "subject_measures.csv", index=False)
    G = group_tests(S, MEASURES)
    G.to_csv(out / "group_tests.csv", index=False)
    symptom_correlations(S, MEASURES).to_csv(out / "symptom_correlations.csv", index=False)

    # per network composition
    grp = S["group"].values
    names = feats["HC"][0]["oinfo"]["comp_names"]
    comp = np.array([feats[g][k]["oinfo"]["comp_mean"] for g in GROUPS for k in feats[g]])
    rows = []
    for c, name in enumerate(names):
        v = comp[:, c]
        f0, p = perm_F(v, grp, n_perm=2000)
        rows.append({"composition": name, **{f"mean_{g}": v[grp == g].mean() for g in GROUPS}, "p_perm": p,
                     "g_AP_vs_HC": hedges_g(v[grp == "AP"], v[grp == "HC"]),
                     "g_NAP_vs_HC": hedges_g(v[grp == "NAP"], v[grp == "HC"])})
    Gc = pd.DataFrame(rows)
    Gc["q_FDR"] = fdr(Gc["p_perm"])
    Gc.to_csv(out / "group_tests_compositions.csv", index=False)

    # model vs empirical
    rows = []
    for g in GROUPS:
        for k in res[g]:
            om_e, _ = triplet_oinfo(feats[g][k]["FC"])
            om_m, _ = triplet_oinfo(res[g][k]["FC_sim"])
            rows.append({"group": g, "k": k, "Omega_mean_emp": om_e.mean(), "Omega_mean_model": om_m.mean(),
                         "syn_frac_emp": (om_e < 0).mean(), "syn_frac_model": (om_m < 0).mean(),
                         "r_triplets_emp_model": np.corrcoef(om_e, om_m)[0, 1]})
    pd.DataFrame(rows).to_csv(out / "model_vs_empirical.csv", index=False)
    print(G.round(4).to_string())
    print(pd.DataFrame(rows).groupby("group").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
