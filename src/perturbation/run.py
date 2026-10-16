"""
In-silico perturbation study: P2H / P2O maps and the Mana et al. (2023) region priors.

Tests whether the Mana 'restorative' nodes (OFC/ventral PFC, posterior cingulate, parahippocampal)
rank high in each patient's P2H maps, and whether the 'pathogenic' nodes rank high in each control's
P2P (move-toward-patient) map. Statistic: mean within-subject percentile of the set's nodes, against
random node sets of the same size.

Outputs in data/derived/v2/perturbation/:
  mana_sets.csv, mana_tests.csv, best_nodes.csv, subject_measures.csv, group_tests.csv, symptom_correlations.csv

Run from the repository root:  python -m src.perturbation.run
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.core import paths
from src.core.atlas import mana_region_sets
from src.core.data import GROUPS, PATIENTS, load_data, load_model_results
from src.core.stats import group_tests, symptom_correlations

MAPS = ("p2h_noise", "p2o_noise", "p2h_aneg", "p2p_apos")
MEASURES = ["p2h_noise_max", "p2h_aneg_max", "p2o_noise_max", "p2p_apos_max",
            "p2h_noise_mana_rest", "p2h_aneg_mana_rest"]


def set_percentile_test(maps, node_set, n_rand=5000, seed=0):
    pct = np.array([(rankdata(m) - 1) / (len(m) - 1) for m in maps])  # 1 = highest value
    obs = pct[:, node_set].mean()
    rng = np.random.default_rng(seed)
    null = np.array([pct[:, rng.choice(pct.shape[1], len(node_set), replace=False)].mean() for _ in range(n_rand)])
    return obs, (np.sum(null >= obs) + 1) / (n_rand + 1)


def main():
    out = paths.study_dir("perturbation")
    _, _, scores = load_data()
    mr = load_model_results()
    res, labels = mr["results"], mr["atlas"]["labels"]
    rest, path = mana_region_sets()
    pd.DataFrame([{"set": s, "node": int(i), "label": labels[i]} for s, idx in (("restorative", rest), ("pathogenic", path))
                  for i in idx]).to_csv(out / "mana_sets.csv", index=False)

    rows = []
    for mapname, node_set, setname in (("p2h_noise", rest, "restorative"), ("p2h_aneg", rest, "restorative"),
                                       ("p2p_apos", path, "pathogenic"), ("p2h_noise", path, "pathogenic"),
                                       ("p2h_aneg", path, "pathogenic")):
        for who in (PATIENTS, ("HC",)):
            maps = [res[g][k][mapname] for g in who for k in res[g]]
            obs, p = set_percentile_test(maps, node_set)
            rows.append({"map": mapname, "set": setname, "subjects": "+".join(who),
                         "mean_percentile": obs, "p_vs_random_sets": p})
    pd.DataFrame(rows).to_csv(out / "mana_tests.csv", index=False)

    best = []
    for mapname in ("p2h_noise", "p2h_aneg", "p2p_apos"):
        for g in GROUPS:
            for k in res[g]:
                j = int(np.argmax(res[g][k][mapname]))
                best.append({"map": mapname, "group": g, "k": k, "best_node": j, "best_label": labels[j],
                             "in_mana_restorative": j in rest, "in_mana_pathogenic": j in path})
    pd.DataFrame(best).to_csv(out / "best_nodes.csv", index=False)

    S = pd.DataFrame([{"group": g, "k": k,
                       **{f"{m}_max": res[g][k][m].max() for m in MAPS},
                       "p2h_noise_mana_rest": res[g][k]["p2h_noise"][rest].mean(),
                       "p2h_aneg_mana_rest": res[g][k]["p2h_aneg"][rest].mean(),
                       "PANSS": scores[g]["PANSS"][k] if g != "HC" else np.nan,
                       "YMRS": scores[g]["YMRS"][k] if g != "HC" else np.nan}
                      for g in GROUPS for k in res[g]])
    S.to_csv(out / "subject_measures.csv", index=False)
    group_tests(S, MEASURES).to_csv(out / "group_tests.csv", index=False)
    symptom_correlations(S, MEASURES).to_csv(out / "symptom_correlations.csv", index=False)
    print(pd.read_csv(out / "mana_tests.csv").round(4).to_string())


if __name__ == "__main__":
    main()
