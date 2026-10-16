"""
Leak-free symptom models (PANSS total, YMRS) for many feature sets, evaluated with
src.core.evaluation.evaluate (repeated 10-fold CV, in-fold selection, within-group permutation null).

Output: data/derived/v2/symptom_prediction/model_comparison.csv
Run from the repository root (~20-30 min on 8 cores):
    python -m src.symptom_prediction.run [--n-perm 200]
"""
import argparse
import numpy as np
import pandas as pd

from src.core import paths
from src.core.atlas import mana_region_sets
from src.core.data import PATIENTS, load_data, load_features, load_model_results
from src.core.evaluation import evaluate
from src.core.measures import offdiag, upper


def feature_sets(pt, res, feats, network, mana_rest):
    nz = np.array([res[g][k]["GEC"] for g, k in pt]).mean(0) > 0
    triu = np.triu(np.ones((100, 100), bool), 1)
    rest_edges = np.zeros((100, 100), bool)
    rest_edges[mana_rest, :] = rest_edges[:, mana_rest] = True
    somvis = np.zeros((100, 100), bool)
    somvis[np.ix_(network == "SomMot", network == "Vis")] = True
    somvis |= somvis.T

    def stack(fn):
        return np.array([fn(g, k) for g, k in pt])

    oi = lambda g, k: feats[g][k]["oinfo"]
    return {  # name: (X, k for in-fold top-k selection or None)
        "FC edges (ridge, all)": (stack(lambda g, k: upper(feats[g][k]["FC"])), None),
        "FC edges (in-fold top-200)": (stack(lambda g, k: upper(feats[g][k]["FC"])), 200),
        "lag-1 FS edges (in-fold top-200)": (stack(lambda g, k: offdiag(feats[g][k]["FS1"])), 200),
        "GEC edges (ridge, SC mask)": (stack(lambda g, k: res[g][k]["GEC"][nz]), None),
        "FC network blocks (28)": (stack(lambda g, k: feats[g][k]["blocks"]), None),
        "HOI: O-info network compositions (168)": (stack(lambda g, k: np.r_[oi(g, k)["comp_mean"], oi(g, k)["comp_syn"]]), None),
        "HOI + FC blocks (196)": (stack(lambda g, k: np.r_[oi(g, k)["comp_mean"], oi(g, k)["comp_syn"], feats[g][k]["blocks"]]), None),
        "HOI: nodal O-info (100)": (stack(lambda g, k: oi(g, k)["nodal"]), None),
        "Hierarchy: P_j + nodal irreversibility lag2 (200)": (stack(lambda g, k: np.r_[res[g][k]["fdt_P"], feats[g][k]["irr_nodal"][2]]), None),
        "Perturbation: p2h noise + a-neg maps (200)": (stack(lambda g, k: np.r_[res[g][k]["p2h_noise"], res[g][k]["p2h_aneg"]]), None),
        "Mana restorative: p2h at set": (stack(lambda g, k: np.r_[res[g][k]["p2h_noise"][mana_rest], res[g][k]["p2h_aneg"][mana_rest]]), None),
        "Mana restorative: incident FC edges": (stack(lambda g, k: feats[g][k]["FC"][rest_edges & triu]), None),
        "Keane 2025: SomMot-Vis FC edges": (stack(lambda g, k: feats[g][k]["FC"][somvis & triu]), None),
    }


def main(n_perm, n_repeats):
    out = paths.study_dir("symptom_prediction")
    _, _, scores = load_data()
    mr = load_model_results()
    res, network = mr["results"], mr["atlas"]["network"]
    feats = load_features()
    mana_rest, _ = mana_region_sets()
    pt = [(g, k) for g in PATIENTS for k in res[g]]
    gvec = np.array([g for g, _ in pt])
    sets = feature_sets(pt, res, feats, network, mana_rest)
    rows = []
    for target in ("PANSS", "YMRS"):
        y = np.array([scores[g][target][k] for g, k in pt])
        m = ~np.isnan(y)
        for name, (X, k) in sets.items():
            r = evaluate(X[m], y[m], gvec[m], k=k, n_repeats=n_repeats, n_perm=n_perm)
            rows.append({"target": target, "features": name, "n": int(m.sum()), "n_features": X.shape[1], **r})
            print(target, name, {kk: round(float(v), 3) for kk, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-repeats", type=int, default=5)
    a = ap.parse_args()
    main(a.n_perm, a.n_repeats)
