"""
Fit the GEC (src.core.hopf.fit_gec) for every subject and compute per-node perturbation maps and
the FDT perturbability map. Output: data/derived/v2/gec_model/model_results.pkl

Healthy / patient reference FCs are leave-one-out for subjects that belong to them.
Run from the repository root (about 20 min on 8 cores):
    OMP_NUM_THREADS=1 python -m src.gec_model.refit
"""
import argparse
import pickle
import time
import numpy as np
from joblib import Parallel, delayed

from src.core import hopf, paths
from src.core.atlas import atlas_info
from src.core.data import GROUPS, load_data


def run_subject(group, k, ts_sub, sc, homologues, fc_hc_sum, n_hc, fc_pt_sum, n_pt):
    fc_emp = np.corrcoef(ts_sub)
    fc_hc = (fc_hc_sum - fc_emp) / (n_hc - 1) if group == "HC" else fc_hc_sum / n_hc
    fc_pt = fc_pt_sum / n_pt if group == "HC" else (fc_pt_sum - fc_emp) / (n_pt - 1)
    t = time.time()
    fit = hopf.fit_gec(ts_sub, sc, homologues)
    maps = hopf.perturbation_maps(fit["GEC"], fit["omega"], fc_hc, fc_pt)
    maps["fdt_P"] = hopf.fdt_perturbability(fit["GEC"], fit["omega"])
    print(f"{group} {k}: iters={fit['n_iter']} fitFC={fit['fit_FC']:.3f} fitFS={fit['fit_FS']:.3f} "
          f"({time.time() - t:.0f}s)", flush=True)
    return group, k, {**fit, **maps}


def main(n_jobs):
    ts, sc, _ = load_data()
    atlas = atlas_info()
    fcs = {g: np.array([np.corrcoef(x) for x in ts[g]]) for g in GROUPS}
    fc_hc_sum, n_hc = fcs["HC"].sum(0), len(fcs["HC"])
    fc_pt_sum, n_pt = fcs["AP"].sum(0) + fcs["NAP"].sum(0), len(fcs["AP"]) + len(fcs["NAP"])
    jobs = [(g, k) for g in GROUPS for k in range(len(ts[g]))]
    res = Parallel(n_jobs=n_jobs)(
        delayed(run_subject)(g, k, ts[g][k], sc, atlas["homologues"], fc_hc_sum, n_hc, fc_pt_sum, n_pt)
        for g, k in jobs)
    out = {g: {} for g in GROUPS}
    for g, k, r in res:
        out[g][k] = r
    paths.MODEL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.MODEL_RESULTS, "wb") as f:
        pickle.dump({"results": out, "atlas": atlas,
                     "settings": {"TR": hopf.TR, "sigma": hopf.SIGMA, "a": hopf.A_BIF, "maxC": hopf.MAXC,
                                  "lag": 1, "delta_sigma": 0.1, "delta_a": 0.02}}, f)
    print("saved", paths.MODEL_RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    main(ap.parse_args().n_jobs)
