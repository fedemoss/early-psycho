"""
Audit of the v1 headline PANSS model (legacy/notebooks/panss_model_region_overfit.ipynb):
lag-1 FC edges, |corr(edge, PANSS)| > threshold selected on the FULL sample, then RidgeCV LOOCV.

Compares (a) that leaky pipeline, (b) the same pipeline with selection inside each LOOCV fold,
(c) a permutation null of the leaky pipeline (PANSS shuffled, whole pipeline rerun).
Output: data/derived/v2/symptom_prediction/leakage_audit.csv (+ leakage_null.csv)

Run from the repository root:  python -m src.symptom_prediction.leakage_audit
"""
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from src.core import paths
from src.core.data import PATIENTS, load_data

ALPHAS = np.logspace(-3, 3, 13)  # as in v1


def ftau_edges(ts):
    N = ts.shape[0]
    return np.nan_to_num(np.corrcoef(ts[:, :-1], ts[:, 1:])[:N, N:])[~np.eye(N, dtype=bool)]


def select(X, y, thr):
    Xc, yc = X - X.mean(0), y - y.mean()
    sd = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    r = np.nan_to_num((Xc * yc[:, None]).sum(0) / np.where(sd == 0, np.inf, sd))
    s = np.where(np.abs(r) > thr)[0]
    return s if s.size else np.array([np.argmax(np.abs(r))])


def loocv(X, y, thr, nested):
    s_full = None if nested else select(X, y, thr)
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool)
        tr[i] = False
        s = select(X[tr], y[tr], thr) if nested else s_full
        pred[i] = RidgeCV(alphas=ALPHAS).fit(X[tr][:, s], y[tr]).predict(X[i:i + 1][:, s])[0]
    return r2_score(y, pred), pearsonr(y, pred)[0]


def main(n_perm, n_jobs):
    out = paths.study_dir("symptom_prediction")
    ts, _, scores = load_data()
    X, y = [], []
    for g in PATIENTS:
        for k in range(len(ts[g])):
            if not np.isnan(scores[g]["PANSS"][k]):
                X.append(ftau_edges(ts[g][k]))
                y.append(scores[g]["PANSS"][k])
    X, y = np.array(X), np.array(y)
    rows = []
    for thr in (0.15, 0.2, 0.25):
        for nested in (False, True):
            r2, r = loocv(X, y, thr, nested)
            rows.append({"threshold": thr, "selection": "inside folds" if nested else "full sample (leaky)",
                         "R2": r2, "r": r})
    rng = np.random.default_rng(0)
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(loocv)(X, rng.permutation(y), 0.2, False) for _ in range(n_perm)))
    leaky = [r for r in rows if r["threshold"] == 0.2 and r["selection"].startswith("full")][0]
    rows.append({"threshold": 0.2, "selection": f"leaky, permutation null (n={n_perm})",
                 "R2": null[:, 0].mean(), "r": null[:, 1].mean(),
                 "p_R2": (np.sum(null[:, 0] >= leaky["R2"]) + 1) / (n_perm + 1)})
    pd.DataFrame(rows).to_csv(out / "leakage_audit.csv", index=False)
    pd.DataFrame(null, columns=["R2", "r"]).to_csv(out / "leakage_null.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    main(a.n_perm, a.n_jobs)
