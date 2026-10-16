"""
Leak-free evaluation of symptom models.

Feature selection, standardisation and the ridge penalty are all fit inside each training fold.
Group (AP/NAP) means are removed inside the training fold and added back to predictions, and the
permutation null shuffles scores WITHIN group, so a significant result means the features explain
symptom variation beyond the AP/NAP difference.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ALPHAS = np.logspace(-3, 5, 17)


def _fold_fit_predict(Xtr, ytr, gtr, Xte, gte, k):
    gm = {g: ytr[gtr == g].mean() for g in np.unique(gtr)}
    rtr = ytr - np.array([gm[g] for g in gtr])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if k is not None and k < Ztr.shape[1]:
        rc = rtr - rtr.mean()
        corr = np.abs(Ztr.T @ rc) / (np.sqrt((Ztr ** 2).sum(0)) * np.sqrt((rc ** 2).sum()) + 1e-12)
        idx = np.argsort(corr)[::-1][:k]
        Ztr, Zte = Ztr[:, idx], Zte[:, idx]
    model = RidgeCV(alphas=ALPHAS).fit(Ztr, rtr)
    return model.predict(Zte) + np.array([gm.get(g, ytr.mean()) for g in gte])


def cv_predict(X, y, g, k=None, n_splits=10, seed=0):
    """Out-of-fold predictions from one shuffled K-fold split."""
    pred = np.zeros_like(y, dtype=float)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        pred[te] = _fold_fit_predict(X[tr], y[tr], g[tr], X[te], g[te], k)
    return pred


def group_only_predict(y, g, n_splits=10, seed=0):
    pred = np.zeros_like(y, dtype=float)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(y):
        gm = {gg: y[tr][g[tr] == gg].mean() for gg in np.unique(g[tr])}
        pred[te] = [gm.get(gg, y[tr].mean()) for gg in g[te]]
    return pred


def score(y, pred, g):
    """(R^2 against total variance, r between prediction and truth after removing group means)."""
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    yr, pr = y.astype(float).copy(), pred.copy()
    for gg in np.unique(g):
        yr[g == gg] -= y[g == gg].mean()
        pr[g == gg] -= pred[g == gg].mean()
    return r2, np.corrcoef(yr, pr)[0, 1]


def evaluate(X, y, g, k=None, n_repeats=5, n_perm=200, seed=0, n_jobs=8):
    """
    Repeated 10-fold CV; statistic = within-group r averaged over repeats. Null = the full pipeline on
    scores permuted within group (2 repeats per permutation, which makes the test slightly conservative).
    Note: CV correlations are biased negative when predictions shrink to the mean; the null has the same bias.
    """
    def stat(yy, seeds):
        rs, r2s = [], []
        for s in seeds:
            r2, r = score(yy, cv_predict(X, yy, g, k, seed=s), g)
            rs.append(r)
            r2s.append(r2)
        return np.mean(rs), np.mean(r2s)

    seeds = list(range(seed, seed + n_repeats))
    r_obs, r2_obs = stat(y, seeds)
    r2_group = np.mean([score(y, group_only_predict(y, g, seed=s), g)[0] for s in seeds])
    rng = np.random.default_rng(seed + 12345)
    perms = []
    for _ in range(n_perm):
        yp = y.copy()
        for gg in np.unique(g):
            yp[g == gg] = rng.permutation(y[g == gg])
        perms.append(yp)
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(stat)(yp, seeds[:2]) for yp in perms))[:, 0]
    return {"r_within": r_obs, "R2": r2_obs, "R2_group_only": r2_group,
            "p_perm": (np.sum(null >= r_obs) + 1) / (n_perm + 1),
            "null_mean": null.mean(), "null_95": np.percentile(null, 95)}
