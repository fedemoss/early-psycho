"""Group comparisons and symptom correlations with permutation inference."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def perm_F(values, groups, n_perm=5000, seed=0):
    """One-way ANOVA F statistic with a label-permutation p-value."""
    labels = np.unique(groups)

    def F(v):
        gm = [v[groups == u].mean() for u in labels]
        ns = [np.sum(groups == u) for u in labels]
        between = sum(n * (m - v.mean()) ** 2 for n, m in zip(ns, gm)) / (len(gm) - 1)
        within = sum(((v[groups == u] - v[groups == u].mean()) ** 2).sum() for u in labels) / (len(v) - len(gm))
        return between / within

    rng = np.random.default_rng(seed)
    f0 = F(values)
    null = np.array([F(rng.permutation(values)) for _ in range(n_perm)])
    return f0, (np.sum(null >= f0) + 1) / (n_perm + 1)


def hedges_g(a, b):
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s * (1 - 3 / (4 * (len(a) + len(b)) - 9))


def fdr(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def group_tests(df, measures, groups=("HC", "AP", "NAP"), n_perm=5000):
    """Permutation F test + pairwise Hedges g for each column of df (needs a 'group' column)."""
    grp = df["group"].values
    rows = []
    for m in measures:
        v = df[m].values.astype(float)
        f0, p = perm_F(v, grp, n_perm)
        row = {"measure": m, **{f"mean_{g}": v[grp == g].mean() for g in groups}, "F": f0, "p_perm": p}
        for a, b in (("AP", "HC"), ("NAP", "HC"), ("NAP", "AP")):
            row[f"g_{a}_vs_{b}"] = hedges_g(v[grp == a], v[grp == b])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_FDR"] = fdr(out["p_perm"])
    return out


def partial_spearman(x, y, g, n_perm=5000, seed=0):
    """Spearman correlation after removing group means from both ranks; p from within-group permutations."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[m], y[m], g[m]

    def resid(v):
        v = rankdata(v).astype(float)
        for u in np.unique(g):
            v[g == u] -= v[g == u].mean()
        return v

    rx = resid(x)
    r0 = np.corrcoef(rx, resid(y))[0, 1]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for u in np.unique(g):
            yp[g == u] = rng.permutation(y[g == u])
        null.append(np.corrcoef(rx, resid(yp))[0, 1])
    return r0, (np.sum(np.abs(null) >= abs(r0)) + 1) / (n_perm + 1), int(m.sum())


def symptom_correlations(df, measures, targets=("PANSS", "YMRS"), n_perm=5000):
    """Partial Spearman (controlling AP/NAP) of each measure with each target, patients only."""
    P = df[df["group"] != "HC"]
    rows = []
    for t in targets:
        for m in measures:
            r, p, n = partial_spearman(P[m].values.astype(float), P[t].values.astype(float),
                                       P["group"].values, n_perm)
            rows.append({"target": t, "measure": m, "rho_partial": r, "p_perm": p, "n": n})
    out = pd.DataFrame(rows)
    out["q_FDR"] = out.groupby("target")["p_perm"].transform(fdr)
    return out
