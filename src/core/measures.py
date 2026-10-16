"""Model-free measures: irreversibility, Gaussian higher-order information, FC summaries."""
import itertools
import numpy as np
from scipy.stats import norm, rankdata

from .hopf import empirical_fs


def upper(M):
    return M[np.triu_indices(M.shape[0], 1)]


def offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def irreversibility(ts, lag):
    """
    Pairwise time-irreversibility (INSIDEOUT-style): FS(lag) of the forward vs time-reversed series
    (the reversed FS is FS^T). Returns (matrix (FS - FS^T)^2, nodal mean, global mean, hierarchy = SD nodal).
    """
    fs = empirical_fs(ts, lag)
    M = (fs - fs.T) ** 2
    np.fill_diagonal(M, 0)
    nodal = M.sum(1) / (M.shape[0] - 1)
    return M, nodal, nodal.mean(), nodal.std()


def copula_normalize(ts):
    """Gaussian-copula transform per region (Ince et al. 2017): ranks -> standard normal scores."""
    r = np.apply_along_axis(rankdata, 1, ts)
    return norm.ppf(r / (ts.shape[1] + 1))


TRIPLETS = np.array(list(itertools.combinations(range(100), 3)))


def triplet_oinfo(R, triplets=TRIPLETS):
    """
    Gaussian O-information (Rosas et al. 2019) for every triplet, closed form from a correlation matrix:
    Omega = TC - DTC (> 0 redundancy-dominated, < 0 synergy-dominated). Returns (Omega, TC) in nats.
    """
    a = R[triplets[:, 0], triplets[:, 1]]
    b = R[triplets[:, 0], triplets[:, 2]]
    g = R[triplets[:, 1], triplets[:, 2]]
    det = 1 - a * a - b * b - g * g + 2 * a * b * g
    tc = -0.5 * np.log(det)
    # diag(R^-1) = cofactor / det
    dtc = 0.5 * np.log(det) + 0.5 * (np.log((1 - g * g) / det) + np.log((1 - b * b) / det)
                                     + np.log((1 - a * a) / det))
    return tc - dtc, tc


def oinfo_features(R, network, triplets=TRIPLETS):
    """
    Subject-level higher-order summary:
      comp_mean / comp_syn : mean Omega / fraction synergistic per network composition (84 multisets of 3 Yeo networks)
      nodal                : mean Omega over all triplets containing each node
      global_mean / global_syn / global_tc
    """
    om, tc = triplet_oinfo(R, triplets)
    nets = np.unique(network)
    code = {n: k for k, n in enumerate(nets)}
    tn = np.sort(np.vectorize(code.get)(network[triplets]), axis=1)
    combos = list(itertools.combinations_with_replacement(range(len(nets)), 3))
    key = tn[:, 0] * 49 + tn[:, 1] * 7 + tn[:, 2]
    comp_mean, comp_syn = np.zeros(len(combos)), np.zeros(len(combos))
    for k, (x, y, z) in enumerate(combos):
        sel = key == x * 49 + y * 7 + z
        comp_mean[k] = om[sel].mean()
        comp_syn[k] = (om[sel] < 0).mean()
    N = R.shape[0]
    nodal, cnt = np.zeros(N), np.zeros(N)
    for col in range(3):
        np.add.at(nodal, triplets[:, col], om)
        np.add.at(cnt, triplets[:, col], 1)
    return {"comp_mean": comp_mean, "comp_syn": comp_syn,
            "comp_names": ["+".join(nets[list(c)]) for c in combos],
            "nodal": nodal / cnt, "global_mean": om.mean(), "global_syn": (om < 0).mean(),
            "global_tc": tc.mean()}


def fc_network_blocks(R, network):
    """Mean FC within/between Yeo networks (28 values): the matched-dimension pairwise baseline."""
    nets = np.unique(network)
    out, names = [], []
    for x, y in itertools.combinations_with_replacement(range(len(nets)), 2):
        bx, by = network == nets[x], network == nets[y]
        blk = R[np.ix_(bx, by)]
        out.append(blk[~np.eye(*blk.shape, dtype=bool)].mean() if x == y else blk.mean())
        names.append(f"{nets[x]}-{nets[y]}")
    return np.array(out), names
