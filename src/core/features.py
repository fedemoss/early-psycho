"""
Empirical per-subject features shared by several studies (FC, lag-1 FS, O-information,
irreversibility, FC network blocks). Cached to data/derived/v2/features/empirical_features.pkl.

Run from the repository root:  python -m src.core.features
"""
import pickle
import numpy as np
from joblib import Parallel, delayed

from . import paths
from .atlas import atlas_info
from .data import GROUPS, load_data
from .hopf import empirical_fs
from .measures import copula_normalize, fc_network_blocks, irreversibility, oinfo_features

LAGS = (1, 2, 3)


def subject_features(x, network):
    fc = np.corrcoef(x)
    irr = {lag: irreversibility(x, lag) for lag in LAGS}
    blocks, block_names = fc_network_blocks(fc, network)
    return {"FC": fc, "FS1": empirical_fs(x, 1),
            "oinfo": oinfo_features(np.corrcoef(copula_normalize(x)), network),
            "oinfo_pearson": oinfo_features(fc, network),
            "blocks": blocks, "block_names": block_names,
            "irr_nodal": {l: irr[l][1] for l in LAGS}, "irr_global": {l: irr[l][2] for l in LAGS},
            "irr_hier": {l: irr[l][3] for l in LAGS}}


def main(n_jobs=8):
    ts, _, _ = load_data()
    network = atlas_info()["network"]
    feats = {g: dict(enumerate(Parallel(n_jobs=n_jobs)(delayed(subject_features)(x, network) for x in ts[g])))
             for g in GROUPS}
    paths.FEATURES.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.FEATURES, "wb") as f:
        pickle.dump(feats, f)
    print("saved", paths.FEATURES)


if __name__ == "__main__":
    main()
