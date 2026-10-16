"""Data loading: timeseries, structural connectome, clinical scores, derived results."""
import pickle
import numpy as np
from . import paths

GROUPS = ("HC", "AP", "NAP")  # healthy controls, affective psychosis, non-affective psychosis
PATIENTS = ("AP", "NAP")


def load_data():
    """Return (ts, sc, scores): ts[group] -> (n_subj, 100, T); sc (100, 100); scores[group][PANSS|YMRS] -> (n_subj,)."""
    f = np.load(paths.TIMESERIES)
    c = np.load(paths.CLINICAL)
    sc = np.load(paths.SC)["SC"]
    ts = {"HC": f["ts_hc"], "AP": f["ts_ap"], "NAP": f["ts_nap"]}
    scores = {}
    for g in PATIENTS:
        panss = c[f"panss_{g}"].astype(float)
        ymrs = c[f"ymrs_{g}"].astype(float)
        # A row with PANSS missing (0/NaN) and YMRS == 0 is a missing assessment, not a zero score.
        ymrs[np.isnan(panss) & (np.nan_to_num(ymrs) == 0)] = np.nan
        scores[g] = {"PANSS": panss, "YMRS": ymrs}
    return ts, sc, scores


def load_model_results():
    """Output of src.gec_model.refit: dict with 'results'[group][k], 'atlas', 'settings'."""
    with open(paths.MODEL_RESULTS, "rb") as f:
        return pickle.load(f)


def load_features():
    """Output of src.core.features: empirical per-subject features, [group][k] -> dict."""
    with open(paths.FEATURES, "rb") as f:
        return pickle.load(f)


def patient_index(groups=PATIENTS, n_per_group=None):
    """List of (group, k) for patients, in the canonical AP-then-NAP order."""
    if n_per_group is None:
        ts, _, _ = load_data()
        n_per_group = {g: len(ts[g]) for g in groups}
    return [(g, k) for g in groups for k in range(n_per_group[g])]
