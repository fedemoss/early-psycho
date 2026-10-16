"""Filesystem layout. Override the data root with the EP_DATA_DIR environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("EP_DATA_DIR", ROOT / "data"))
INPUT = DATA / "input"          # preprocessed inputs: timeseries, SC, clinical scores
RAW = DATA / "raw"              # HCP-EP Schaefer-1000 runs with subject IDs, clinical tables
DERIVED = DATA / "derived" / "v2"

TIMESERIES = INPUT / "HCP_early_psychosis_timeseries_cleaned.npz"
CLINICAL = INPUT / "HCP_early_psychosis_clinical_scores.npz"
SC = INPUT / "Structural_Connectomes.npz"
MODEL_RESULTS = DERIVED / "gec_model" / "model_results.pkl"
FEATURES = DERIVED / "features" / "empirical_features.pkl"


def study_dir(name):
    """Output directory for one study (created on demand)."""
    d = DERIVED / name
    d.mkdir(parents=True, exist_ok=True)
    return d
