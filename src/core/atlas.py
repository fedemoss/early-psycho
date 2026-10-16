"""Schaefer-100 (7 networks) atlas information and literature-derived region sets."""
import numpy as np


def atlas_info():
    """Labels, Yeo network, hemisphere, centroids and left/right homologue pairs (mirrored centroids, same network)."""
    from nilearn import datasets
    from nilearn.plotting import find_parcellation_cut_coords
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, verbose=0)
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas.labels]
    labels = [l.replace("7Networks_", "") for l in labels if l != "Background"]
    coords = find_parcellation_cut_coords(atlas.maps)
    hemi = np.array([l.split("_")[0] for l in labels])
    net = np.array([l.split("_")[1] for l in labels])
    pairs = set()
    for src, dst in (("LH", "RH"), ("RH", "LH")):
        for i in np.where(hemi == src)[0]:
            cand = np.where((hemi == dst) & (net == net[i]))[0]
            mirrored = coords[i] * np.array([-1, 1, 1])
            j = cand[np.argmin(np.linalg.norm(coords[cand] - mirrored, axis=1))]
            pairs.add((min(i, j), max(i, j)))
    return {"labels": labels, "network": net, "hemi": hemi, "coords": coords,
            "homologues": sorted(pairs)}


# Mana et al. 2023 (Cereb Cortex 33:7642, Desikan-Killiany atlas) critical areas, mapped to
# Schaefer-100 via majority Harvard-Oxford overlap (see mana_region_sets()).
MANA_RESTORATIVE_HO = (  # SZ model -> healthy (noise protocol, a_j decreased)
    "Frontal Orbital Cortex", "Frontal Medial Cortex", "Subcallosal Cortex",  # medial/lateral OFC, pars orbitalis
    "Cingulate Gyrus, posterior division",                                   # posterior + isthmus cingulate
    "Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division",
)
MANA_PATHOGENIC_HO = (  # healthy model -> SZ (synchronisation protocol, a_j increased)
    "Postcentral Gyrus", "Insular Cortex",
    "Supramarginal Gyrus, anterior division", "Supramarginal Gyrus, posterior division", "Angular Gyrus",
    "Temporal Fusiform Cortex, anterior division", "Temporal Fusiform Cortex, posterior division",
    "Temporal Occipital Fusiform Cortex", "Occipital Fusiform Gyrus",
    "Middle Temporal Gyrus, anterior division", "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part",
)


def harvard_oxford_overlap():
    """(100, n_HO) fraction of each Schaefer parcel's voxels in each Harvard-Oxford cortical label."""
    import nibabel as nib
    from nilearn import datasets, image
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, verbose=0)
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", verbose=0)
    s_img = nib.load(sch.maps)
    h_img = image.resample_to_img(ho.maps, s_img, interpolation="nearest", force_resample=True, copy_header=True)
    s, h = s_img.get_fdata().astype(int), h_img.get_fdata().astype(int)
    ho_labels = list(ho.labels)
    share = np.zeros((100, len(ho_labels)))
    for r in range(1, 101):
        v = h[s == r]
        v = v[v > 0]
        share[r - 1] = np.bincount(v, minlength=len(ho_labels)) / max(1, v.size)
    return share, ho_labels


def mana_region_sets(min_share_restorative=0.30, min_share_pathogenic=0.50):
    """
    Schaefer-100 indices whose summed Harvard-Oxford overlap with a Mana et al. 2023 region set
    reaches min_share. The pathogenic list is much broader (12 HO labels), hence the stricter
    threshold; nodes in both sets are kept as restorative only.
    """
    share, ho_labels = harvard_oxford_overlap()

    def pick(names, thr):
        cols = [ho_labels.index(n) for n in names]
        return np.where(share[:, cols].sum(1) >= thr)[0]

    restorative = pick(MANA_RESTORATIVE_HO, min_share_restorative)
    pathogenic = np.setdiff1d(pick(MANA_PATHOGENIC_HO, min_share_pathogenic), restorative)
    return restorative, pathogenic
