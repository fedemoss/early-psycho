"""
Linearised Hopf (Stuart-Landau) whole-brain model, GEC fitting and in-silico perturbations.

Fixes relative to the v1 port (legacy/gec_and_pert_analysis.py):
  - omega = 2*pi*f (rad/s); v1 used f (Hz) directly in the Jacobian.
  - GEC rescaled to max = MAXC = 0.1 (legacy/Ceff_EP_HCP.m), not to 1.
  - the empirical lag-tau FS target now has the same orientation as the model's expm(tau*A) K
    (v1 compared cov(x_i(t), x_j(t+tau)) with cov(x_i(t+tau), x_j(t))).
  - homotopic updates between true left/right homologues, not region i and i+N/2.
  - TR = 0.8 s (HCP-EP).
"""
import numpy as np
import scipy.linalg as la
from scipy.signal import welch

TR = 0.8
SIGMA = 0.01
A_BIF = -0.02
MAXC = 0.1


def intrinsic_omega(ts, tr=TR, band=(0.008, 0.08)):
    """Peak frequency per region within band, as angular frequency (rad/s)."""
    freqs, psd = welch(ts, fs=1.0 / tr, nperseg=min(ts.shape[1], 512))
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return 2 * np.pi * freqs[idx[np.argmax(psd[:, idx], axis=1)]]


def jacobian(C, omega, a=A_BIF):
    """2N x 2N Jacobian of the linearised network; a may be scalar or (N,)."""
    N = C.shape[0]
    a = np.broadcast_to(np.asarray(a, dtype=float), (N,))
    Jxx = np.diag(a - C.sum(1)) + C
    W = np.diag(omega)
    return np.block([[Jxx, W], [-W, Jxx]])


def stationary_cov(A, sigma=SIGMA, extra_noise=None):
    """Solve A K + K A^T + Q = 0. extra_noise=(node, delta_sigma) adds delta^2 to that node's x and y noise."""
    N2 = A.shape[0]
    Q = (sigma ** 2) * np.eye(N2)
    if extra_noise is not None:
        j, d = extra_noise
        Q[j, j] += d ** 2
        Q[j + N2 // 2, j + N2 // 2] += d ** 2
    return la.solve_continuous_lyapunov(A, -Q)


def cov_to_corr(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)


def model_fc(C, omega, a=A_BIF, sigma=SIGMA, extra_noise=None):
    """Simulated FC (real-part correlation), full 2N covariance K and Jacobian A."""
    A = jacobian(C, omega, a)
    K = stationary_cov(A, sigma, extra_noise)
    N = C.shape[0]
    return cov_to_corr(K[:N, :N]), K, A


def model_fs(K, A, tau_s):
    """FS_ij = cov(x_i(t), x_j(t+tau)) / sqrt(var_i var_j), same convention as empirical_fs."""
    N = A.shape[0] // 2
    Ktau = (la.expm(tau_s * A) @ K)[:N, :N]  # <x(t+tau) x(t)^T>
    d = np.sqrt(np.clip(np.diag(K)[:N], 1e-12, None))
    return (Ktau / np.outer(d, d)).T


def empirical_fs(ts, lag=1):
    """FS_ij = cov(x_i(t), x_j(t+lag)) / sqrt(var_i var_j)."""
    x = ts - ts.mean(1, keepdims=True)
    T = x.shape[1]
    cov = x[:, :-lag] @ x[:, lag:].T / T
    v = x.var(1, ddof=1)
    return cov / np.sqrt(np.outer(v, v))


def fit_gec(ts, sc, homologues, tr=TR, lag=1, maxC=MAXC, sigma=SIGMA,
            eps_fc=0.0002, eps_fs=0.00004, max_iter=5000):
    """
    Pseudo-gradient Generative Effective Connectivity fit (Deco/Kringelbach) to FC and lag-`lag` FS.
    Only SC-connected pairs and homologues are updated. Returns GEC, omega, simulated FC/FS, fit r, iterations.
    """
    N = ts.shape[0]
    tril = np.tril_indices(N, -1)
    offd = ~np.eye(N, dtype=bool)
    fc_emp = np.corrcoef(ts)
    fs_emp = empirical_fs(ts, lag)
    omega = intrinsic_omega(ts, tr)
    mask = sc > 0
    for i, j in homologues:
        mask[i, j] = mask[j, i] = True
    np.fill_diagonal(mask, False)
    C = sc / sc.max() * maxC
    C[~mask] = 0
    old = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        fc_sim, K, A = model_fc(C, omega, sigma=sigma)
        fs_sim = model_fs(K, A, lag * tr)
        if it % 100 == 0:
            err = np.mean((fc_emp - fc_sim) ** 2) + np.mean((fs_emp - fs_sim) ** 2)
            if (old - err) / err < 0.01 or old < err:
                break
            old = err
        C[mask] += eps_fc * (fc_emp - fc_sim)[mask] + eps_fs * (fs_emp - fs_sim)[mask]
        C[C < 0] = 0
        C = C / C.max() * maxC
    fc_sim, K, A = model_fc(C, omega, sigma=sigma)
    fs_sim = model_fs(K, A, lag * tr)
    return {"GEC": C, "omega": omega, "FC_sim": fc_sim, "FS_sim": fs_sim,
            "fit_FC": np.corrcoef(fc_emp[tril], fc_sim[tril])[0, 1],
            "fit_FS": np.corrcoef(fs_emp[offd], fs_sim[offd])[0, 1], "n_iter": it}


def fdt_perturbability(C, omega, a=A_BIF, sigma=SIGMA):
    """
    Acero-Pousa et al. 2025 (Transl Psychiatry) eq. 16: P_j = sum_i(2 K_ij / sigma^2 - R_ij) / sum_i R_ij,
    R = -J^{-1} (real block). mean_j P_j = FDT deviation ("hierarchy").
    v1's compute_perturbability_fdt omitted the -R term, i.e. returned P_j + 1.
    """
    N = C.shape[0]
    A = jacobian(C, omega, a)
    K = stationary_cov(A, sigma)[:N, :N]
    R = (-la.inv(A))[:N, :N]
    return ((2.0 / sigma ** 2) * K - R).sum(0) / R.sum(0)


def _sim(fc, ref, tri):
    return np.corrcoef(fc[tri], ref[tri])[0, 1]


def perturbation_maps(C, omega, fc_healthy, fc_patient, delta_sigma=0.1, delta_a=0.02):
    """
    Per-node in-silico perturbations relative to the unperturbed model FC:
      p2h_noise[j] : change in sim(FC, FC_HC) with extra noise at j    (HIGHER = toward healthy; best = argmax)
      p2o_noise[j] : 1 - sim(FC_pert, FC_base)                          (size of the shift)
      p2h_aneg[j]  : change in sim(FC, FC_HC) with a_j -> a - delta_a   (Mana et al. 'noise' protocol)
      p2p_apos[j]  : change in sim(FC, FC_patient) with a_j -> a + delta_a/2  (Mana 'synchronisation'; stays a<0)
    """
    N = C.shape[0]
    tri = np.triu_indices(N, 1)
    fc0, _, _ = model_fc(C, omega)
    base_h, base_p = _sim(fc0, fc_healthy, tri), _sim(fc0, fc_patient, tri)
    out = {k: np.zeros(N) for k in ("p2h_noise", "p2o_noise", "p2h_aneg", "p2p_apos")}
    for j in range(N):
        fcn, _, _ = model_fc(C, omega, extra_noise=(j, delta_sigma))
        out["p2h_noise"][j] = _sim(fcn, fc_healthy, tri) - base_h
        out["p2o_noise"][j] = 1 - _sim(fcn, fc0, tri)
        a = np.full(N, A_BIF)
        a[j] -= delta_a
        fca, _, _ = model_fc(C, omega, a=a)
        out["p2h_aneg"][j] = _sim(fca, fc_healthy, tri) - base_h
        a = np.full(N, A_BIF)
        a[j] += delta_a / 2
        fcp, _, _ = model_fc(C, omega, a=a)
        out["p2p_apos"][j] = _sim(fcp, fc_patient, tri) - base_p
    return out
