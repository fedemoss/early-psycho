import os
import pickle
import numpy as np
import pandas as pd
import logging
import scipy.linalg as la
from scipy.signal import welch
from scipy.stats import pearsonr

# ==============================================================================
# 1. CORE CORE MATHEMATICAL & MODELING FUNCTIONS
# ==============================================================================

def hopf_int(Cnew, f_diff, sigma):
    """Linearized Hopf oscillator (Stuart-Landau) model solver."""
    N = Cnew.shape[0]
    a = -0.02 
    S = np.sum(Cnew, axis=1)
    Jxx = np.diag(a - S) + Cnew
    Jxy = np.diag(f_diff)
    A = np.block([
        [Jxx,  Jxy],
        [-Jxy, Jxx]
    ])
    Q = (sigma**2) * np.eye(2 * N)
    COVsim = la.solve_continuous_lyapunov(A, -Q)
    real_cov = COVsim[:N, :N]
    
    d = np.diag(real_cov).copy()
    d[d <= 0] = 1e-12
    std_inv = 1.0 / np.sqrt(d)
    FCsim = real_cov * np.outer(std_inv, std_inv)
    
    return FCsim, COVsim, None, A

def estimate_intrinsic_frequencies(ts, TR, band=(0.008, 0.08)):
    """Estimates the peak frequency for each brain region within a specific band."""
    N, T = ts.shape
    fs = 1.0 / TR
    f_diff = np.zeros(N)
    freqs, psd = welch(ts, fs=fs, nperseg=min(T, 512))
    idx_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    
    for i in range(N):
        peak_idx = idx_band[np.argmax(psd[i, idx_band])]
        f_diff[i] = freqs[peak_idx]
        
    return f_diff

def estimate_gec(ts, sc, TR, max_iter=5000):
    """Estimates the Generative Effective Connectivity (GEC) using a Linear Hopf model."""
    N, T = ts.shape
    Tau = 1 
    sigma = 0.01 
    epsFC = 0.0002 
    epsFCtau = 0.00004 
    
    tril_idx = np.tril_indices(N, k=-1)
    ts_c = ts - np.mean(ts, axis=1, keepdims=True)
    FCemp = np.corrcoef(ts)
    COVemp = np.cov(ts, ddof=1)
    
    COVtauemp_raw = np.dot(ts_c[:, :-Tau], ts_c[:, Tau:].T) / T
    diag_cov = np.diag(COVemp)
    sigratio = 1.0 / np.sqrt(np.outer(diag_cov, diag_cov))
    FSemp = COVtauemp_raw * sigratio

    f_diff = estimate_intrinsic_frequencies(ts, TR)
    Cnew = np.copy(sc)
    olderror = 1e6
    update_mask = (sc > 0) | np.eye(N, k=N//2, dtype=bool)

    for iteration in range(1, max_iter + 1):
        FCsim_curr, COVsim_full, aux, A = hopf_int(Cnew, f_diff, sigma)
        
        matrix_exp = la.expm((Tau * TR) * A)
        COVtausim_raw = matrix_exp @ COVsim_full
        COVtausim_real = COVtausim_raw[:N, :N] 
        
        diag_sim = np.diag(COVsim_full[:N, :N]).copy()
        diag_sim[diag_sim <= 0] = 1e-12 
        sigratiosim = 1.0 / np.sqrt(np.outer(diag_sim, diag_sim))
        FSsim_curr = COVtausim_real * sigratiosim
        
        error_FC = np.mean((FCemp - FCsim_curr)**2)
        error_FS = np.mean((FSemp - FSsim_curr)**2)
        
        if iteration % 100 == 0:
            errornow = error_FC + error_FS
            if (olderror - errornow) / errornow < 0.01 or olderror < errornow:
                print(f"      Converged at iteration {iteration} | MSE: {errornow:.6f}")
                break
            olderror = errornow

        gradient = epsFC * (FCemp - FCsim_curr) + epsFCtau * (FSemp - FSsim_curr)
        Cnew[update_mask] += gradient[update_mask]
        
        Cnew[Cnew < 0] = 0
        if Cnew.max() > 0:
            Cnew = (Cnew / Cnew.max())

    FCsim, COVsim_full, aux, A = hopf_int(Cnew, f_diff, sigma)
    matrix_exp = la.expm((Tau * TR) * A)
    FSsim_raw = (matrix_exp @ COVsim_full)[:N, :N]
    diag_final = np.diag(COVsim_full[:N, :N]).copy()
    diag_final[diag_final <= 0] = 1e-12
    sig_final = 1.0 / np.sqrt(np.outer(diag_final, diag_final))
    FSsim = FSsim_raw * sig_final
    
    fittFC = np.corrcoef(FCemp[tril_idx], FCsim[tril_idx])[0, 1]
    fittFS = np.corrcoef(FSemp[tril_idx], FSsim[tril_idx])[0, 1]
    
    return Cnew, FCsim, FSsim, fittFC, fittFS

def compute_model_metrics(emp, sim, matrix_type="FC"):
    """Computes a comprehensive suite of comparison metrics between empirical and simulated matrices."""
    N = emp.shape[0]
    
    if matrix_type.upper() == "FC":
        idx = np.tril_indices(N, k=-1)
        v_emp = emp[idx]
        v_sim = sim[idx]
    else:
        mask = ~np.eye(N, dtype=bool)
        v_emp = emp[mask]
        v_sim = sim[mask]

    r_val, _ = pearsonr(v_emp, v_sim)
    rmse = np.sqrt(np.mean((v_emp - v_sim)**2))
    mae = np.mean(np.abs(v_emp - v_sim))
    euclidean = np.linalg.norm(v_emp - v_sim)
    
    directedness_sim = None
    if matrix_type.upper() == "FS":
        directedness_sim = np.linalg.norm(sim - sim.T)

    return {
        "Pearson_r": r_val,
        "RMSE": rmse,
        "MAE": mae,
        "Euclidean_Dist": euclidean,
        "Directedness_Index": directedness_sim
    }

def calculate_group_mean_fc(ts_list):
    """Calculates the group-average Functional Connectivity (FC) matrix."""
    fcs = [np.corrcoef(ts) for ts in ts_list]
    return np.mean(fcs, axis=0)


# ==============================================================================
# 2. PERTURBABILITY & DYNAMICS FUNCTIONS
# ==============================================================================

def compute_perturbability_fdt(GEC, f_diff, sigma):
    """Computes the Perturbability map based on the Fluctuation-Dissipation Theorem (FDT)."""
    N = GEC.shape[0]
    a, Jxy = -0.02, np.diag(f_diff)
    S = np.sum(GEC, axis=1)
    Jxx = np.diag(a - S) + GEC
    A = np.block([[Jxx, Jxy], [-Jxy, Jxx]])
    
    Q = (sigma**2) * np.eye(2 * N)
    K = la.solve_continuous_lyapunov(A, -Q)
    K_real = K[:N, :N]
    
    R_real = (-la.inv(A))[:N, :N]
    
    numerator = (2.0 / sigma**2) * np.sum(K_real, axis=0)
    denominator = np.sum(R_real, axis=0)
    
    return numerator / denominator

def simulate_perturbed_fc(GEC, f_diff, sigma, node_idx, delta_sigma):
    """Simulates the system's Functional Connectivity (FC) under a local noise perturbation."""
    N = GEC.shape[0]
    a, Jxy = -0.02, np.diag(f_diff)
    Jxx = np.diag(a - np.sum(GEC, axis=1)) + GEC
    A = np.block([[Jxx, Jxy], [-Jxy, Jxx]])
    
    Q = (sigma**2) * np.eye(2 * N)
    Q[node_idx, node_idx] += delta_sigma**2
    Q[node_idx + N, node_idx + N] += delta_sigma**2
    
    K_p = la.solve_continuous_lyapunov(A, -Q)
    real_cov = K_p[:N, :N]
    
    d = np.diag(real_cov).copy()
    d[d <= 0] = 1e-12
    std_inv = 1.0 / np.sqrt(d)
    return real_cov * np.outer(std_inv, std_inv)

def compute_p2h_map(GEC, f_diff, sigma, mean_hc_fc, delta_sigma=0.1):
    """Calculates the Perturbability to Healthy (P2H) map."""
    N = GEC.shape[0]
    tri = np.triu_indices(N, k=1)
    
    fc_base, cov_sim, aux, A_mat = hopf_int(GEC, f_diff, sigma)
    sim_base = np.corrcoef(fc_base[tri], mean_hc_fc[tri])[0, 1]
    
    p2h_map = np.zeros(N)
    for j in range(N):
        fc_pert = simulate_perturbed_fc(GEC, f_diff, sigma, node_idx=j, delta_sigma=delta_sigma)
        sim_pert = np.corrcoef(fc_pert[tri], mean_hc_fc[tri])[0, 1]
        p2h_map[j] = sim_pert - sim_base
        
    return p2h_map

def compute_p2h_raw_map(GEC, f_diff, sigma, mean_hc_fc, delta_sigma=0.1):
    """
    Calculates the raw Perturbability-to-Healthy (P2H_raw) map: the absolute
    similarity between the perturbed FC and the healthy-group mean FC, without
    subtracting the unperturbed baseline similarity (contrast with
    compute_p2h_map, which returns sim_pert - sim_base).
    """
    N = GEC.shape[0]
    tri = np.triu_indices(N, k=1)

    p2h_raw_map = np.zeros(N)
    for j in range(N):
        fc_pert = simulate_perturbed_fc(GEC, f_diff, sigma, node_idx=j, delta_sigma=delta_sigma)
        sim_pert = np.corrcoef(fc_pert[tri], mean_hc_fc[tri])[0, 1]
        p2h_raw_map[j] = sim_pert

    return p2h_raw_map

def compute_p2o_map(GEC, f_diff, sigma, delta_sigma=0.1):
    """
    Calculates the Perturbability to Origin (P2O) map.
    Measures how much perturbing a node shifts the FC away from its own unperturbed baseline.
    """
    N = GEC.shape[0]
    tri = np.triu_indices(N, k=1)
    
    # 1. Get the unperturbed (origin) FC matrix
    fc_base, cov_sim, aux, A_mat = hopf_int(GEC, f_diff, sigma)
    
    # Extract the upper triangle of the unperturbed FC for vector correlation
    fc_base_vec = fc_base[tri]
    
    p2o_map = np.zeros(N)
    
    # 2. Loop through each node, perturb it, and see how far it moves from origin
    for j in range(N):
        # Simulate FC with node j perturbed
        fc_pert = simulate_perturbed_fc(GEC, f_diff, sigma, node_idx=j, delta_sigma=delta_sigma)
        fc_pert_vec = fc_pert[tri]
        
        # Calculate similarity between perturbed FC and original unperturbed FC
        sim_to_origin = np.corrcoef(fc_pert_vec, fc_base_vec)[0, 1]
        
        # P2O is the distance moved away from the origin (1.0 - similarity)
        # Higher values mean the node's perturbation caused a massive shift from baseline
        p2o_map[j] = 1.0 - sim_to_origin
        
    return p2o_map

# ==============================================================================
# 3. MODULAR PIPELINE EXECUTION WRAPPERS
# ==============================================================================

def execute_group_precision_modeling(ts_dict, sc, TR_value, output_dir):
    """Runs GEC optimization for all available subjects across groups and saves the map."""
    study_results = {group: {} for group in ts_dict.keys()}
    
    print("\nStarting Group-Wise Precision Modeling...")
    for group_name, ts_list in ts_dict.items():
        n_total = len(ts_list)
        print(f"\n{'='*50}")
        print(f"PROCESSING GROUP: {group_name} | Found {n_total} subjects total.")
        print(f"{'='*50}")
        
        for i in range(n_total):
            print(f"  -> [Subject {i+1}/{n_total}] Running GEC Optimization...")
            subject_ts = ts_list[i]
            
            # GEC Estimation
            gec, fc_sim, fs_sim, r_fc, r_fs = estimate_gec(subject_ts, sc, TR=TR_value)
            
            # Reconstruct Empirical Targets for metrics
            ts_c = subject_ts - np.mean(subject_ts, axis=1, keepdims=True)
            fc_emp = np.corrcoef(subject_ts)
            
            cov_tau_raw = np.dot(ts_c[:, :-1], ts_c[:, 1:].T) / subject_ts.shape[1]
            diag_cov = np.diag(np.cov(subject_ts))
            sig_inv = 1.0 / np.sqrt(np.outer(diag_cov, diag_cov))
            fs_emp = cov_tau_raw * sig_inv

            # Metrics
            metrics_fc = compute_model_metrics(fc_emp, fc_sim, matrix_type="FC")
            metrics_fs = compute_model_metrics(fs_emp, fs_sim, matrix_type="FS")
            
            study_results[group_name][i] = {
                "GEC_matrix": gec,
                "FC_sim": fc_sim,
                "FS_sim": fs_sim,
                "metrics_static": metrics_fc,
                "metrics_shifted": metrics_fs
            }
            
    # Save output dictionary securely
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "study_results.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(study_results, f)
    print(f"\n>>> Saved core modeling results securely to: {out_file}")
    
    return study_results

def execute_perturbation_analysis(study_results, ts_dict, mean_hc_fc, TR_value, sigma_val, output_dir):
    """Computes Perturbability, P2H, and P2O maps for all processed subjects and saves them."""
    p_maps = {group: {} for group in study_results.keys()}
    p2h_maps = {group: {} for group in study_results.keys()}
    p2o_maps = {group: {} for group in study_results.keys()}
    p2h_raw_maps = {group: {} for group in study_results.keys()}

    print("\nCalculating Potential Dynamics (Perturbation Maps) for all subjects...")
    for group_name, subjects in study_results.items():
        print(f"\nAnalyzing Dynamics for Group: {group_name}")
        ts_list = ts_dict[group_name]
        
        for i in subjects.keys():
            print(f"  -> Processing Subject {i+1}/{len(subjects)}")
            gec_matrix = study_results[group_name][i]["GEC_matrix"]
            subj_ts = ts_list[i]
            
            f_diff_subj = estimate_intrinsic_frequencies(subj_ts, TR_value)
            
            # Compute maps
            p_maps[group_name][i] = compute_perturbability_fdt(gec_matrix, f_diff_subj, sigma_val)
            p2h_maps[group_name][i] = compute_p2h_map(gec_matrix, f_diff_subj, sigma_val, mean_hc_fc)
            p2o_maps[group_name][i] = compute_p2o_map(gec_matrix, f_diff_subj, sigma_val)
            p2h_raw_maps[group_name][i] = compute_p2h_raw_map(gec_matrix, f_diff_subj, sigma_val, mean_hc_fc)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    p_file = os.path.join(output_dir, "perturbability_maps.pkl")
    p2h_file = os.path.join(output_dir, "p2h_maps.pkl")
    p2o_file = os.path.join(output_dir, "p2o_maps.pkl")
    p2h_raw_file = os.path.join(output_dir, "p2h_raw_maps.pkl")

    with open(p_file, 'wb') as f:
        pickle.dump(p_maps, f)
    with open(p2h_file, 'wb') as f:
        pickle.dump(p2h_maps, f)
    with open(p2o_file, 'wb') as f: # <-- Save P2O maps
        pickle.dump(p2o_maps, f)
    with open(p2h_raw_file, 'wb') as f:
        pickle.dump(p2h_raw_maps, f)

    print(f"\n>>> Saved Perturbability maps to: {p_file}")
    print(f">>> Saved P2H maps to: {p2h_file}")
    print(f">>> Saved P2O maps to: {p2o_file}")
    print(f">>> Saved P2H_raw maps to: {p2h_raw_file}")

    return p_maps, p2h_maps, p2o_maps, p2h_raw_maps # <-- Returning all four map dictionaries now
# ==============================================================================
# 4. ENVIRONMENT RUNNER BLOCK
# ==============================================================================


if __name__ == "__main__":
    # Path Declarations
    INPUT_DATA_DIR = "/share/data1/mossf/data/early-psycho/input/"
    OUTPUT_DATA_DIR = "/share/data1/mossf/data/early-psycho/output/"
    
    # =========================================================================
    # Logging Configuration
    # =========================================================================
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DATA_DIR, "perturbation_pipeline.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),  # Saves to file in your output dir
            logging.StreamHandler()         # Prints to console/slurm output
        ]
    )
    
    logging.info("Reading matrices from RAID0 environment storage...")
    connectomes = np.load(os.path.join(INPUT_DATA_DIR, "Structural_Connectomes.npz"))
    fmri = np.load(os.path.join(INPUT_DATA_DIR, "HCP_early_psychosis_timeseries_cleaned.npz"))
    
    # Extract Base Matrices
    sc = connectomes['SC']
    ts_dict = {
        'HC': fmri['ts_hc'],
        'AP': fmri['ts_ap'],
        'NAP': fmri['ts_nap']
    }
    
    # Base Cluster Parameters
    TR_value = 0.8
    sigma_val = 0.01
    
    # Calculate global Healthy Baseline
    logging.info("Calculating Baseline Healthy Functional Connectivity matrix...")
    mean_hc_fc = calculate_group_mean_fc(ts_dict['HC'])
    
    # =========================================================================
    # Pipeline Step 1: Run optimization for every single subject detected
    # =========================================================================
    # study_results = execute_group_precision_modeling(
    #     ts_dict=ts_dict, 
    #     sc=sc, 
    #     TR_value=TR_value, 
    #     output_dir=OUTPUT_DATA_DIR
    # )

    # LOAD PRECOMPUTED GEC RESULTS INSTEAD
    logging.info("Loading precomputed GEC study results...")
    study_results_file = os.path.join(OUTPUT_DATA_DIR, "study_results.pkl") 
    with open(study_results_file, 'rb') as f:
        study_results = pickle.load(f)
    logging.info("Study results loaded successfully!")
    
    # =========================================================================
    # Pipeline Step 2: Compute full-scale perturbation maps
    # =========================================================================
    logging.info("Initiating Step 2: Perturbation Analysis...")
    p_maps, p2h_maps, p2o_maps, p2h_raw_maps = execute_perturbation_analysis(
        study_results=study_results,
        ts_dict=ts_dict,
        mean_hc_fc=mean_hc_fc,
        TR_value=TR_value,
        sigma_val=sigma_val,
        output_dir=OUTPUT_DATA_DIR
    )
    
    logging.info("[COMPLETE] All pipeline components executed and stored cleanly on the cluster!")
