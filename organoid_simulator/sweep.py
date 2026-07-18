import os
import time
import json
import logging
import numpy as np
import scipy.sparse as sp
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from organoid_simulator.model import (
    build_recurrent_weights,
    scale_spectral_radius,
    run_homeostasis,
    run_simulation
)
from organoid_simulator.metrics import (
    compute_memory_capacity,
    compute_criticality_metrics
)

logger = logging.getLogger(__name__)

# Enforce single-threaded BLAS execution per worker to avoid high CPU oversubscription/thrashing
def enforce_single_thread():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_single_config(N, K, rho, seed, topology='random', lambda_ring=0.1, 
                      T=5000, washout=500, k_max=50, alpha=1e-4, 
                      target_mean=0.1, a=0.3):
    """
    Runs a single simulation config and returns metrics plus timing and memory profiling.
    """
    enforce_single_thread()
    
    t0 = time.time()
    
    try:
        # Build the recurrent network
        W_rec_raw, ids = build_recurrent_weights(N, K, topology=topology, lambda_ring=lambda_ring, seed=seed)
        
        # Scale to target spectral radius
        W_rec = scale_spectral_radius(W_rec_raw, rho)
        
        # Run homeostasis to set the input gain
        W_in, homeostasis_history = run_homeostasis(W_rec, target_mean=target_mean, a=a, num_iters=10, probe_steps=500, seed=seed)
        
        # Drive with input sequence
        np.random.seed(seed + 1000)
        u_seq = np.random.uniform(-1.0, 1.0, size=T).astype(np.float32)
        
        # Simulate network dynamics
        states = run_simulation(W_rec, W_in, u_seq, a=a)
        
        # Apply washout
        states_post_washout = states[washout:, :]
        u_post_washout = u_seq[washout:]
        
        # Compute Reservoir Memory Capacity (MC)
        total_mc, mc_curves = compute_memory_capacity(
            states_post_washout, u_post_washout, k_max=k_max, alpha=alpha, train_frac=0.8
        )
        
        # Compute Criticality Metrics
        crit_metrics = compute_criticality_metrics(states_post_washout)
        
        elapsed_time = time.time() - t0
        
        # Approximate memory footprint of the state matrix in MB
        state_memory_mb = states.nbytes / (1024 * 1024)
        
        # Pack results
        result = {
            'N': N,
            'K': K,
            'rho': rho,
            'seed': seed,
            'topology': topology,
            'lambda_ring': lambda_ring,
            'T': T,
            'washout': washout,
            'k_max': k_max,
            'alpha': alpha,
            'target_mean': target_mean,
            'a': a,
            'total_mc': total_mc,
            'mc_curves': mc_curves,
            'avalanche_exponent': crit_metrics['avalanche_exponent'],
            'branching_ratio': crit_metrics['branching_ratio'],
            'num_avalanches': crit_metrics['num_avalanches'],
            'threshold_sigma_used': crit_metrics['threshold_sigma_used'],
            'ks_distance': crit_metrics['ks_distance'],
            'xmin': crit_metrics['xmin'],
            'log_likelihood_ratio': crit_metrics['log_likelihood_ratio'],
            'p_value': crit_metrics['p_value'],
            'elapsed_time_seconds': elapsed_time,
            'state_memory_mb': state_memory_mb,
            'status': 'success',
            'error_message': ''
        }
        
    except Exception as e:
        elapsed_time = time.time() - t0
        logger.error(f"Failed configuration (N={N}, K={K}, rho={rho}, seed={seed}): {e}")
        result = {
            'N': N,
            'K': K,
            'rho': rho,
            'seed': seed,
            'topology': topology,
            'lambda_ring': lambda_ring,
            'T': T,
            'washout': washout,
            'k_max': k_max,
            'alpha': alpha,
            'target_mean': target_mean,
            'a': a,
            'total_mc': np.nan,
            'mc_curves': {},
            'avalanche_exponent': np.nan,
            'branching_ratio': np.nan,
            'num_avalanches': 0,
            'threshold_sigma_used': np.nan,
            'ks_distance': np.nan,
            'xmin': np.nan,
            'log_likelihood_ratio': np.nan,
            'p_value': np.nan,
            'elapsed_time_seconds': elapsed_time,
            'state_memory_mb': 0.0,
            'status': 'failed',
            'error_message': str(e)
        }
        
    return result

def load_checkpoints(checkpoint_file):
    """
    Load previously completed configurations.
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file {checkpoint_file}: {e}. Starting fresh.")
    return []

def save_checkpoint(checkpoint_file, results):
    """
    Save the current results list to checkpoint file.
    """
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_file}: {e}")

def run_parameter_sweep(configs, checkpoint_file='sweep_checkpoints.json', time_budget_minutes=30.0):
    """
    Run a parameter sweep over lists of configurations.
    
    Each config in `configs` is a dictionary with keys: N, K, rho, seed, topology.
    We estimate runtime scaling of large configs to avoid over-budget runs.
    
    To avoid oversubscription, we execute in parallel with ProcessPoolExecutor.
    Small-N (N <= 10000) are run with full concurrency (cpu_count).
    Large-N (N > 10000) are run with low concurrency (1 worker) to protect RAM and CPU resources.
    """
    completed_results = load_checkpoints(checkpoint_file)
    completed_keys = {(r['N'], r['K'], r['rho'], r['seed'], r['topology']) for r in completed_results if r['status'] == 'success'}
    
    # Filter out already completed configs
    configs_to_run = [c for c in configs if (c['N'], c['K'], c['rho'], c['seed'], c['topology']) not in completed_keys]
    
    if not configs_to_run:
        logger.info("All configurations in the sweep have already been completed!")
        return completed_results
    
    logger.info(f"Loaded {len(completed_results)} completed configs. Starting sweep for {len(configs_to_run)} pending configs.")
    
    # We will measure the average time taken for small N configurations to build a linear scaling profile
    # Let's group configurations by size N to handle them sequentially or scale-gated
    sizes = sorted(list({c['N'] for c in configs}))
    
    # Map to track reference execution time per unit size * K * T
    # Time ~ C * N * K * T
    scaling_coefficients = []
    
    # Let's divide configs into "small" (N <= 10000) and "large" (N > 10000)
    small_configs = [c for c in configs_to_run if c['N'] <= 10000]
    large_configs = [c for c in configs_to_run if c['N'] > 10000]
    
    # 1. Run small configs in parallel
    num_workers = multiprocessing.cpu_count()
    logger.info(f"Running small configs (N <= 10000) with {num_workers} parallel workers...")
    
    if small_configs:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all small configs
            future_to_config = {
                executor.submit(
                    run_single_config,
                    N=c['N'], K=c['K'], rho=c['rho'], seed=c['seed'], topology=c['topology'],
                    lambda_ring=c.get('lambda_ring', 0.1), T=c.get('T', 5000), washout=c.get('washout', 500),
                    k_max=c.get('k_max', 50), alpha=c.get('alpha', 1e-4), target_mean=c.get('target_mean', 0.1),
                    a=c.get('a', 0.3)
                ): c for c in small_configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    res = future.result()
                    completed_results.append(res)
                    save_checkpoint(checkpoint_file, completed_results)
                    
                    if res['status'] == 'success':
                        # Compute timing coefficient: C = Time / (N * K * T)
                        N, K, T = res['N'], res['K'], res['T']
                        C = res['elapsed_time_seconds'] / (N * K * T)
                        scaling_coefficients.append(C)
                        logger.info(f"Completed N={N}, K={K}, rho={res['rho']}, seed={res['seed']}. Time: {res['elapsed_time_seconds']:.2f}s, MC: {res['total_mc']:.2f}, Avalanches: {res['num_avalanches']}")
                except Exception as e:
                    logger.error(f"Executor error on config {config}: {e}")
                    
    # Calculate average coefficient to predict runtime for large configs
    if scaling_coefficients:
        avg_C = np.mean(scaling_coefficients)
    else:
        # Fallback default if no small configs were run this turn (loaded from checkpoint)
        avg_C = 2e-7  # Empirical baseline
        
    logger.info(f"Calibrated empirical cost coefficient: {avg_C:.4e} seconds / (N * K * T)")
    
    # 2. Run large configs sequentially (or 1 worker ProcessPool) to prevent OOM/thrashing, with timing gate
    if large_configs:
        logger.info(f"Evaluating large configs (N > 10000) with time-budget gate of {time_budget_minutes} mins...")
        
        # Group by size N to process smaller of the large ones first
        large_configs_sorted = sorted(large_configs, key=lambda c: (c['N'], c['K']))
        
        for c in large_configs_sorted:
            N = c['N']
            K = c['K']
            T = c.get('T', 5000)
            rho = c['rho']
            seed = c['seed']
            topology = c['topology']
            
            # Predict runtime
            projected_time_seconds = avg_C * N * K * T
            projected_time_minutes = projected_time_seconds / 60.0
            
            if projected_time_minutes > time_budget_minutes:
                logger.warning(f"SKIPPING config (N={N}, K={K}, rho={rho}, seed={seed}) because projected runtime {projected_time_minutes:.2f} mins exceeds budget of {time_budget_minutes} mins.")
                # Save a skipped result entry
                skipped_res = {
                    'N': N,
                    'K': K,
                    'rho': rho,
                    'seed': seed,
                    'topology': topology,
                    'lambda_ring': c.get('lambda_ring', 0.1),
                    'T': T,
                    'washout': c.get('washout', 500),
                    'k_max': c.get('k_max', 50),
                    'alpha': c.get('alpha', 1e-4),
                    'target_mean': c.get('target_mean', 0.1),
                    'a': c.get('a', 0.3),
                    'total_mc': np.nan,
                    'mc_curves': {},
                    'avalanche_exponent': np.nan,
                    'branching_ratio': np.nan,
                    'num_avalanches': 0,
                    'threshold_sigma_used': np.nan,
                    'ks_distance': np.nan,
                    'xmin': np.nan,
                    'log_likelihood_ratio': np.nan,
                    'p_value': np.nan,
                    'elapsed_time_seconds': 0.0,
                    'state_memory_mb': 0.0,
                    'status': 'skipped',
                    'error_message': f"Projected runtime {projected_time_minutes:.2f}m exceeded budget of {time_budget_minutes}m."
                }
                completed_results.append(skipped_res)
                save_checkpoint(checkpoint_file, completed_results)
                continue
                
            # Run the config using single worker
            logger.info(f"Running large config N={N}, K={K}, rho={rho}, seed={seed} (Projected: {projected_time_minutes:.2f} mins)...")
            res = run_single_config(
                N=N, K=K, rho=rho, seed=seed, topology=topology,
                lambda_ring=c.get('lambda_ring', 0.1), T=T, washout=c.get('washout', 500),
                k_max=c.get('k_max', 50), alpha=c.get('alpha', 1e-4), target_mean=c.get('target_mean', 0.1),
                a=c.get('a', 0.3)
            )
            completed_results.append(res)
            save_checkpoint(checkpoint_file, completed_results)
            
            if res['status'] == 'success':
                logger.info(f"Completed large N={N}, K={K}. Actual time: {res['elapsed_time_seconds']:.2f}s, MC: {res['total_mc']:.2f}, Avalanches: {res['num_avalanches']}")
            else:
                logger.error(f"Failed large N={N}, K={K}: {res['error_message']}")
                
    return completed_results
