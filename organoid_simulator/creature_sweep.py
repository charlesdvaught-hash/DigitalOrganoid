import os
import json
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from organoid_simulator.creature_model import build_creature_network, run_creature_simulation
from organoid_simulator.creature_metrics import firing_stats, avalanche_stats, weight_drift, metabolic_stats

logger = logging.getLogger(__name__)


def enforce_single_thread():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def run_single_config(N, K, fi, dale, stdp_on, seed, T=2000, record_every=1):
    enforce_single_thread()
    t0 = time.time()
    try:
        net = build_creature_network(N, K, fi=fi, seed=seed)
        result = run_creature_simulation(net, T, dale=dale, stdp_on=stdp_on,
                                          seed=seed + 1000, record_every=record_every)

        metrics = {}
        metrics.update(firing_stats(result['spikes']))
        metrics.update(avalanche_stats(result['spikes']))
        metrics.update(weight_drift(net, result))
        metrics.update(metabolic_stats(result))

        elapsed = time.time() - t0
        res = dict(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=T,
                   elapsed_time_seconds=elapsed, status='success', error_message='')
        res.update(metrics)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Failed config (N={N}, K={K}, fi={fi}, dale={dale}, "
                     f"stdp={stdp_on}, seed={seed}): {e}")
        res = dict(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=T,
                   elapsed_time_seconds=elapsed, status='failed', error_message=str(e))
    return res


def load_checkpoints(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}. Starting fresh.")
    return []


def save_checkpoint(path, results):
    try:
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {e}")


def run_parameter_sweep(configs, checkpoint_file='creature_sweep_checkpoints.json', time_budget_minutes=30.0):
    """
    configs: list of dicts with keys N, K, fi, dale, stdp_on, seed (T optional).
    Mirrors sweep.py: parallel small-N, checkpointed, resumable, with cost scaling projection for large N.
    """
    completed = load_checkpoints(checkpoint_file)
    done_keys = {(r['N'], r['K'], r['fi'], r['dale'], r['stdp_on'], r['seed'])
                 for r in completed if r['status'] == 'success'}

    pending = [c for c in configs
               if (c['N'], c['K'], c['fi'], c['dale'], c['stdp_on'], c['seed']) not in done_keys]

    if not pending:
        logger.info("All configurations already completed.")
        return completed

    logger.info(f"{len(completed)} completed, {len(pending)} pending.")

    # Divide configs into small (N <= 600) and large (N > 600) to estimate scaling coefficient
    small_configs = [c for c in pending if c['N'] <= 600]
    large_configs = [c for c in pending if c['N'] > 600]

    scaling_coefficients = []

    # 1. Run small configs in parallel
    num_workers = multiprocessing.cpu_count()
    if small_configs:
        logger.info(f"Running small configs (N <= 600) with {num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(run_single_config, N=c['N'], K=c['K'], fi=c['fi'],
                                 dale=c['dale'], stdp_on=c['stdp_on'], seed=c['seed'],
                                 T=c.get('T', 2000)): c
                for c in small_configs
            }
            for future in as_completed(futures):
                c = futures[future]
                try:
                    res = future.result()
                    completed.append(res)
                    save_checkpoint(checkpoint_file, completed)
                    if res['status'] == 'success':
                        N, K, T = res['N'], res['K'], res['T']
                        C = res['elapsed_time_seconds'] / (N * K * T)
                        scaling_coefficients.append(C)
                        logger.info(f"Done N={res['N']} K={res['K']} fi={res['fi']} "
                                    f"dale={res['dale']} stdp={res['stdp_on']} seed={res['seed']} "
                                    f"rate={res['mean_pop_rate']:.4f} "
                                    f"drift={res['weight_drift']:.4f} "
                                    f"time={res['elapsed_time_seconds']:.1f}s")
                except Exception as e:
                    logger.error(f"Executor error on {c}: {e}")

    # Calculate average coefficient to predict runtime for large configs
    if scaling_coefficients:
        avg_C = np.mean(scaling_coefficients)
    else:
        avg_C = 2e-7  # Empirical baseline default if all small ones are already in checkpoint

    logger.info(f"Calibrated empirical cost coefficient: {avg_C:.4e} seconds / (N * K * T)")

    # 2. Run large configs sequentially (or 1 worker ProcessPool) to prevent memory bloating, with timing gate
    if large_configs:
        logger.info(f"Evaluating large configs (N > 600) with time-budget gate of {time_budget_minutes} mins...")
        large_configs_sorted = sorted(large_configs, key=lambda c: (c['N'], c['K']))

        for c in large_configs_sorted:
            N = c['N']
            K = c['K']
            fi = c['fi']
            dale = c['dale']
            stdp_on = c['stdp_on']
            seed = c['seed']
            T = c.get('T', 2000)

            # Predict runtime
            projected_time_seconds = avg_C * N * K * T
            projected_time_minutes = projected_time_seconds / 60.0

            if projected_time_minutes > time_budget_minutes:
                logger.warning(f"SKIPPING config (N={N}, K={K}, fi={fi}, dale={dale}, stdp_on={stdp_on}, seed={seed}) "
                               f"because projected runtime {projected_time_minutes:.2f} mins exceeds budget of {time_budget_minutes} mins.")
                skipped_res = dict(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=T,
                                   elapsed_time_seconds=0.0, status='skipped',
                                   error_message=f"Projected runtime {projected_time_minutes:.2f}m exceeded budget of {time_budget_minutes}m.")
                completed.append(skipped_res)
                save_checkpoint(checkpoint_file, completed)
                continue

            logger.info(f"Running large config N={N}, K={K}, fi={fi}, dale={dale}, stdp_on={stdp_on}, seed={seed} "
                        f"(Projected: {projected_time_minutes:.2f} mins)...")
            res = run_single_config(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=T)
            completed.append(res)
            save_checkpoint(checkpoint_file, completed)

            if res['status'] == 'success':
                logger.info(f"Completed large N={N}, K={K}. Actual time: {res['elapsed_time_seconds']:.2f}s, rate: {res['mean_pop_rate']:.4f}")
            else:
                logger.error(f"Failed large N={N}, K={K}: {res['error_message']}")

    return completed


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    configs = []
    for N in [200, 600, 1200]:
        for K in [6, 10, 16]:
            for fi in [0.1, 0.2, 0.3]:
                for dale in [True, False]:
                    for stdp_on in [True, False]:
                        for seed in [1, 2, 3]:
                            configs.append(dict(N=N, K=K, fi=fi, dale=dale,
                                                 stdp_on=stdp_on, seed=seed, T=2000))

    results = run_parameter_sweep(configs)
    logger.info(f"Sweep complete: {len(results)} total results.")
