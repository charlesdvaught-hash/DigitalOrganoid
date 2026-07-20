"""
Parameter sweep over the EMBODIED creature (creature_embodied.py) — the real
index.html closed loop (arena, sensory pools, motors, cost-of-transport,
dopamine reward), not the noise-driven creature_model.py abstraction.

Mirrors creature_sweep.py: parallel small-N, sequential budget-gated large-N,
checkpointed and resumable. Records neural metrics (firing / avalanche / weight
drift / metabolism) AND creature behavior metrics (food eaten, distance, speed,
survival, foraging efficiency).
"""
import os
import json
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from organoid_simulator.creature_embodied import build_creature_network, run_creature_simulation
from organoid_simulator.creature_metrics import firing_stats, avalanche_stats, weight_drift, metabolic_stats

logger = logging.getLogger(__name__)

DEFAULT_T = 6000


def enforce_single_thread():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def run_single_config(N, K, fi, dale, stdp_on, seed, T=DEFAULT_T, record_every=1):
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
        metrics.update(result['behavior'])  # embodied behavior metrics

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
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {e}")


def run_parameter_sweep(configs, checkpoint_file='creature_embodied_sweep_checkpoints.json',
                        time_budget_minutes=30.0):
    completed = load_checkpoints(checkpoint_file)
    done_keys = {(r['N'], r['K'], r['fi'], r['dale'], r['stdp_on'], r['seed'])
                 for r in completed if r['status'] == 'success'}
    pending = [c for c in configs
               if (c['N'], c['K'], c['fi'], c['dale'], c['stdp_on'], c['seed']) not in done_keys]

    if not pending:
        logger.info("All configurations already completed.")
        return completed
    logger.info(f"{len(completed)} completed, {len(pending)} pending.")

    small_configs = [c for c in pending if c['N'] <= 600]
    large_configs = [c for c in pending if c['N'] > 600]
    scaling_coefficients = []

    num_workers = multiprocessing.cpu_count()
    if small_configs:
        logger.info(f"Running small configs (N <= 600) with {num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(run_single_config, N=c['N'], K=c['K'], fi=c['fi'],
                                dale=c['dale'], stdp_on=c['stdp_on'], seed=c['seed'],
                                T=c.get('T', DEFAULT_T)): c
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
                        scaling_coefficients.append(res['elapsed_time_seconds'] / (N * K * T))
                        logger.info(f"Done N={res['N']} K={res['K']} fi={res['fi']} "
                                    f"dale={res['dale']} stdp={res['stdp_on']} seed={res['seed']} "
                                    f"rate={res['mean_pop_rate']:.4f} food={res['food_eaten']} "
                                    f"time={res['elapsed_time_seconds']:.1f}s")
                except Exception as e:
                    logger.error(f"Executor error on {c}: {e}")

    avg_C = np.mean(scaling_coefficients) if scaling_coefficients else 2e-7
    logger.info(f"Calibrated empirical cost coefficient: {avg_C:.4e} seconds / (N * K * T)")

    if large_configs:
        logger.info(f"Evaluating large configs (N > 600) with time-budget gate of {time_budget_minutes} mins...")
        for c in sorted(large_configs, key=lambda c: (c['N'], c['K'])):
            N, K, fi = c['N'], c['K'], c['fi']
            dale, stdp_on, seed = c['dale'], c['stdp_on'], c['seed']
            T = c.get('T', DEFAULT_T)
            projected_min = avg_C * N * K * T / 60.0
            if projected_min > time_budget_minutes:
                logger.warning(f"SKIPPING (N={N}, K={K}, fi={fi}, dale={dale}, stdp_on={stdp_on}, seed={seed}) "
                               f"projected {projected_min:.2f}m > budget {time_budget_minutes}m.")
                completed.append(dict(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=T,
                                      elapsed_time_seconds=0.0, status='skipped',
                                      error_message=f"Projected {projected_min:.2f}m exceeded {time_budget_minutes}m."))
                save_checkpoint(checkpoint_file, completed)
                continue
            logger.info(f"Running large N={N}, K={K}, fi={fi}, dale={dale}, stdp_on={stdp_on}, seed={seed} "
                        f"(Projected {projected_min:.2f}m)...")
            res = run_single_config(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=T)
            completed.append(res)
            save_checkpoint(checkpoint_file, completed)
            if res['status'] == 'success':
                logger.info(f"Completed large N={N}, K={K}. time {res['elapsed_time_seconds']:.1f}s, "
                            f"rate {res['mean_pop_rate']:.4f}, food {res['food_eaten']}")
            else:
                logger.error(f"Failed large N={N}, K={K}: {res['error_message']}")

    return completed


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    configs = []
    for N in [200, 600, 1200]:
        for K in [6, 10, 16]:
            for fi in [0.1, 0.2, 0.3]:
                for dale in [True, False]:
                    for stdp_on in [True, False]:
                        for seed in [1, 2, 3]:
                            configs.append(dict(N=N, K=K, fi=fi, dale=dale,
                                                stdp_on=stdp_on, seed=seed, T=DEFAULT_T))
    results = run_parameter_sweep(configs)
    logger.info(f"Sweep complete: {len(results)} total results.")
