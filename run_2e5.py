"""
Focused single-size runner for N = 2x10^5.

Bypasses the sweep's time-budget skip logic by calling run_single_config directly.
Runs K=50 and K=100 at rho=0.95, random topology, one seed each, and prints a
clean summary plus saves results to results_2e5.json.

Usage:
    python run_2e5.py
    python run_2e5.py --k 50 --seeds 1 2 3      # override K / seeds
"""
import os
# Keep BLAS single-threaded per process (matches the sweep; avoids oversubscription).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
import argparse
import logging
import numpy as np

from organoid_simulator.sweep import run_single_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_2e5")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--k", type=int, nargs="+", default=[50, 100])
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--topology", type=str, default="random")
    p.add_argument("--out", type=str, default="results_2e5.json")
    args = p.parse_args()

    results = []
    for K in args.k:
        for seed in args.seeds:
            logger.info(f"=== Running N={args.N}, K={K}, rho={args.rho}, "
                        f"seed={seed}, topology={args.topology} ===")
            t0 = time.time()
            r = run_single_config(
                N=args.N, K=K, rho=args.rho, seed=seed,
                topology=args.topology,
            )
            r["wall_clock_s"] = round(time.time() - t0, 1)
            results.append(r)

            # Clean per-run summary
            mc_curve = r.get("mc_curves", {}) or {}
            first5 = [round(mc_curve.get(str(k), mc_curve.get(k, 0.0)), 3)
                      for k in range(1, 6)]
            print("\n" + "-" * 60)
            print(f"N={r['N']}  K={r['K']}  rho={r['rho']}  seed={r['seed']}  "
                  f"status={r['status']}")
            if r["status"] == "success":
                print(f"  Total MC          : {r['total_mc']:.3f}")
                print(f"  r^2(k) for k=1..5 : {first5}")
                print(f"  Avalanche exponent: {r['avalanche_exponent']}")
                print(f"  Branching ratio   : {r['branching_ratio']:.3f}")
                print(f"  # avalanches      : {r['num_avalanches']}")
                print(f"  power-law LLR / p : {r['log_likelihood_ratio']} / {r['p_value']}")
                print(f"  state matrix (MB) : {r['state_memory_mb']:.0f}")
            else:
                print(f"  ERROR: {r['error_message']}")
            print(f"  wall clock (s)    : {r['wall_clock_s']}")
            print("-" * 60 + "\n")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None
                  if isinstance(o, float) and np.isnan(o) else o)
    logger.info(f"Saved {len(results)} result(s) to {args.out}")


if __name__ == "__main__":
    main()
