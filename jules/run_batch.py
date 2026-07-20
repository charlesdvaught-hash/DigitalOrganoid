"""Serial, resumable batch driver: run pending embodied-sweep configs for a
time budget, then exit. Safe to call repeatedly (checkpointed)."""
import sys, time, json, logging
from organoid_simulator.creature_embodied_sweep import (
    run_single_config, load_checkpoints, save_checkpoint, DEFAULT_T)

logging.basicConfig(level=logging.WARNING)
CKPT = 'creature_embodied_sweep_checkpoints.json'
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0

configs = []
for N in [200, 600, 1200]:
    for K in [6, 10, 16]:
        for fi in [0.1, 0.2, 0.3]:
            for dale in [True, False]:
                for stdp_on in [True, False]:
                    for seed in [1, 2, 3]:
                        configs.append(dict(N=N, K=K, fi=fi, dale=dale, stdp_on=stdp_on, seed=seed, T=DEFAULT_T))

completed = load_checkpoints(CKPT)
done = {(r['N'], r['K'], r['fi'], r['dale'], r['stdp_on'], r['seed']) for r in completed if r['status'] == 'success'}
pending = [c for c in configs if (c['N'], c['K'], c['fi'], c['dale'], c['stdp_on'], c['seed']) not in done]
pending.sort(key=lambda c: (c['N'], c['K']))  # cheap first

t0 = time.time(); n = 0
for c in pending:
    if time.time() - t0 > budget:
        break
    res = run_single_config(**c)
    completed.append(res); save_checkpoint(CKPT, completed); n += 1
    print(f"{len(completed)}/{len(configs)}  N={res['N']} K={res['K']} fi={res['fi']} "
          f"dale={res['dale']} stdp={res['stdp_on']} s={res['seed']} "
          f"food={res.get('food_eaten','?')} rate={res.get('mean_pop_rate',0):.4f} {res['status']}")
print(f"batch done: {n} run this call, {len(completed)}/{len(configs)} total")
