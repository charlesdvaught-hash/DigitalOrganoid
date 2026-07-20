"""
Serial, resumable batch driver for the breedstock GA. Safe to call repeatedly
(checkpointed after every genome evaluation). Same pattern as run_batch.py: the
workspace kills long jobs, so each call runs for a time budget then exits.

Usage:
    python3 run_breedstock.py [budget_seconds] [max_generations] [n_workers]

    budget_seconds  wall-time this call runs before checkpointing and exiting
                    (default 150). Running locally with no sandbox cap, pass a
                    big number to blast through many generations unattended,
                    e.g.  python3 run_breedstock.py 7200 25
    n_workers       parallel sim processes. Defaults to os.cpu_count(). On an
                    8-core/16-thread CPU (e.g. 7800X3D), 8 keeps each worker on
                    a physical core for CPU-bound single-threaded numpy; try
                    16 to also use SMT. Override:  python3 run_breedstock.py 7200 25 8

Checkpoint:  breedstock_checkpoints.json   (current pop + fitnesses + rng + history)
Champions:   breedstock_champions.json     (top genomes, weights + provenance)
"""
import sys
import os
import json
import time
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from organoid_simulator import creature_breedstock as bs

logging.basicConfig(level=logging.WARNING)

# Config. Defaults are the full "medium" run; env vars let a cheaper demo share
# this same code path with its own checkpoint (used to prove compounding in a
# 2-core sandbox). Local full runs just use the defaults.
CKPT = os.environ.get('BS_CKPT', 'breedstock_checkpoints.json')
CHAMPS = os.environ.get('BS_CHAMPS', 'breedstock_champions.json')

POP_SIZE = int(os.environ.get('BS_POP', bs.POP_SIZE))
MAX_GENS = int(os.environ.get('BS_MAXGENS', 25))
_N_SEEDS = int(os.environ.get('BS_SEEDS', len(bs.DEFAULT_EVAL_SEEDS)))
EVAL_SEEDS = list(bs.DEFAULT_EVAL_SEEDS[:_N_SEEDS]) if _N_SEEDS <= len(bs.DEFAULT_EVAL_SEEDS) \
    else list(range(1000, 1000 + _N_SEEDS))
T = int(os.environ.get('BS_T', bs.DEFAULT_T))
N_WORKERS = os.cpu_count() or 2   # default: saturate the machine (override via argv[3])
N_CHAMPIONS = 5          # how many top genomes to persist in the champion store

# GA operator knobs (env-overridable so you can tune without editing code).
# Note: crossover of two trained weight vectors is often destructive; lowering
# BS_XOVER and BS_SIGMA usually raises the population mean and steadies the climb.
ELITE_FRAC = float(os.environ.get('BS_ELITE', bs.ELITE_FRAC))
MUT_SIGMA = float(os.environ.get('BS_SIGMA', bs.MUT_SIGMA))
MUT_RATE = float(os.environ.get('BS_MUTRATE', bs.MUT_RATE))
CROSSOVER_P = float(os.environ.get('BS_XOVER', bs.CROSSOVER_P))

# --------------------------------------------------------------------------- #
# Worker: build the frozen scaffold once per process, then evaluate genomes.
# --------------------------------------------------------------------------- #
_CTX = None


def _init_worker():
    global _CTX
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    _, _CTX = bs.build_scaffold()


def _eval_worker(args):
    idx, genome_list = args
    genome = np.asarray(genome_list, dtype=float)
    res = bs.evaluate_genome(_CTX, genome, seeds=EVAL_SEEDS, T=T)
    return idx, res


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def _save(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def new_state(ctx):
    rng = np.random.default_rng(12345)
    pop = bs.init_population(ctx, pop_size=POP_SIZE, sigma=bs.MUT_SIGMA, rng=rng)
    return dict(
        generation=0,
        scaffold=ctx['scaffold'],
        scaffold_seed=ctx['scaffold_seed'],
        n_genes=ctx['n_genes'],
        eval_seeds=EVAL_SEEDS,
        T=T,
        pop=[g.tolist() for g in pop],
        fitness=[None] * len(pop),        # per-genome fitness dict or None
        history=[],                       # per-generation scalar summaries
        rng_state=rng.bit_generator.state,
    )


def write_champions(ctx, state):
    """Persist the top genomes (weights + provenance) as the breedstock pool."""
    scored = [(f['fitness'], i) for i, f in enumerate(state['fitness']) if f]
    scored.sort(reverse=True)
    champs = []
    for rank, (fit, i) in enumerate(scored[:N_CHAMPIONS]):
        f = state['fitness'][i]
        champs.append(dict(
            rank=rank,
            fitness=f['fitness'],
            fitness_sd=f['fitness_sd'],
            mean_food=f['mean_food'],
            mean_survival=f['mean_survival'],
            generation=state['generation'],
            genome=state['pop'][i],       # plastic weights only
        ))
    out = dict(
        scaffold=state['scaffold'],
        scaffold_seed=state['scaffold_seed'],
        n_genes=state['n_genes'],
        eval_seeds=state['eval_seeds'],
        T=state['T'],
        survival_weight=bs.SURVIVAL_WEIGHT,
        note=("Genome = plastic-synapse weights on the fixed scaffold. Reflex "
              "weights are innate; rebuild topology with build_scaffold() and set "
              "net['weight'][~is_reflex] = genome."),
        champions=champs,
    )
    _save(CHAMPS, out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    max_gens = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_GENS
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else N_WORKERS

    _, ctx = bs.build_scaffold()
    state = _load(CKPT)
    if state is None:
        state = new_state(ctx)
        _save(CKPT, state)
        print(f"init: pop={POP_SIZE} n_genes={ctx['n_genes']} "
              f"eval_seeds={len(EVAL_SEEDS)} T={T} workers={n_workers}")

    t0 = time.time()
    n_evals = 0

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        while time.time() - t0 < budget and state['generation'] < max_gens:
            pending = [i for i, f in enumerate(state['fitness']) if f is None]

            if pending:
                # Evaluate pending genomes in parallel chunks; checkpoint after
                # each chunk so a killed call loses at most n_workers evals.
                for k in range(0, len(pending), n_workers):
                    if time.time() - t0 > budget:
                        break
                    chunk = pending[k:k + n_workers]
                    args = [(i, state['pop'][i]) for i in chunk]
                    for idx, res in ex.map(_eval_worker, args):
                        state['fitness'][idx] = res
                        n_evals += 1
                    _save(CKPT, state)
                continue

            # All genomes in this generation are scored -> record, breed, advance.
            fits = [f['fitness'] for f in state['fitness']]
            order = list(np.argsort(fits)[::-1])
            best = state['fitness'][order[0]]
            summary = dict(
                generation=state['generation'],
                best_fitness=best['fitness'],
                best_food=best['mean_food'],
                best_survival=best['mean_survival'],
                best_fitness_sd=best['fitness_sd'],
                mean_fitness=float(np.mean(fits)),
                median_fitness=float(np.median(fits)),
            )
            state['history'].append(summary)
            write_champions(ctx, state)
            print(f"gen {state['generation']:2d}  best={best['fitness']:.3f} "
                  f"(food={best['mean_food']:.2f} surv={best['mean_survival']:.0f}) "
                  f"mean={summary['mean_fitness']:.3f}")

            if state['generation'] + 1 >= max_gens:
                state['generation'] += 1
                _save(CKPT, state)
                break

            # Breed next generation with restored rng stream.
            rng = np.random.default_rng()
            rng.bit_generator.state = state['rng_state']
            ranked = [np.asarray(state['pop'][i], dtype=float) for i in order]
            fits_ranked = [state['fitness'][i] for i in order]
            next_pop = bs.breed_next_generation(
                ctx, ranked, pop_size=POP_SIZE, elite_frac=ELITE_FRAC,
                sigma=MUT_SIGMA, mut_rate=MUT_RATE,
                crossover_p=CROSSOVER_P, rng=rng)

            # Elitism: champion (index 0 of next_pop) keeps its known fitness.
            new_fit = [None] * POP_SIZE
            new_fit[0] = fits_ranked[0]
            state['pop'] = [g.tolist() for g in next_pop]
            state['fitness'] = new_fit
            state['generation'] += 1
            state['rng_state'] = rng.bit_generator.state
            _save(CKPT, state)

    done = sum(1 for f in state['fitness'] if f is not None)
    print(f"batch done: {n_evals} evals this call | gen={state['generation']} "
          f"| {done}/{POP_SIZE} scored in current gen | {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
