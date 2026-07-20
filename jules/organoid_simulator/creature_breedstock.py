"""
Breedstock GA — evolve the creature's *synaptic weights* (the trained brain),
not just hyperparameters. Offspring inherit a weight vector, so selection
compounds across generations (real breedstock, not just good body plans).

Builds directly on creature_embodied.py. See BREEDSTOCK_PLAN.md.

Design decisions (settled at thread start):
  - Fixed genome scaffold from the sweep: N=600, K=6, fi=0.2, dale=True.
    Topology (pre/post/is_reflex) is FIXED and shared by every genome.
  - Genome = the plastic-synapse weight vector (net['weight'][~is_reflex]).
    Reflex weights (sL->mL, sR->mR, sW->mL/mR) stay innate/fixed, never evolved.
  - Blend fitness = food_eaten + 0.0003 * survival_steps, averaged over a FIXED
    set of arena seeds (default 15). Fixed seeds => deterministic, paired
    fitness per genome: clean selection, exact elitism, no re-evaluation of
    surviving parents. A held-out seed set validates the champion at the end.
  - STDP OFF during evaluation: faster, and it scores the innate evolved genome
    rather than one-life learning (the sweep showed STDP ~no behavioral effect).
  - Breeding: truncation selection (keep top ~20%), per-synapse uniform
    crossover from two parents, Gaussian mutation respecting Dale sign clamps
    and W_MAX. Elitism carries the champion genome unchanged.
"""
import numpy as np

from organoid_simulator.creature_embodied import (
    build_creature_network, run_creature_simulation, W_MAX,
)

# Fixed scaffold (from the 324-run embodied sweep).
SCAFFOLD = dict(N=600, K=6, fi=0.2, dale=True)
SCAFFOLD_SEED = 42          # topology is built once with this seed and frozen

# Fitness / evaluation defaults.
DEFAULT_T = 6000
DEFAULT_EVAL_SEEDS = tuple(range(1000, 1015))     # 15 fixed arena seeds
DEFAULT_HELDOUT_SEEDS = tuple(range(5000, 5015))  # 15 fresh validation seeds
SURVIVAL_WEIGHT = 0.0003    # blend: food + 0.0003 * survival_steps

# GA defaults (medium scale).
POP_SIZE = 48
ELITE_FRAC = 0.20           # top 20% become parents
MUT_SIGMA = 0.05            # Gaussian mutation std (in weight units)
MUT_RATE = 0.5              # fraction of plastic synapses perturbed per offspring
CROSSOVER_P = 0.9           # prob an offspring is a crossover (else clone+mutate)


# --------------------------------------------------------------------------- #
# Scaffold + genome context
# --------------------------------------------------------------------------- #
def build_scaffold(scaffold=SCAFFOLD, seed=SCAFFOLD_SEED):
    """Build the frozen topology once. Returns (net, ctx) where ctx holds the
    fixed masks the GA needs: which synapses are plastic and their Dale signs."""
    net = build_creature_network(scaffold['N'], scaffold['K'],
                                 fi=scaffold['fi'], seed=seed)
    plastic = ~net['is_reflex']
    types = net['types']
    pre = net['pre']
    # Dale sign of each PLASTIC synapse, from the presynaptic neuron's type.
    exc = types[pre][plastic] > 0     # weight in [0, W_MAX]
    inh = types[pre][plastic] < 0     # weight in [-W_MAX, 0]
    ctx = dict(
        net=net,
        plastic=plastic,
        n_genes=int(plastic.sum()),
        exc=exc,
        inh=inh,
        lo=np.where(exc, 0.0, -W_MAX),   # per-gene lower clamp
        hi=np.where(exc, W_MAX, 0.0),    # per-gene upper clamp
        base_genome=net['weight'][plastic].copy(),
        scaffold=dict(scaffold),
        scaffold_seed=seed,
    )
    return net, ctx


def clamp_genome(genome, ctx):
    """Respect Dale sign clamps and W_MAX per gene."""
    return np.clip(genome, ctx['lo'], ctx['hi'])


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _apply_genome(ctx, genome):
    """Return a net dict whose plastic weights are `genome` (topology shared)."""
    net = ctx['net']
    w = net['weight'].copy()
    w[ctx['plastic']] = genome
    child = dict(net)
    child['weight'] = w
    return child


def evaluate_genome(ctx, genome, seeds=DEFAULT_EVAL_SEEDS, T=DEFAULT_T,
                    survival_weight=SURVIVAL_WEIGHT, stdp_on=False):
    """Blend fitness averaged over fixed arena seeds. Deterministic per genome.
    Returns dict(fitness, fitness_sd, mean_food, mean_survival, n)."""
    net = _apply_genome(ctx, genome)
    dale = ctx['scaffold']['dale']
    scores, foods, survs = [], [], []
    for s in seeds:
        r = run_creature_simulation(net, T, dale=dale, stdp_on=stdp_on,
                                    seed=s, record_every=200)
        b = r['behavior']
        scores.append(b['food_eaten'] + survival_weight * b['survival_steps'])
        foods.append(b['food_eaten'])
        survs.append(b['survival_steps'])
    scores = np.asarray(scores, dtype=float)
    return dict(
        fitness=float(scores.mean()),
        fitness_sd=float(scores.std()),
        mean_food=float(np.mean(foods)),
        mean_survival=float(np.mean(survs)),
        n=len(seeds),
    )


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #
def init_population(ctx, pop_size=POP_SIZE, sigma=MUT_SIGMA, rng=None):
    """Gen-0: the base weights plus (pop_size-1) mutated variants of them."""
    rng = rng or np.random.default_rng(0)
    base = ctx['base_genome']
    pop = [base.copy()]
    for _ in range(pop_size - 1):
        g = base + rng.standard_normal(ctx['n_genes']) * sigma
        pop.append(clamp_genome(g, ctx))
    return pop


def crossover(ctx, p1, p2, rng):
    """Per-synapse uniform crossover: each gene taken from a random parent."""
    mask = rng.random(ctx['n_genes']) < 0.5
    child = np.where(mask, p1, p2)
    return clamp_genome(child, ctx)


def mutate(ctx, genome, sigma=MUT_SIGMA, rate=MUT_RATE, rng=None):
    """Gaussian noise on a random subset of plastic weights, Dale/W_MAX clamped."""
    rng = rng or np.random.default_rng()
    child = genome.copy()
    hit = rng.random(ctx['n_genes']) < rate
    child[hit] += rng.standard_normal(int(hit.sum())) * sigma
    return clamp_genome(child, ctx)


def breed_next_generation(ctx, ranked_pop, pop_size=POP_SIZE,
                          elite_frac=ELITE_FRAC, sigma=MUT_SIGMA,
                          mut_rate=MUT_RATE, crossover_p=CROSSOVER_P, rng=None):
    """ranked_pop: list of genomes sorted best-first. Returns the next
    generation's genomes (list). Champion carried unchanged (elitism)."""
    rng = rng or np.random.default_rng()
    n_elite = max(2, int(round(pop_size * elite_frac)))
    parents = ranked_pop[:n_elite]

    next_pop = [parents[0].copy()]                # elitism: champion unchanged
    while len(next_pop) < pop_size:
        if rng.random() < crossover_p and len(parents) >= 2:
            i, j = rng.choice(len(parents), size=2, replace=False)
            child = crossover(ctx, parents[i], parents[j], rng)
        else:
            child = parents[rng.integers(len(parents))].copy()
        child = mutate(ctx, child, sigma=sigma, rate=mut_rate, rng=rng)
        next_pop.append(child)
    return next_pop
