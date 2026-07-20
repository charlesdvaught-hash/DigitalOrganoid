"""Unit tests for the breedstock GA operators (creature_breedstock.py)."""
import numpy as np
import pytest

from organoid_simulator import creature_breedstock as bs
from organoid_simulator.creature_embodied import W_MAX


@pytest.fixture(scope="module")
def scaffold():
    net, ctx = bs.build_scaffold()
    return net, ctx


def test_genome_is_plastic_weights_only(scaffold):
    net, ctx = scaffold
    plastic = ~net['is_reflex']
    assert ctx['n_genes'] == int(plastic.sum())
    assert ctx['n_genes'] > 0
    # Reflex synapses exist and are excluded from the genome.
    assert net['is_reflex'].sum() > 0
    assert len(ctx['base_genome']) == ctx['n_genes']


def test_apply_genome_leaves_reflex_untouched(scaffold):
    net, ctx = scaffold
    reflex = net['is_reflex']
    genome = ctx['base_genome'] + 1.0            # arbitrary change
    genome = bs.clamp_genome(genome, ctx)
    child = bs._apply_genome(ctx, genome)
    # Reflex weights identical to the frozen topology.
    assert np.array_equal(child['weight'][reflex], net['weight'][reflex])
    # Plastic weights replaced by the genome.
    assert np.array_equal(child['weight'][~reflex], genome)
    # Topology (pre/post/is_reflex) shared, not copied-and-mutated.
    assert child['pre'] is net['pre']
    assert child['post'] is net['post']


def test_clamp_respects_dale_signs_and_wmax(scaffold):
    net, ctx = scaffold
    # Push every gene wildly out of bounds in both directions.
    hot = bs.clamp_genome(np.full(ctx['n_genes'], 999.0), ctx)
    cold = bs.clamp_genome(np.full(ctx['n_genes'], -999.0), ctx)
    # Excitatory genes clamp to [0, W_MAX]; inhibitory to [-W_MAX, 0].
    assert np.all(hot[ctx['exc']] == W_MAX)
    assert np.all(hot[ctx['inh']] == 0.0)
    assert np.all(cold[ctx['exc']] == 0.0)
    assert np.all(cold[ctx['inh']] == -W_MAX)
    # No excitatory weight ever negative, no inhibitory ever positive.
    for g in (hot, cold):
        assert np.all(g[ctx['exc']] >= 0.0)
        assert np.all(g[ctx['inh']] <= 0.0)
        assert np.all(np.abs(g) <= W_MAX + 1e-12)


def test_mutation_stays_in_bounds(scaffold):
    net, ctx = scaffold
    rng = np.random.default_rng(1)
    g = ctx['base_genome'].copy()
    for _ in range(20):
        g = bs.mutate(ctx, g, sigma=0.5, rate=1.0, rng=rng)
        assert np.all(g[ctx['exc']] >= 0.0) and np.all(g[ctx['exc']] <= W_MAX)
        assert np.all(g[ctx['inh']] <= 0.0) and np.all(g[ctx['inh']] >= -W_MAX)


def test_crossover_preserves_topology_and_genes(scaffold):
    net, ctx = scaffold
    rng = np.random.default_rng(2)
    p1 = bs.clamp_genome(ctx['base_genome'] + 0.3, ctx)
    p2 = bs.clamp_genome(ctx['base_genome'] - 0.3, ctx)
    child = bs.crossover(ctx, p1, p2, rng)
    assert child.shape == p1.shape == (ctx['n_genes'],)
    # Every gene came from one of the two parents (before clamping both are
    # already in-bounds, so clamp is a no-op here).
    from_p1 = child == p1
    from_p2 = child == p2
    assert np.all(from_p1 | from_p2)
    # Dale signs preserved.
    assert np.all(child[ctx['exc']] >= 0.0) and np.all(child[ctx['inh']] <= 0.0)


def test_elitism_carries_champion_unchanged(scaffold):
    net, ctx = scaffold
    rng = np.random.default_rng(3)
    POP = 12
    ranked = [bs.clamp_genome(ctx['base_genome'] + d, ctx)
              for d in np.linspace(0.0, 0.2, POP)]
    nxt = bs.breed_next_generation(ctx, ranked, pop_size=POP, rng=rng)
    assert len(nxt) == POP
    # Champion (best, index 0) carried forward byte-identical.
    assert np.array_equal(nxt[0], ranked[0])


def test_init_population_shape_and_bounds(scaffold):
    net, ctx = scaffold
    pop = bs.init_population(ctx, pop_size=10, sigma=0.1,
                             rng=np.random.default_rng(4))
    assert len(pop) == 10
    # First member is the untouched base genome.
    assert np.array_equal(pop[0], ctx['base_genome'])
    for g in pop:
        assert g.shape == (ctx['n_genes'],)
        assert np.all(g[ctx['exc']] >= 0.0) and np.all(g[ctx['inh']] <= 0.0)


def test_fitness_is_deterministic_for_fixed_seeds(scaffold):
    net, ctx = scaffold
    g = ctx['base_genome']
    a = bs.evaluate_genome(ctx, g, seeds=(1000, 1001), T=1500)
    b = bs.evaluate_genome(ctx, g, seeds=(1000, 1001), T=1500)
    assert a['fitness'] == b['fitness']
    assert a['mean_food'] == b['mean_food']
    assert a['mean_survival'] == b['mean_survival']
