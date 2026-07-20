import unittest
import numpy as np

from organoid_simulator.creature_embodied import (
    build_creature_network, run_creature_simulation, POOLS, W_MAX, _sensors,
)
from organoid_simulator.creature_metrics import (
    firing_stats, avalanche_stats, weight_drift, metabolic_stats,
)


class TestEmbodiedCreature(unittest.TestCase):

    def test_build_network(self):
        N, K = 200, 6
        net = build_creature_network(N, K, fi=0.2, seed=42)
        self.assertEqual(net['N'], N)
        for key in ('types', 'subtypes', 'a', 'b', 'c', 'd', 'v', 'u',
                    'px', 'py', 'pz'):
            self.assertEqual(len(net[key]), N)
        # synapse arrays share length; reflex mask present
        L = len(net['pre'])
        self.assertEqual(len(net['post']), L)
        self.assertEqual(len(net['weight']), L)
        self.assertEqual(len(net['is_reflex']), L)
        self.assertTrue(net['is_reflex'].sum() > 0)          # innate reflexes exist
        self.assertTrue((~net['is_reflex']).sum() > 0)       # learned synapses exist

    def test_pool_neurons_excitatory(self):
        net = build_creature_network(200, 6, fi=0.4, seed=1)
        for (lo, hi) in POOLS.values():
            self.assertTrue(np.all(net['types'][lo:hi] == 1))

    def test_dale_sign(self):
        net = build_creature_network(200, 6, fi=0.3, seed=7)
        pre_types = net['types'][net['pre']]
        signs = np.sign(net['weight'])
        self.assertTrue(np.all(signs[pre_types > 0] >= 0))
        self.assertTrue(np.all(signs[pre_types < 0] <= 0))

    def test_sensors_directional(self):
        rng = np.random.default_rng(0)
        food = np.array([[0.8, 0.5]])   # to the +x
        L0, R0, W, fc, near = _sensors(0.5, 0.5, 0.0, food, rng)   # facing food
        self.assertGreater(L0 + R0, 0.3)
        self.assertGreater(fc, 0.0)
        # facing away -> weaker odor
        La, Ra, _, _, _ = _sensors(0.5, 0.5, np.pi, food, rng)
        self.assertLess(La + Ra, L0 + R0)

    def test_no_stdp_keeps_learned_weights_fixed(self):
        net = build_creature_network(200, 6, fi=0.2, seed=3)
        res = run_creature_simulation(net, 100, dale=True, stdp_on=False, seed=3)
        plastic = ~net['is_reflex']
        np.testing.assert_allclose(net['weight'][plastic], res['final_weight'][plastic])

    def test_reflex_weights_never_change(self):
        net = build_creature_network(200, 6, fi=0.2, seed=5)
        rfx = net['is_reflex']
        res = run_creature_simulation(net, 300, dale=True, stdp_on=True, seed=5)
        np.testing.assert_allclose(net['weight'][rfx], res['final_weight'][rfx])

    def test_dale_bounds_after_stdp(self):
        net = build_creature_network(200, 6, fi=0.3, seed=9)
        res = run_creature_simulation(net, 400, dale=True, stdp_on=True, seed=9)
        w = res['final_weight']; pre_types = net['types'][net['pre']]
        self.assertTrue(np.all(w[pre_types > 0] >= -1e-9))
        self.assertTrue(np.all(w[pre_types > 0] <= W_MAX + 1e-9))
        self.assertTrue(np.all(w[pre_types < 0] <= 1e-9))
        self.assertTrue(np.all(w[pre_types < 0] >= -W_MAX - 1e-9))

    def test_behavior_metrics_present(self):
        net = build_creature_network(600, 10, fi=0.2, seed=2)
        res = run_creature_simulation(net, 1000, dale=True, stdp_on=True, seed=2)
        b = res['behavior']
        for key in ('food_eaten', 'distance_traveled', 'mean_speed',
                    'final_body_energy', 'survival_steps', 'alive_end',
                    'food_per_1k_steps', 'food_per_distance'):
            self.assertIn(key, b)
        self.assertGreaterEqual(b['distance_traveled'], 0.0)
        self.assertGreaterEqual(b['food_eaten'], 0)
        # creature actually moved
        self.assertGreater(b['distance_traveled'], 0.0)

    def test_metrics_plug_in(self):
        net = build_creature_network(300, 6, fi=0.2, seed=4)
        res = run_creature_simulation(net, 500, dale=True, stdp_on=True, seed=4)
        m = {}
        m.update(firing_stats(res['spikes']))
        m.update(avalanche_stats(res['spikes']))
        m.update(weight_drift(net, res))
        m.update(metabolic_stats(res))
        self.assertGreaterEqual(m['mean_pop_rate'], 0.0)
        self.assertIn('weight_drift', m)


if __name__ == '__main__':
    unittest.main()
