import unittest
import numpy as np
import os
import json

from organoid_simulator.creature_model import (
    build_creature_network,
    run_creature_simulation,
    W_MAX
)
from organoid_simulator.creature_metrics import (
    firing_stats,
    avalanche_stats,
    weight_drift,
    metabolic_stats
)


class TestCreatureSpikingSimulation(unittest.TestCase):

    def test_build_network(self):
        N, K = 100, 6
        fi = 0.2
        net = build_creature_network(N, K, fi=fi, seed=42)

        self.assertEqual(net['N'], N)
        self.assertEqual(net['K'], K)
        self.assertEqual(len(net['types']), N)
        self.assertEqual(len(net['subtypes']), N)
        self.assertEqual(len(net['a']), N)
        self.assertEqual(len(net['b']), N)
        self.assertEqual(len(net['c']), N)
        self.assertEqual(len(net['d']), N)
        self.assertEqual(len(net['v']), N)
        self.assertEqual(len(net['u']), N)
        self.assertEqual(len(net['pre']), N * K)
        self.assertEqual(len(net['post']), N * K)
        self.assertEqual(len(net['weight']), N * K)

        # Check excitatory / inhibitory split
        # We expect a fraction ~fi of elements in types to be -1
        inh_count = np.sum(net['types'] == -1)
        self.assertGreaterEqual(inh_count, 0)
        self.assertLessEqual(inh_count, N)

        # Dale's law constraint: presynaptic type should dictate synapse weight sign
        pre_types = net['types'][net['pre']]
        syn_signs = np.sign(net['weight'])
        # A positive weight or zero can exist, but signs should not be strictly opposite to presynaptic type
        for pt, s_sign in zip(pre_types, syn_signs):
            if pt > 0:
                self.assertTrue(s_sign >= 0)
            else:
                self.assertTrue(s_sign <= 0)

    def test_simulation_run_no_stdp(self):
        N, K = 100, 6
        net = build_creature_network(N, K, fi=0.2, seed=42)
        T = 50
        result = run_creature_simulation(net, T, dale=True, stdp_on=False, noise_scale=0.15, seed=42)

        self.assertEqual(result['spikes'].shape, (T, N))
        self.assertEqual(len(result['weight_history']), T)
        self.assertEqual(len(result['final_energy']), N)
        self.assertEqual(len(result['final_fatigue']), N)
        self.assertEqual(len(result['final_weight']), N * K)

        # Since STDP is OFF, weights should be exactly unchanged
        np.testing.assert_array_equal(net['weight'], result['final_weight'])

    def test_simulation_run_with_stdp_and_dale(self):
        N, K = 50, 4
        net = build_creature_network(N, K, fi=0.2, seed=10)
        T = 100
        # Use high noise to elicit spikes and STDP updates
        result = run_creature_simulation(net, T, dale=True, stdp_on=True, noise_scale=1.5, seed=10)

        # Weights should change
        self.assertFalse(np.array_equal(net['weight'], result['final_weight']))

        # Dale's law must hold on the final weights as well
        pre_types = net['types'][net['pre']]
        final_weight = result['final_weight']
        for pt, w in zip(pre_types, final_weight):
            if pt > 0:
                self.assertGreaterEqual(w, 0.0)
                self.assertLessEqual(w, W_MAX)
            else:
                self.assertLessEqual(w, 0.0)
                self.assertGreaterEqual(w, -W_MAX)

    def test_metrics(self):
        T, N = 100, 50
        spikes = np.zeros((T, N), dtype=bool)
        # Put some spikes
        spikes[10, :] = True
        spikes[20, 0:10] = True
        spikes[30, 0:10] = True

        f_stats = firing_stats(spikes)
        self.assertIn('mean_pop_rate', f_stats)
        self.assertIn('pop_rate_cv', f_stats)
        self.assertIn('per_neuron_rate_std', f_stats)

        # Avalanches
        a_stats = avalanche_stats(spikes)
        self.assertIn('num_avalanches', a_stats)
        self.assertIn('mean_avalanche_size', a_stats)
        self.assertIn('max_avalanche_size', a_stats)
        # Since spikes happened at discrete non-contiguous time steps:
        # Step 10 has spikes -> avalanche 1 (size 1)
        # Step 20 has spikes -> avalanche 2 (size 1)
        # Step 30 has spikes -> avalanche 3 (size 1)
        self.assertEqual(a_stats['num_avalanches'], 3)
        self.assertEqual(a_stats['mean_avalanche_size'], 1.0)
        self.assertEqual(a_stats['max_avalanche_size'], 1)

        # Weight drift
        net = {'weight': np.array([1.0, -1.0])}
        result = {'final_weight': np.array([1.5, -0.5])}
        drift = weight_drift(net, result)
        self.assertEqual(drift['init_mean_abs_weight'], 1.0)
        self.assertEqual(drift['final_mean_abs_weight'], 1.0)
        self.assertEqual(drift['weight_drift'], 0.0)

        # Metabolic stats
        res_met = {
            'final_energy': np.array([0.9, 0.8]),
            'final_fatigue': np.array([0.1, 0.2])
        }
        met = metabolic_stats(res_met)
        self.assertAlmostEqual(met['mean_final_energy'], 0.85)
        self.assertAlmostEqual(met['mean_final_fatigue'], 0.15)


if __name__ == '__main__':
    unittest.main()
