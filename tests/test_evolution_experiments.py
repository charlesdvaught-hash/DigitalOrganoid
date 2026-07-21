import unittest
import numpy as np
from organoid_simulator.creature_embodied import build_creature_network
from organoid_simulator.snapshot_utils import capture_snapshot, reconstruct_organism
from organoid_simulator.creature_metrics import (
    calculate_network_metrics,
    calculate_behavioral_metrics,
    firing_stats,
    metabolic_stats
)
from organoid_simulator.evolution_experiments import (
    experiment_a_structural_combination,
    experiment_b_developmental_inheritance,
    experiment_c_co_culture_fusion,
    experiment_d_neural_transplantation,
    experiment_e_trait_crossover,
    experiment_f_experience_transfer,
    experiment_g_multistage_pipeline
)

class TestEvolutionExperiments(unittest.TestCase):
    def setUp(self):
        self.p1 = build_creature_network(200, 4, fi=0.2, seed=12)
        self.p2 = build_creature_network(200, 4, fi=0.2, seed=34)

    def test_experiment_a(self):
        res = experiment_a_structural_combination(self.p1, self.p2, seed=42)
        self.assertEqual(res['N'], 200)
        # Verify Dale's law holds on child SNN weights
        pre_types = res['types'][res['pre']]
        for pt, w in zip(pre_types, res['weight']):
            if pt > 0:
                self.assertGreaterEqual(w, 0.0)
            else:
                self.assertLessEqual(w, 0.0)

    def test_experiment_b(self):
        res = experiment_b_developmental_inheritance(self.p1, self.p2, seed=42)
        self.assertEqual(res['N'], 200)
        self.assertEqual(res['K'], 4)

    def test_experiment_c(self):
        res = experiment_c_co_culture_fusion(self.p1, self.p2, seed=42)
        self.assertEqual(res['N'], 400) # double the size

    def test_experiment_d(self):
        res = experiment_d_neural_transplantation(self.p1, self.p2, transplant_frac=0.10, seed=42)
        self.assertEqual(res['N'], 200)

    def test_experiment_e(self):
        res = experiment_e_trait_crossover(self.p1, self.p2, seed=42)
        self.assertEqual(res['N'], 200)
        self.assertIn('dev_params', res)

    def test_experiment_f(self):
        res = experiment_f_experience_transfer(self.p1, seed=42)
        self.assertEqual(res['N'], 200)

    def test_experiment_g(self):
        res = experiment_g_multistage_pipeline(self.p1, self.p2, seed=42)
        self.assertEqual(res['N'], 200)

    def test_snapshot_integration(self):
        snapshot = capture_snapshot(self.p1, label='Parent-1')
        self.assertEqual(snapshot['label'], 'Parent-1')

        reconstructed = reconstruct_organism(snapshot)
        self.assertEqual(reconstructed['N'], self.p1['N'])
        np.testing.assert_allclose(reconstructed['weight'], self.p1['weight'], atol=1e-5)

    def test_metrics_calculations(self):
        net_m = calculate_network_metrics(self.p1)
        self.assertIn('density', net_m)
        self.assertIn('spatial_modularity', net_m)
        self.assertGreaterEqual(net_m['density'], 0.0)
        self.assertLessEqual(net_m['density'], 1.0)

        behav_m = calculate_behavioral_metrics({'food_eaten': 5, 'survival_steps': 1500, 'distance_traveled': 10.5}, 1500)
        self.assertEqual(behav_m['food_eaten'], 5)
        self.assertEqual(behav_m['foraging_efficiency'], 5 / 1500)

if __name__ == '__main__':
    unittest.main()
