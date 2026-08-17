import unittest
import numpy as np

from organoid_simulator.creature_embodied import (
    build_creature_network, run_creature_simulation, POOLS
)


class TestFlockeMechanisms(unittest.TestCase):

    def test_bilateral_symmetric_initialization(self):
        N, K = 200, 6
        net = build_creature_network(N, K, fi=0.2, seed=42, bilateral_symmetric=True)
        sL0, sL1 = POOLS['sL']; sR0, sR1 = POOLS['sR']
        mL0, mL1 = POOLS['mL']; mR0, mR1 = POOLS['mR']
        # Check sagittal mirror symmetry of left/right sensor and motor pools
        np.testing.assert_allclose(net['px'][sL0:sL1], -net['px'][sR0:sR1])
        np.testing.assert_allclose(net['py'][sL0:sL1], net['py'][sR0:sR1])
        np.testing.assert_allclose(net['pz'][sL0:sL1], net['pz'][sR0:sR1])

        np.testing.assert_allclose(net['px'][mL0:mL1], -net['px'][mR0:mR1])
        np.testing.assert_allclose(net['py'][mL0:mL1], net['py'][mR0:mR1])
        np.testing.assert_allclose(net['pz'][mL0:mL1], net['pz'][mR0:mR1])

    def test_cpg_enabled_simulation(self):
        N, K = 200, 6
        net = build_creature_network(N, K, fi=0.2, seed=7)
        res = run_creature_simulation(net, 300, cpg_enabled=True, seed=7)
        b = res['behavior']
        self.assertGreater(b['distance_traveled'], 0.0)
        self.assertGreater(b['survival_steps'], 0)

    def test_cerebellar_forward_model(self):
        N, K = 200, 6
        net = build_creature_network(N, K, fi=0.2, seed=123)
        res = run_creature_simulation(net, 300, cerebellar_enabled=True, seed=123)
        self.assertIn('behavior', res)
        self.assertGreater(res['behavior']['survival_steps'], 0)

    def test_multichannel_neuromodulation(self):
        N, K = 200, 6
        net = build_creature_network(N, K, fi=0.2, seed=99)
        res = run_creature_simulation(net, 300, multichannel_neuromod=True, seed=99)
        self.assertIn('behavior', res)
        self.assertGreater(res['behavior']['survival_steps'], 0)

    def test_all_flocke_mechanisms_combined(self):
        N, K = 200, 6
        net = build_creature_network(N, K, fi=0.2, seed=42, bilateral_symmetric=True)
        res = run_creature_simulation(
            net, 500, cpg_enabled=True, cerebellar_enabled=True,
            multichannel_neuromod=True, seed=42
        )
        b = res['behavior']
        self.assertGreater(b['distance_traveled'], 0.0)


if __name__ == '__main__':
    unittest.main()
